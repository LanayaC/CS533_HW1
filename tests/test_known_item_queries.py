"""Synthetic MD5 selection/validation checks; no real datasets are read."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from src.known_item_eligibility import check_known_item_eligibility
from src.known_item_queries import (
    QUERY_COLUMNS,
    _url_hash,
    compare_known_item_queries,
    select_known_item_queries,
)


class KnownItemQueryTests(unittest.TestCase):
    def setUp(self):
        self.urls = [
            "https://www.nytimes.com/2008/09/10/arts/music/orchestra-opens-new-season.html",
            "https://www.nytimes.com/2010/03/15/science/researchers-discover-new-planet.html",
            "https://www.nytimes.com/2012/06/22/sports/baseball/the-yankees-win-game-3.html",
            "https://www.nytimes.com/2018/04/01/science/a-new-day-in-the-city.html",
            "https://www.nytimes.com/2020/04/01/science/teachers-study-new-stars.html",
            "https://www.nytimes.com/2025/04/01/arts/music/singers-open-city-festival.html",
        ]
        self.collections = {
            "C1": pd.DataFrame(
                {
                    "docno": ["C1-000000", "C1-000001", "C1-000002"],
                    "web_url": self.urls[:3],
                    "year": [2008, 2010, 2012],
                }
            ),
            "C2": pd.DataFrame(
                {
                    "docno": ["C2-000000", "C2-000001", "C2-000002"],
                    "web_url": self.urls[3:],
                    "year": [2018, 2020, 2025],
                }
            ),
        }
        self.eligibility = check_known_item_eligibility(pd.DataFrame({"web_url": self.urls}))
        self.eligibility.index = [7] * len(self.eligibility)

    def test_complete_url_hashes_against_fixed_reference_values(self):
        self.assertEqual(_url_hash(self.urls[0]), 54_269_876_349_314)
        self.assertEqual(_url_hash(self.urls[1]), 217_066_328_886_323)
        self.assertEqual(_url_hash(self.urls[2]), 74_766_887_978_872)

    def test_selection_order_ids_query_text_years_and_input_preservation(self):
        originals = {name: frame.copy(deep=True) for name, frame in self.collections.items()}
        eligibility_before = self.eligibility.copy(deep=True)
        queries = select_known_item_queries(self.collections, self.eligibility, n_queries=2)

        expected_c1 = pd.DataFrame(
            {
                "qid": ["KI1-0001", "KI1-0002"],
                "query": ["orchestra opens new season", "the yankees win game"],
                "docno": ["C1-000000", "C1-000002"],
                "year": [2008, 2012],
            }
        )
        assert_frame_equal(queries["C1"], expected_c1)
        self.assertEqual(queries["C2"]["qid"].tolist(), ["KI2-0001", "KI2-0002"])
        self.assertEqual(queries["C2"].columns.tolist(), QUERY_COLUMNS)
        expected_years = self.collections["C2"].set_index("docno")["year"]
        self.assertEqual(
            queries["C2"]["year"].tolist(),
            queries["C2"]["docno"].map(expected_years).tolist(),
        )
        for name, frame in self.collections.items():
            assert_frame_equal(frame, originals[name])
        assert_frame_equal(self.eligibility, eligibility_before)

    def test_only_eligible_collection_members_are_candidates(self):
        outside = [
            "https://www.nytimes.com/2020/06/22/unlisted/orchestra-opens-new-season.html",
            "https://www.nytimes.com/2020/06/22/unlisted/scientists-discover-distant-planets.html",
        ]
        eligibility = check_known_item_eligibility(
            pd.DataFrame({"web_url": self.urls + outside})
        )
        queries = select_known_item_queries(self.collections, eligibility, n_queries=2)

        # A duplicate slug outside C1/C2 makes the orchestra article ineligible.
        self.assertEqual(queries["C1"]["docno"].tolist(), ["C1-000002", "C1-000001"])
        for name, frame in queries.items():
            self.assertTrue(frame["docno"].isin(self.collections[name]["docno"]).all())

    def test_query_text_retains_stop_words_and_single_letter_tokens(self):
        queries = select_known_item_queries(self.collections, self.eligibility, n_queries=3)
        query = queries["C2"].set_index("docno").loc["C2-000000", "query"]
        self.assertEqual(query, "a new day in the city")

    def test_equal_hashes_preserve_collection_order(self):
        with patch("src.known_item_queries._url_hash", return_value=7):
            queries = select_known_item_queries(self.collections, self.eligibility, n_queries=2)
        for name, frame in queries.items():
            self.assertEqual(frame["docno"].tolist(), self.collections[name]["docno"].head(2).tolist())

    def test_insufficient_candidates_and_invalid_selection_sizes_raise(self):
        with self.assertRaisesRegex(ValueError, "1000 are required"):
            select_known_item_queries(self.collections, self.eligibility)
        for value in [0, -1, 1.5, True]:
            with self.subTest(n_queries=value), self.assertRaises(ValueError):
                select_known_item_queries(self.collections, self.eligibility, n_queries=value)

    def test_missing_eligibility_record_is_not_silently_treated_as_ineligible(self):
        with self.assertRaisesRegex(ValueError, "does not cover"):
            select_known_item_queries(self.collections, self.eligibility.iloc[1:], n_queries=1)

    def test_comparison_missing_extra_and_multi_field_mismatch_counts(self):
        official = pd.DataFrame(
            [
                ["KI1-0001", "first query", "C1-000000", 2005],
                ["KI1-0002", "second query", "C1-000001", 2006],
                ["KI1-0003", "third query", "C1-000002", 2007],
            ],
            columns=QUERY_COLUMNS,
        )
        selected = official.iloc[[0, 2]].copy()
        selected.loc[2, ["query", "docno", "year"]] = ["changed query", "C1-000010", 2008]
        selected.loc[3] = ["KI1-0004", "extra query", "C1-000003", 2009]
        empty = official.iloc[:0].copy()
        before = selected.copy(deep=True)
        with TemporaryDirectory() as temporary:
            official.to_csv(Path(temporary) / "queries_ki_P1.tsv", sep="\t", index=False)
            empty.to_csv(Path(temporary) / "queries_ki_P2.tsv", sep="\t", index=False)
            summary, details = compare_known_item_queries(
                {"C1": selected, "C2": empty}, temporary
            )
        self.assertEqual(summary.loc["C1", ["missing", "extra", "mismatched"]].tolist(), [1, 1, 1])
        self.assertEqual(summary.loc["C1", "total_differences"], 3)
        self.assertEqual(summary.loc["C2", "total_differences"], 0)
        self.assertEqual(details["C1"]["missing"]["qid"].tolist(), ["KI1-0002"])
        self.assertEqual(details["C1"]["extra"]["qid"].tolist(), ["KI1-0004"])
        self.assertEqual(details["C1"]["mismatched"]["qid"].tolist(), ["KI1-0003"])
        self.assertEqual(
            details["C1"]["mismatched"]["mismatched_fields"].tolist(),
            ["query, docno, year"],
        )
        assert_frame_equal(selected, before)

    def test_identical_records_in_different_row_order_have_zero_differences(self):
        queries = select_known_item_queries(self.collections, self.eligibility, n_queries=2)
        with TemporaryDirectory() as temporary:
            queries["C1"].to_csv(Path(temporary) / "queries_ki_P1.tsv", sep="\t", index=False)
            queries["C2"].to_csv(Path(temporary) / "queries_ki_P2.tsv", sep="\t", index=False)
            summary, _ = compare_known_item_queries(
                {name: frame.iloc[::-1] for name, frame in queries.items()}, temporary
            )
        self.assertEqual(summary["total_differences"].tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()
