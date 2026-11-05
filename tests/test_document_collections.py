"""Synthetic Appendix A reconstruction/validation checks; no real datasets read."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.document_collections import (
    COLLECTION_COLUMNS,
    TOPIC_PATHS,
    build_collections,
    clean_articles,
    compare_collections,
)


def url(year, path, slug):
    return f"https://www.nytimes.com/{year}/01/01/{path}/{slug}.html"


class DocumentCollectionTests(unittest.TestCase):
    def test_cleaning_order_threshold_and_input_preservation(self):
        def long_text(last):
            return "one two three four five six seven eight nine " + last

        articles = pd.DataFrame(
            {
                "text": [
                    "too short", long_text("alpha"), long_text("alpha"),
                    long_text("beta"), long_text("gamma"), long_text("delta"), None,
                ],
                "web_url": ["a", "a", "b", "a", "b", "c", "d"],
            }
        )
        before = articles.copy(deep=True)
        cleaned, report = clean_articles(articles)

        self.assertEqual(cleaned.index.tolist(), [1, 4, 5])
        self.assertEqual(report["removed"].tolist(), [2, 1, 1])
        self.assertEqual(report["remaining"].tolist(), [5, 4, 3])
        self.assertEqual(
            report.index.tolist(), ["short_text", "duplicate_text", "duplicate_url"]
        )
        assert_frame_equal(articles, before)

    def test_boundaries_timestamp_url_sorting_ids_and_publication_year(self):
        articles = pd.DataFrame(
            {
                "pub_date": [
                    "2025-12-31T23:59:59", "2014-12-31T23:59:59",
                    "2015-01-01", "2005-01-01T01:00:00",
                    "2005-01-01T00:00:00", "2005-01-01T00:00:00",
                    "2004-12-31", "2026-01-01",
                ],
                "web_url": [
                    url(2025, "science", "last"), url(2014, "science", "last"),
                    url(2004, "science", "different-url-year"),
                    url(2005, "science", "later-time"), url(2005, "science", "z"),
                    url(2005, "science", "a"), url(2004, "science", "excluded"),
                    url(2026, "science", "excluded"),
                ],
            },
            index=[4] * 8,
        )
        before = articles.copy(deep=True)
        collections = build_collections(articles)
        c1, c2 = collections["C1"], collections["C2"]

        self.assertEqual(c1["web_url"].tolist(), articles["web_url"].iloc[[5, 4, 3, 1]].tolist())
        self.assertEqual(c2["web_url"].tolist(), articles["web_url"].iloc[[2, 0]].tolist())
        self.assertEqual(c1["docno"].tolist(), [f"C1-{i:06d}" for i in range(4)])
        self.assertEqual(c2["docno"].tolist(), ["C2-000000", "C2-000001"])
        self.assertEqual(c1["year"].tolist(), [2005, 2005, 2005, 2014])
        self.assertEqual(c2["year"].tolist(), [2015, 2025])
        self.assertEqual(c1.columns.tolist(), COLLECTION_COLUMNS)
        self.assertEqual(c2.columns.tolist(), COLLECTION_COLUMNS)
        assert_frame_equal(articles, before)

    def test_all_table_2_topics_and_both_books_paths(self):
        # These expected pairs come from Table 2, not the official topics.tsv.
        pairs = [
            ("world/africa", "africa"), ("arts/design", "art"),
            ("world/asia", "asia"), ("sports/baseball", "baseball"),
            ("sports/basketball", "basketball"), ("books", "books"),
            ("books/review", "books"), ("arts/dance", "dance"),
            ("business/economy", "economy"), ("world/europe", "europe"),
            ("fashion", "fashion"), ("dining", "food"),
            ("sports/football", "football"), ("sports/golf", "golf"),
            ("health", "health"), ("sports/hockey", "hockey"),
            ("world/americas", "latin america"), ("business/media", "media"),
            ("world/middleeast", "middle east"), ("movies", "movies"),
            ("arts/music", "music"), ("sports/olympics", "olympics"),
            ("your-money", "personal finance"), ("us/politics", "politics"),
            ("realestate", "real estate"), ("science", "science"),
            ("sports/soccer", "soccer"), ("technology", "technology"),
            ("arts/television", "television"), ("sports/tennis", "tennis"),
            ("theater", "theater"), ("travel", "travel"),
            ("fashion/weddings", "weddings"),
        ]
        articles = pd.DataFrame(
            {
                "pub_date": ["2010-01-01"] * len(pairs),
                "web_url": [url(2010, path, "story") for path, _ in pairs],
            }
        )
        result = build_collections(articles)["C1"]
        self.assertEqual(len(TOPIC_PATHS), 33)
        self.assertEqual(len(set(result["topic"])), 32)
        self.assertEqual(
            result.set_index("web_url")["topic"].to_dict(),
            {url(2010, path, "story"): topic for path, topic in pairs},
        )

    def test_exact_url_and_path_matching_and_invalid_dates(self):
        valid = url(2010, "science", "valid")
        urls = [
            valid, valid.replace(".html", ""), valid.replace("https:", "http:"),
            valid.replace("www.nytimes.com", "dealbook.nytimes.com"),
            url(2010, "sports/baseball/minor-league", "story"),
            url(2010, "world/european", "story"),
            url(2010, "Science", "story"),
            valid.replace("/2010/", "/10/"),
            "https://www.nytimes.com/2010/01/01/science/", None,
            url(2010, "science", "missing-date"),
        ]
        articles = pd.DataFrame(
            {"web_url": urls, "pub_date": ["2010-01-01"] * 10 + [None]}
        )
        result = build_collections(articles)
        self.assertEqual(set(result["C1"]["web_url"]), set(urls[:2]))
        self.assertTrue(result["C2"].empty)

    def test_empty_input_and_duplicate_url_rejection(self):
        empty = pd.DataFrame(columns=["pub_date", "web_url"])
        for frame in build_collections(empty).values():
            self.assertTrue(frame.empty)
            self.assertEqual(frame.columns.tolist(), COLLECTION_COLUMNS)
        duplicate = pd.DataFrame(
            {"pub_date": ["2010-01-01"] * 2, "web_url": [url(2010, "science", "a")] * 2}
        )
        with self.assertRaisesRegex(ValueError, "duplicate URLs"):
            build_collections(duplicate)

    def test_comparison_counts_and_details(self):
        official_c1 = pd.DataFrame(
            [
                ["C1-000000", "url-a", 2005, "science"],
                ["C1-000001", "url-b", 2005, "science"],
                ["C1-000002", "url-c", 2005, "science"],
            ],
            columns=COLLECTION_COLUMNS,
        )
        rebuilt = official_c1.iloc[[0, 2]].copy()
        rebuilt.loc[2, ["docno", "year", "topic"]] = ["C1-000010", 2006, "art"]
        rebuilt.loc[3] = ["C1-000003", "url-extra", 2005, "science"]
        empty = official_c1.iloc[:0].copy()
        before = rebuilt.copy(deep=True)
        with TemporaryDirectory() as temporary:
            official_c1.to_csv(Path(temporary) / "collection_P1.tsv", sep="\t", index=False)
            empty.to_csv(Path(temporary) / "collection_P2.tsv", sep="\t", index=False)
            summary, details = compare_collections({"C1": rebuilt, "C2": empty}, temporary)
        self.assertEqual(summary.loc["C1", ["missing", "extra", "mismatched"]].tolist(), [1, 1, 1])
        self.assertEqual(summary.loc["C1", "total_differences"], 3)
        self.assertEqual(summary.loc["C2", "total_differences"], 0)
        self.assertEqual(details["C1"]["missing"]["web_url"].tolist(), ["url-b"])
        self.assertEqual(details["C1"]["extra"]["web_url"].tolist(), ["url-extra"])
        self.assertEqual(details["C1"]["mismatched"]["web_url"].tolist(), ["url-c"])
        self.assertEqual(
            details["C1"]["mismatched"]["mismatched_fields"].tolist(),
            ["docno, year, topic"],
        )
        assert_frame_equal(rebuilt, before)

    def test_identical_records_in_different_row_order_have_zero_differences(self):
        original = pd.DataFrame(
            [["C1-000000", "a", 2005, "science"], ["C1-000001", "b", 2006, "art"]],
            columns=COLLECTION_COLUMNS,
        )
        empty = original.iloc[:0].copy()
        with TemporaryDirectory() as temporary:
            original.to_csv(Path(temporary) / "collection_P1.tsv", sep="\t", index=False)
            empty.to_csv(Path(temporary) / "collection_P2.tsv", sep="\t", index=False)
            summary, _ = compare_collections(
                {"C1": original.iloc[::-1], "C2": empty}, temporary
            )
        self.assertEqual(summary["total_differences"].tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main()
