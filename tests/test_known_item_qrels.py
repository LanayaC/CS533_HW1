"""Synthetic known-item qrels checks; no real datasets are read."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.known_item_qrels import (
    QRELS_COLUMNS,
    build_known_item_qrels,
    compare_known_item_qrels,
    format_known_item_qrels,
)


class KnownItemQrelsTests(unittest.TestCase):
    def setUp(self):
        self.queries = {
            "C1": pd.DataFrame(
                {
                    "qid": ["KI1-0002", "KI1-0001"],
                    "query": ["second query", "first query"],
                    "docno": ["C1-000005", "C1-000003"],
                    "year": [2010, 2005],
                },
                index=[7, 7],
            ),
            "C2": pd.DataFrame(
                {"qid": ["KI2-0001"], "query": ["new query"], "docno": ["C2-000009"], "year": [2020]}
            ),
        }

    def test_one_judgment_per_query_correct_targets_order_and_no_mutation(self):
        before = {name: frame.copy(deep=True) for name, frame in self.queries.items()}
        judgments = build_known_item_qrels(self.queries)

        expected = pd.DataFrame(
            {
                "qid": ["KI1-0002", "KI1-0001"],
                "iteration": [0, 0],
                "docno": ["C1-000005", "C1-000003"],
                "relevance": [1, 1],
            }
        )
        assert_frame_equal(judgments["C1"], expected)
        self.assertEqual(judgments["C2"]["docno"].tolist(), ["C2-000009"])
        for name, frame in judgments.items():
            self.assertEqual(len(frame), len(self.queries[name]))
            self.assertEqual(frame.columns.tolist(), QRELS_COLUMNS)
            self.assertTrue(frame["iteration"].eq(0).all())
            self.assertTrue(frame["relevance"].eq(1).all())
            assert_frame_equal(self.queries[name], before[name])

    def test_exact_headerless_space_separated_trec_format(self):
        judgments = build_known_item_qrels(self.queries)
        self.assertEqual(
            format_known_item_qrels(judgments["C1"]),
            "KI1-0002 0 C1-000005 1\nKI1-0001 0 C1-000003 1\n",
        )
        self.assertEqual(
            format_known_item_qrels(judgments["C2"]), "KI2-0001 0 C2-000009 1\n"
        )

    def test_empty_query_frames_produce_empty_qrels_and_text(self):
        empty = {name: frame.iloc[:0].copy() for name, frame in self.queries.items()}
        for frame in build_known_item_qrels(empty).values():
            self.assertTrue(frame.empty)
            self.assertEqual(frame.columns.tolist(), QRELS_COLUMNS)
            self.assertEqual(format_known_item_qrels(frame), "")

    def test_duplicate_queries_missing_targets_and_whitespace_ids_are_rejected(self):
        for column, values in [
            ("qid", ["KI1-0001", "KI1-0001"]),
            ("docno", [None, "C1-000003"]),
            ("docno", ["C1 000005", "C1-000003"]),
        ]:
            queries = {name: frame.copy() for name, frame in self.queries.items()}
            queries["C1"][column] = values
            with self.subTest(column=column, values=values), self.assertRaises(ValueError):
                build_known_item_qrels(queries)

    def test_serializer_rejects_non_known_item_iteration_or_relevance(self):
        judgments = build_known_item_qrels(self.queries)["C1"]
        for column, value in [("iteration", 1), ("relevance", 0)]:
            altered = judgments.copy()
            altered[column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                format_known_item_qrels(altered)

    def test_missing_extra_and_multi_field_mismatch_count_once(self):
        official = pd.DataFrame(
            [
                ["KI1-0001", 0, "C1-000000", 1],
                ["KI1-0002", 0, "C1-000001", 1],
                ["KI1-0003", 0, "C1-000002", 1],
            ],
            columns=QRELS_COLUMNS,
        )
        reconstructed = official.iloc[[0, 2]].copy()
        reconstructed.loc[2, ["iteration", "docno", "relevance"]] = [1, "C1-000010", 0]
        reconstructed.loc[3] = ["KI1-0004", 0, "C1-000003", 1]
        empty = official.iloc[:0].copy()
        before = reconstructed.copy(deep=True)
        with TemporaryDirectory() as temporary:
            official.to_csv(Path(temporary) / "qrels_ki_P1.txt", sep=" ", header=False, index=False)
            (Path(temporary) / "qrels_ki_P2.txt").write_text("", encoding="utf-8")
            summary, details = compare_known_item_qrels(
                {"C1": reconstructed, "C2": empty}, temporary
            )
        self.assertEqual(summary.loc["C1", ["missing", "extra", "mismatched"]].tolist(), [1, 1, 1])
        self.assertEqual(summary.loc["C1", "total_differences"], 3)
        self.assertEqual(summary.loc["C2", "total_differences"], 0)
        self.assertEqual(details["C1"]["missing"]["qid"].tolist(), ["KI1-0002"])
        self.assertEqual(details["C1"]["extra"]["qid"].tolist(), ["KI1-0004"])
        self.assertEqual(details["C1"]["mismatched"]["qid"].tolist(), ["KI1-0003"])
        self.assertEqual(
            details["C1"]["mismatched"]["mismatched_fields"].tolist(),
            ["iteration, docno, relevance"],
        )
        assert_frame_equal(reconstructed, before)

    def test_matching_records_ignore_row_order_and_field_whitespace(self):
        judgments = build_known_item_qrels(self.queries)
        with TemporaryDirectory() as temporary:
            for name, filename in [("C1", "qrels_ki_P1.txt"), ("C2", "qrels_ki_P2.txt")]:
                text = format_known_item_qrels(judgments[name]).replace(" ", "\t  ")
                (Path(temporary) / filename).write_text(text, encoding="utf-8")
            summary, _ = compare_known_item_qrels(
                {name: frame.iloc[::-1] for name, frame in judgments.items()}, temporary
            )
        self.assertEqual(summary["total_differences"].tolist(), [0, 0])
        self.assertEqual(summary["reconstructed_count"].tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()
