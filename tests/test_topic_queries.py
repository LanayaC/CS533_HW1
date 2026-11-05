"""Synthetic tests for the shared Table 2 topic definitions."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from src.topic_queries import TOPIC_COLUMNS, build_topic_queries, compare_topic_queries


class TopicQueryTests(unittest.TestCase):
    def setUp(self):
        self.official = pd.DataFrame(
            [
                ["T01", "africa", "world/africa"],
                ["T06", "books", "books books/review"],
                ["T32", "weddings", "fashion/weddings"],
            ],
            columns=TOPIC_COLUMNS,
        )

    def test_build_has_all_32_shared_ids_without_reading_official_files(self):
        with patch("src.topic_queries.pd.read_csv", side_effect=AssertionError("No official input")):
            p1_topics = build_topic_queries()
            p2_topics = build_topic_queries()
        self.assertEqual(p1_topics.columns.tolist(), TOPIC_COLUMNS)
        self.assertEqual(p1_topics["tid"].tolist(), [f"T{i:02d}" for i in range(1, 33)])
        self.assertEqual(p1_topics["query"].nunique(), 32)
        assert_frame_equal(p1_topics, p2_topics)
        indexed = p1_topics.set_index("tid")
        self.assertEqual(indexed.loc["T06", "paths"], "books books/review")
        self.assertEqual(indexed.loc["T16", "query"], "latin america")
        self.assertEqual(indexed.loc["T18", "paths"], "world/middleeast")
        self.assertEqual(indexed.loc["T22", "query"], "personal finance")
        self.assertEqual(indexed.loc["T24", "paths"], "realestate")
        self.assertEqual(indexed.loc["T32", "query"], "weddings")

    def test_missing_extra_and_multi_field_mismatch_count_once(self):
        reconstructed = self.official.iloc[[0, 1]].copy()
        reconstructed.loc[1, ["query", "paths"]] = ["wrong books name", "books/wrong"]
        reconstructed.loc[2] = ["T99", "extra", "extra/path"]
        before = reconstructed.copy(deep=True)
        with TemporaryDirectory() as temporary:
            self.official.to_csv(Path(temporary) / "topics.tsv", sep="\t", index=False)
            summary, details = compare_topic_queries(reconstructed, temporary)
        self.assertEqual(summary.loc["topics", ["missing", "extra", "mismatched"]].tolist(), [1, 1, 1])
        self.assertEqual(summary.loc["topics", "total_differences"], 3)
        self.assertEqual(details["missing"]["tid"].tolist(), ["T32"])
        self.assertEqual(details["extra"]["tid"].tolist(), ["T99"])
        self.assertEqual(details["mismatched"]["tid"].tolist(), ["T06"])
        self.assertEqual(details["mismatched"]["mismatched_fields"].tolist(), ["query, paths"])
        assert_frame_equal(reconstructed, before)

    def test_row_order_and_path_listing_order_do_not_change_definitions(self):
        reconstructed = self.official.iloc[::-1].copy()
        reconstructed.loc[1, "paths"] = "books/review   books"
        with TemporaryDirectory() as temporary:
            self.official.to_csv(Path(temporary) / "topics.tsv", sep="\t", index=False)
            summary, details = compare_topic_queries(reconstructed, temporary)
        self.assertEqual(summary.loc["topics", "total_differences"], 0)
        self.assertTrue(all(frame.empty for frame in details.values()))

    def test_query_case_and_exact_path_spelling_are_checked(self):
        reconstructed = self.official.copy()
        reconstructed.loc[0, "query"] = "Africa"
        reconstructed.loc[2, "paths"] = "fashion/weddings/"
        with TemporaryDirectory() as temporary:
            self.official.to_csv(Path(temporary) / "topics.tsv", sep="\t", index=False)
            summary, details = compare_topic_queries(reconstructed, temporary)
        self.assertEqual(summary.loc["topics", "mismatched"], 2)
        self.assertEqual(details["mismatched"]["mismatched_fields"].tolist(), ["query", "paths"])

    def test_duplicate_ids_missing_columns_and_missing_values_are_rejected(self):
        duplicate = self.official.copy()
        duplicate.loc[1, "tid"] = "T01"
        missing_value = self.official.copy()
        missing_value.loc[0, "query"] = None
        blank_paths = self.official.copy()
        blank_paths.loc[0, "paths"] = "  "
        for invalid in [duplicate, missing_value, blank_paths, self.official.drop(columns="paths")]:
            with self.subTest(columns=invalid.columns.tolist()), self.assertRaises(ValueError):
                compare_topic_queries(invalid, "unused")

    def test_invalid_official_definitions_are_rejected(self):
        official = self.official.copy()
        official.loc[1, "tid"] = "T01"
        with TemporaryDirectory() as temporary:
            official.to_csv(Path(temporary) / "topics.tsv", sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "duplicate topic IDs"):
                compare_topic_queries(self.official, temporary)


if __name__ == "__main__":
    unittest.main()
