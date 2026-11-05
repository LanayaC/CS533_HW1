"""Small synthetic Appendix A.5 checks; no real datasets are read."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.document_collections import build_collections
from src.topic_qrels import (
    QRELS_COLUMNS,
    TOPIC_IDS,
    build_topic_qrels,
    compare_topic_qrels,
    format_topic_qrels,
)


class TopicQrelsTests(unittest.TestCase):
    def setUp(self):
        self.collections = {
            "C1": pd.DataFrame(
                {
                    "docno": ["C1-000002", "C1-000000", "C1-000001"],
                    "topic": ["science", "art", "science"],
                },
                index=[7, 7, 7],
            ),
            "C2": pd.DataFrame(
                {"docno": ["C2-000000", "C2-000001"], "topic": ["books", "weddings"]}
            ),
        }

    def test_one_judgment_per_article_repeated_topics_order_and_no_mutation(self):
        before = {name: frame.copy(deep=True) for name, frame in self.collections.items()}
        judgments = build_topic_qrels(self.collections)

        expected_c1 = pd.DataFrame(
            {
                "tid": ["T25", "T02", "T25"],
                "iteration": [0, 0, 0],
                "docno": ["C1-000002", "C1-000000", "C1-000001"],
                "relevance": [1, 1, 1],
            }
        )
        assert_frame_equal(judgments["C1"], expected_c1)
        self.assertEqual(judgments["C2"]["tid"].tolist(), ["T06", "T32"])
        for name, frame in judgments.items():
            self.assertEqual(len(frame), len(self.collections[name]))
            self.assertEqual(frame.columns.tolist(), QRELS_COLUMNS)
            self.assertTrue(frame["docno"].is_unique)
            assert_frame_equal(self.collections[name], before[name])

    def test_32_table_2_ids_and_multiword_topic_labels(self):
        topics = ["africa", "latin america", "middle east", "personal finance", "real estate", "weddings"]
        frame = pd.DataFrame(
            {"docno": [f"C1-{i:06d}" for i in range(len(topics))], "topic": topics}
        )
        result = build_topic_qrels({"C1": frame, "C2": frame.iloc[:0]})["C1"]

        self.assertEqual(len(TOPIC_IDS), 32)
        self.assertEqual(set(TOPIC_IDS.values()), {f"T{i:02d}" for i in range(1, 33)})
        self.assertEqual(result["tid"].tolist(), ["T01", "T16", "T18", "T22", "T24", "T32"])

    def test_both_books_paths_map_to_the_same_topic_id(self):
        articles = pd.DataFrame(
            {
                "pub_date": ["2010-01-01", "2010-01-02"],
                "web_url": [
                    "https://www.nytimes.com/2010/01/01/books/new-book-story.html",
                    "https://www.nytimes.com/2010/01/02/books/review/another-book-story.html",
                ],
            }
        )
        judgments = build_topic_qrels(build_collections(articles))
        self.assertEqual(judgments["C1"]["tid"].tolist(), ["T06", "T06"])
        self.assertEqual(len(judgments["C1"]), 2)
        self.assertTrue(judgments["C2"].empty)

    def test_exact_trec_format_and_empty_collections(self):
        judgments = build_topic_qrels(self.collections)
        self.assertEqual(
            format_topic_qrels(judgments["C1"]),
            "T25 0 C1-000002 1\nT02 0 C1-000000 1\nT25 0 C1-000001 1\n",
        )
        empty = {name: frame.iloc[:0] for name, frame in self.collections.items()}
        for frame in build_topic_qrels(empty).values():
            self.assertTrue(frame.empty)
            self.assertEqual(frame.columns.tolist(), QRELS_COLUMNS)
            self.assertEqual(format_topic_qrels(frame), "")

    def test_unknown_missing_topics_and_duplicate_document_ids_are_rejected(self):
        for column, values in [
            ("topic", ["unknown label", "art", "science"]),
            ("topic", [None, "art", "science"]),
            ("docno", ["C1-000000", "C1-000000", "C1-000001"]),
        ]:
            frames = {name: frame.copy() for name, frame in self.collections.items()}
            frames["C1"][column] = values
            with self.subTest(column=column, values=values), self.assertRaises(ValueError):
                build_topic_qrels(frames)

    def test_serializer_rejects_invalid_topic_ids_iteration_and_relevance(self):
        original = build_topic_qrels(self.collections)["C1"]
        for column, value in [("tid", "T99"), ("iteration", 1), ("relevance", 0)]:
            frame = original.copy()
            frame[column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                format_topic_qrels(frame)

    def test_missing_extra_and_multi_field_mismatch_count_once_per_document(self):
        official = pd.DataFrame(
            [
                ["T02", 0, "C1-000000", 1],
                ["T02", 0, "C1-000001", 1],
                ["T25", 0, "C1-000002", 1],
            ],
            columns=QRELS_COLUMNS,
        )
        reconstructed = official.iloc[[0, 2]].copy()
        reconstructed.loc[2, ["tid", "iteration", "relevance"]] = ["T06", 1, 0]
        reconstructed.loc[3] = ["T32", 0, "C1-000003", 1]
        before = reconstructed.copy(deep=True)
        with TemporaryDirectory() as temporary:
            official.to_csv(Path(temporary) / "qrels_topics_P1.txt", sep=" ", header=False, index=False)
            (Path(temporary) / "qrels_topics_P2.txt").write_text("", encoding="utf-8")
            summary, details = compare_topic_qrels(
                {"C1": reconstructed, "C2": official.iloc[:0]}, temporary
            )
        self.assertEqual(summary.loc["C1", ["missing", "extra", "mismatched"]].tolist(), [1, 1, 1])
        self.assertEqual(summary.loc["C1", "total_differences"], 3)
        self.assertEqual(summary.loc["C2", "total_differences"], 0)
        self.assertEqual(details["C1"]["missing"]["docno"].tolist(), ["C1-000001"])
        self.assertEqual(details["C1"]["extra"]["docno"].tolist(), ["C1-000003"])
        self.assertEqual(details["C1"]["mismatched"]["docno"].tolist(), ["C1-000002"])
        self.assertEqual(
            details["C1"]["mismatched"]["mismatched_fields"].tolist(),
            ["tid, iteration, relevance"],
        )
        assert_frame_equal(reconstructed, before)

    def test_matching_records_ignore_row_order_and_inter_field_whitespace(self):
        judgments = build_topic_qrels(self.collections)
        with TemporaryDirectory() as temporary:
            for name, filename in [("C1", "qrels_topics_P1.txt"), ("C2", "qrels_topics_P2.txt")]:
                text = format_topic_qrels(judgments[name]).replace(" ", "\t  ")
                (Path(temporary) / filename).write_text(text, encoding="utf-8")
            summary, _ = compare_topic_qrels(
                {name: frame.iloc[::-1] for name, frame in judgments.items()}, temporary
            )
        self.assertEqual(summary["total_differences"].tolist(), [0, 0])
        self.assertEqual(summary["reconstructed_count"].tolist(), [3, 2])


if __name__ == "__main__":
    unittest.main()
