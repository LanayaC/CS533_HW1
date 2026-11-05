"""Small reporting-only fixtures; do not read the real dataset."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.part1_statistics import official_reporting_statistics


class Part1StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        pd.DataFrame(
            {"tid": ["T01", "T02"], "query": ["art", "science"], "paths": ["arts/design", "science"]}
        ).to_csv(self.root / "topics.tsv", sep="\t", index=False)
        self.collections = {
            "P1": pd.DataFrame(
                {
                    "docno": ["C1-000000", "C1-000001", "C1-000002"],
                    "web_url": ["url-a", "url-b", "url-c"],
                    "year": [2005, 2006, 2007],
                    "topic": ["art", "science", "art"],
                }
            ),
            "P2": pd.DataFrame(
                {"docno": ["C2-000000"], "web_url": ["url-d"], "year": [2025], "topic": ["science"]}
            ),
        }
        self.queries = {
            "P1": pd.DataFrame(
                {
                    "qid": ["KI1-0003", "KI1-0001", "KI1-0002"],
                    "query": ["three full query words", "the first query", "two words"],
                    "docno": ["C1-000002", "C1-000000", "C1-000001"],
                    "year": [2007, 2005, 2006],
                }
            ),
            "P2": pd.DataFrame(
                {"qid": ["KI2-0001"], "query": ["one complete query"], "docno": ["C2-000000"], "year": [2025]}
            ),
        }
        self.write_metadata()

    def write_metadata(self):
        for period in ["P1", "P2"]:
            self.collections[period].to_csv(self.root / f"collection_{period}.tsv", sep="\t", index=False)
            self.queries[period].to_csv(self.root / f"queries_ki_{period}.tsv", sep="\t", index=False)

    def test_complete_distributions_lengths_examples_and_json(self):
        stats = official_reporting_statistics(self.root)
        self.assertEqual(
            stats["topic_distribution"],
            [
                {"tid": "T01", "topic": "art", "P1": 2, "P2": 0},
                {"tid": "T02", "topic": "science", "P1": 1, "P2": 1},
            ],
        )
        self.assertEqual(stats["query_length_statistics"]["P1"], {
            "count": 3, "mean": 3.0, "std": 1.0, "min": 2,
            "25%": 2.5, "50%": 3.0, "75%": 3.5, "max": 4,
        })
        self.assertIsNone(stats["query_length_statistics"]["P2"]["std"])
        self.assertEqual(
            [row["qid"] for row in stats["known_item_examples"]["P1"]],
            ["KI1-0001", "KI1-0002", "KI1-0003"],
        )
        self.assertEqual(stats["known_item_examples"]["P1"][2]["query"], "three full query words")
        self.assertEqual(stats["known_item_examples"]["P1"][0]["length"], 3)
        json.dumps(stats, allow_nan=False)

    def test_unknown_collection_topic_is_not_dropped(self):
        self.collections["P1"].loc[0, "topic"] = "unknown"
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "unknown topic"):
            official_reporting_statistics(self.root)

    def test_missing_query_target_is_rejected(self):
        self.queries["P1"].loc[0, "docno"] = "missing-doc"
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "target absent"):
            official_reporting_statistics(self.root)

    def test_inconsistent_target_year_is_rejected(self):
        self.queries["P1"].loc[0, "year"] = 2008
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "years disagree"):
            official_reporting_statistics(self.root)

    def test_duplicate_query_ids_are_rejected(self):
        self.queries["P1"].loc[0, "qid"] = "KI1-0001"
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "duplicate qid"):
            official_reporting_statistics(self.root)


if __name__ == "__main__":
    unittest.main()
