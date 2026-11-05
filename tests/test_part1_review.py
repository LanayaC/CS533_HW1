"""Synthetic checks for the final Part 1 audit helpers."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.part1_review import inspect_corpus, query_target_statistics, raw_data_statistics


class Part1ReviewTests(unittest.TestCase):
    def test_corpus_counts_format_hash_and_current_tokenizer_alignment(self):
        content = b"the nba team wins\n\n"
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.txt"
            path.write_bytes(content)
            result = inspect_corpus(path, ["The N.B.A. team wins", None])
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(result["article_count"], 2)
        self.assertEqual(result["total_tokens"], 4)
        self.assertEqual(result["distinct_tokens"], 4)
        self.assertEqual(result["empty_lines"], 1)
        self.assertEqual(result["mismatched_article_lines"], 0)
        self.assertEqual(result["sha256"], hashlib.sha256(content).hexdigest())

    def test_corpus_detects_wrong_text_missing_articles_and_bad_line_endings(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.txt"
            path.write_bytes(b"wrong\r\n")
            result = inspect_corpus(path, ["right", "missing article"])
        self.assertEqual(result["mismatched_article_lines"], 2)
        self.assertEqual(result["non_lf_or_unterminated_lines"], 1)

    def test_overlap_distinguishes_repeated_token_and_unique_term_conventions(self):
        clean = pd.DataFrame({"web_url": ["url"], "text": ["the news"]})
        queries = {"C1": pd.DataFrame({"qid": ["KI1-0001"], "query": ["the the missing"], "docno": ["d"]})}
        collections = {"C1": pd.DataFrame({"docno": ["d"], "web_url": ["url"]})}
        before = clean.copy(deep=True)
        result = query_target_statistics(queries, collections, clean)["C1"]
        self.assertAlmostEqual(result["mean_token_occurrence_overlap_pct"], 100 * 2 / 3)
        self.assertAlmostEqual(result["mean_unique_term_overlap_pct"], 50.0)
        assert_frame_equal(clean, before)

    def test_raw_statistics_and_monthly_counts_do_not_mutate_input(self):
        raw = pd.DataFrame({
            "pub_date": ["2009-11-01", "2009-12-01", "2025-05-01"],
            "mapped_section": ["Sports & Athletics", "Other", "Sports & Athletics"],
        })
        before = raw.copy(deep=True)
        result = raw_data_statistics(raw)
        self.assertEqual(result["article_count"], 3)
        self.assertEqual(sum(row["article_count"] for row in result["monthly_2009_2025"]), 3)
        self.assertEqual(len(result["year_category_counts"]), 2)
        assert_frame_equal(raw, before)


if __name__ == "__main__":
    unittest.main()
