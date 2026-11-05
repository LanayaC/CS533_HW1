"""Small synthetic checks for period token counts; no datasets are read."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.token_counts import count_tokens_by_period


class CountTokensByPeriodTests(unittest.TestCase):
    def test_repeated_tokens_stop_words_and_separate_vocabularies(self):
        articles = pd.DataFrame(
            {
                "pub_date": ["2005-01-01", "2014-12-31", "2015-01-01", "2025-12-31"],
                "text": ["The cat cat.", "A U.S. cat.", "The dog dog.", "A dog's dog."],
            },
            # Article counts must not depend on unique row labels.
            index=[7, 7, 7, 7],
        )
        before = articles.copy(deep=True)

        result = count_tokens_by_period(articles)

        # P1: the, cat, cat, a, us, cat -> 6 total, 4 distinct.
        # P2: the, dog, dog, a, dog, dog -> 6 total, 3 distinct.
        expected = pd.DataFrame(
            {"total_tokens": [6, 6], "distinct_tokens": [4, 3]},
            index=pd.Index(["P1", "P2"], name="period"),
        )
        assert_frame_equal(result, expected)
        assert_frame_equal(articles, before)

    def test_inclusive_boundaries_and_excluded_dates(self):
        articles = pd.DataFrame(
            {
                "pub_date": [
                    "2004-12-31", "2005-01-01", "2014-12-31T23:59:59",
                    "2015-01-01", "2025-12-31T23:59:59", "2026-01-01",
                    "invalid date", None,
                ],
                "text": [
                    "excluded", "first", "last", "start", "end", "excluded",
                    "excluded", "excluded",
                ],
            }
        )

        result = count_tokens_by_period(articles)

        expected = pd.DataFrame(
            {"total_tokens": [2, 2], "distinct_tokens": [2, 2]},
            index=pd.Index(["P1", "P2"], name="period"),
        )
        assert_frame_equal(result, expected)

    def test_missing_empty_text_and_empty_input(self):
        expected = pd.DataFrame(
            {"total_tokens": [0, 0], "distinct_tokens": [0, 0]},
            index=pd.Index(["P1", "P2"], name="period"),
        )
        articles = pd.DataFrame(
            {
                "pub_date": pd.to_datetime(
                    ["2005-01-01", "2014-12-31", "2015-01-01", "2025-12-31"]
                ),
                "text": [None, pd.NA, "", " \t\n"],
            }
        )

        for frame in [articles, articles.iloc[:0]]:
            with self.subTest(empty=frame.empty):
                assert_frame_equal(count_tokens_by_period(frame), expected)

    def test_empty_period_and_custom_column_names(self):
        articles = pd.DataFrame(
            {"date": ["2020-01-01"], "body": ["News news and news"]}
        )

        result = count_tokens_by_period(articles, text_col="body", date_col="date")

        expected = pd.DataFrame(
            {"total_tokens": [0, 4], "distinct_tokens": [0, 2]},
            index=pd.Index(["P1", "P2"], name="period"),
        )
        assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
