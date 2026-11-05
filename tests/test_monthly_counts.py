"""Synthetic checks for Part 1.1; no dataset files are read."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.monthly_counts import monthly_article_counts


class MonthlyArticleCountsTests(unittest.TestCase):
    def setUp(self):
        self.articles = pd.DataFrame(
            {
                "pub_date": [
                    "2009-01-01",
                    "2009-01-01",  # Duplicate rows still count as articles.
                    "2009-02-10",
                    "2023-12-01",
                    "2024-01-05",
                    "2024-01-06T10:30:00",  # A different valid date format.
                    "2024-01-09",
                    "2025-03-02",
                    "2025-03-08",
                    "not a date",
                    None,
                ],
                "mapped_section": [
                    "Sports & Athletics",
                    "Sports & Athletics",
                    "Science",
                    "Sports & Athletics",
                    "Sports & Athletics",
                    "Sports & Athletics",
                    "Science",
                    "Sports & Athletics",
                    None,
                    "Sports & Athletics",
                    "Sports & Athletics",
                ],
            },
            # Counts must work even when input row labels are not unique.
            index=[0] * 11,
        )

    def test_overall_monthly_counts(self):
        before = self.articles.copy(deep=True)
        result = monthly_article_counts(self.articles, years=[2025, 2009, 2009])

        expected = pd.DataFrame(
            {"article_count": [2, 1] + [0] * 10 + [0, 0, 2] + [0] * 9},
            index=pd.date_range("2009-01-01", periods=12, freq="MS").append(
                pd.date_range("2025-01-01", periods=12, freq="MS")
            ).rename("month"),
        )
        assert_frame_equal(result, expected, check_freq=False)
        assert_frame_equal(self.articles, before)

    def test_sports_and_category_filtering(self):
        before = self.articles.copy(deep=True)
        result = monthly_article_counts(
            self.articles,
            years=[2023, 2024, 2025],
            categories=["Sports & Athletics", "Science", "Absent category"],
        )

        expected = pd.DataFrame(
            0,
            index=pd.date_range(
                "2023-01-01", "2025-12-01", freq="MS", name="month"
            ),
            columns=pd.Index(
                ["Sports & Athletics", "Science", "Absent category"],
                name="mapped_section",
            ),
        )
        expected.loc["2023-12-01", "Sports & Athletics"] = 1
        expected.loc["2024-01-01", "Sports & Athletics"] = 2
        expected.loc["2024-01-01", "Science"] = 1
        expected.loc["2025-03-01", "Sports & Athletics"] = 1

        assert_frame_equal(result, expected, check_freq=False)
        assert_frame_equal(self.articles, before)


if __name__ == "__main__":
    unittest.main()
