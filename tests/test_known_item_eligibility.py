"""Small synthetic Appendix A.4 eligibility checks; no real datasets are read."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.known_item_eligibility import check_known_item_eligibility


def url(slug, year=2012, path="sports/baseball", extension=".html"):
    return f"https://www.nytimes.com/{year}/06/22/{path}/{slug}{extension}"


class KnownItemEligibilityTests(unittest.TestCase):
    def test_yankees_passing_and_too_few_tokens_failing(self):
        articles = pd.DataFrame(
            {
                "web_url": [url("the-yankees-win-game-3"), url("the-yankees-win-3")],
                "text": ["Do not tokenize article text", "Do not alter this column"],
            },
            index=[7, 7],
        )
        before = articles.copy(deep=True)
        result = check_known_item_eligibility(articles)

        self.assertEqual(result["slug_occurrences"].tolist(), [1, 1])
        self.assertEqual(result["qualifying_token_count"].tolist(), [3, 2])
        self.assertEqual(result["unique_slug"].tolist(), [True, True])
        self.assertEqual(result["enough_qualifying_tokens"].tolist(), [True, False])
        self.assertEqual(result["eligible"].tolist(), [True, False])
        self.assertEqual(result.index.tolist(), [7, 7])
        self.assertEqual(result["web_url"].tolist(), articles["web_url"].tolist())
        assert_frame_equal(articles, before)

    def test_duplicate_slug_across_periods_and_outside_collections_fails(self):
        slug = "the-yankees-win-game-3"
        articles = pd.DataFrame(
            {
                "web_url": [url(slug, 2012), url(slug, 2020, path="unlisted-section")],
                "pub_date": pd.to_datetime(["2012-06-22", "2020-06-22"]),
            }
        )
        result = check_known_item_eligibility(articles)

        self.assertEqual(result["slug"].tolist(), [slug, slug])
        self.assertEqual(result["slug_occurrences"].tolist(), [2, 2])
        self.assertEqual(result["qualifying_token_count"].tolist(), [3, 3])
        self.assertEqual(result["unique_slug"].tolist(), [False, False])
        self.assertEqual(result["enough_qualifying_tokens"].tolist(), [True, True])
        self.assertEqual(result["eligible"].tolist(), [False, False])

    def test_exact_token_rules_lowercase_stop_words_and_whole_token_filtering(self):
        articles = pd.DataFrame(
            {"web_url": [url("THE-YANKEES-win-game-3-x-and-goal42-city's-caf\u00e9")]}
        )
        result = check_known_item_eligibility(articles)

        # Only yankees, win, game qualify. Nonalphabetic tokens are discarded,
        # not further split, stripped, or normalized by preprocess_text().
        self.assertEqual(result["qualifying_token_count"].tolist(), [3])
        self.assertEqual(result["eligible"].tolist(), [True])

    def test_repeated_qualifying_tokens_count_separately(self):
        result = check_known_item_eligibility(
            pd.DataFrame({"web_url": [url("news-news-news")]})
        )
        self.assertEqual(result["qualifying_token_count"].tolist(), [3])
        self.assertEqual(result["eligible"].tolist(), [True])

    def test_optional_html_suffix_produces_the_same_slug(self):
        slug = "the-yankees-win-game-3"
        result = check_known_item_eligibility(
            pd.DataFrame({"web_url": [url(slug), url(slug, 2020, extension="")]})
        )
        self.assertEqual(result["slug"].tolist(), [slug, slug])
        self.assertEqual(result["slug_occurrences"].tolist(), [2, 2])
        self.assertEqual(result["eligible"].tolist(), [False, False])

    def test_uniqueness_uses_extracted_slugs_before_lowercasing_tokens(self):
        result = check_known_item_eligibility(
            pd.DataFrame(
                {"web_url": [url("The-Yankees-Win-Game"), url("the-yankees-win-game")]}
            )
        )
        self.assertEqual(result["slug_occurrences"].tolist(), [1, 1])
        self.assertEqual(result["qualifying_token_count"].tolist(), [3, 3])
        self.assertEqual(result["eligible"].tolist(), [True, True])

    def test_missing_and_nonmatching_urls_fail_both_checks(self):
        valid = url("the-yankees-win-game-3")
        articles = pd.DataFrame(
            {
                "web_url": [
                    None, pd.NA, "", 123, valid.replace("https:", "http:"),
                    valid.replace("www.nytimes.com", "dealbook.nytimes.com"),
                ]
            }
        )
        result = check_known_item_eligibility(articles)

        self.assertTrue(result["slug"].isna().all())
        self.assertTrue(result["slug_occurrences"].eq(0).all())
        self.assertTrue(result["qualifying_token_count"].eq(0).all())
        self.assertFalse(result[["unique_slug", "enough_qualifying_tokens", "eligible"]].any().any())

    def test_empty_input(self):
        result = check_known_item_eligibility(pd.DataFrame(columns=["web_url"]))
        self.assertTrue(result.empty)
        self.assertEqual(
            result.columns.tolist(),
            [
                "web_url", "slug", "slug_occurrences", "qualifying_token_count",
                "unique_slug", "enough_qualifying_tokens", "eligible",
            ],
        )


if __name__ == "__main__":
    unittest.main()
