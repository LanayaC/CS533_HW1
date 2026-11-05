"""Small synthetic checks for Part 1.3; no dataset files are read."""

import unittest

import pandas as pd

from src.preprocessing import preprocess_text


class PreprocessTextTests(unittest.TestCase):
    def test_lowercase_and_keep_stop_words(self):
        self.assertEqual(
            preprocess_text("The NEWS and the Art of Search"),
            ["the", "news", "and", "the", "art", "of", "search"],
        )

    def test_normalize_apostrophes_and_remove_possessives(self):
        for apostrophe in ["'", "\u2018", "\u2019", "\u02bc", "\uff07"]:
            with self.subTest(apostrophe=apostrophe):
                self.assertEqual(
                    preprocess_text(
                        f"Player{apostrophe}s players{apostrophe} don{apostrophe}t"
                    ),
                    ["player", "players", "don't"],
                )

    def test_nyt_acronyms_match_undotted_query_terms(self):
        self.assertEqual(
            preprocess_text("U.S. N.B.A. F.B.I. E.P.A. U.S"),
            preprocess_text("us nba fbi epa us"),
        )
        self.assertEqual(preprocess_text("The U.S.'s team"), ["the", "us", "team"])

    def test_split_punctuation_hyphens_and_underscores(self):
        self.assertEqual(
            preprocess_text("State-of-the-art, news/world; (team)! player\u2014coach x_y"),
            ["state", "of", "the", "art", "news", "world", "team", "player", "coach", "x", "y"],
        )

    def test_preserve_numeric_tokens_and_split_numeric_punctuation(self):
        self.assertEqual(
            preprocess_text("2025 42 3.14 1,000 covid-19"),
            ["2025", "42", "3", "14", "1", "000", "covid", "19"],
        )

    def test_missing_empty_and_non_string_values(self):
        for value in [None, float("nan"), pd.NA, pd.NaT, "", " \t\n", "...---!", 2025]:
            with self.subTest(value=repr(value)):
                self.assertEqual(preprocess_text(value), [])

    def test_unicode_letters_and_documented_ambiguities(self):
        self.assertEqual(preprocess_text("CAF\u00c9 na\u00efve"), ["caf\u00e9", "na\u00efve"])
        self.assertEqual(preprocess_text("It's e.g. U.S."), ["it", "eg", "us"])


if __name__ == "__main__":
    unittest.main()
