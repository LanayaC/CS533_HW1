"""Synthetic corpus-writing tests using temporary directories only."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from src.corpus_writer import write_training_corpora
from src.preprocessing import preprocess_text


class CorpusWriterTests(unittest.TestCase):
    def setUp(self):
        # Deliberately interleave periods and avoid chronological ordering.
        self.articles = pd.DataFrame(
            {
                "pub_date": [
                    "2025-12-31T23:59:59", "2014-12-31T23:59:59",
                    "2015-01-01", "2005-01-01",
                ],
                "text": [
                    "The N.B.A. team's 2025 pre-season.", "Caf\u00e9 news news.",
                    "And the new day.", "The F.B.I. report.",
                ],
            },
            index=[7, 7, 7, 7],
        )

    def test_boundaries_order_format_counts_and_input_preservation(self):
        before = self.articles.copy(deep=True)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "corpora"
            summary = write_training_corpora(self.articles, output)

            self.assertEqual(
                (output / "corpus_P1.txt").read_bytes(),
                "caf\u00e9 news news\nthe fbi report\n".encode("utf-8"),
            )
            self.assertEqual(
                (output / "corpus_P2.txt").read_bytes(),
                b"the nba team 2025 pre season\nand the new day\n",
            )
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"corpus_P1.txt", "corpus_P2.txt"},
            )
            expected = pd.DataFrame(
                {"article_count": [2, 2], "token_count": [6, 10]},
                index=pd.Index(["P1", "P2"], name="period"),
            )
            assert_frame_equal(summary, expected)
        assert_frame_equal(self.articles, before)

    def test_empty_text_empty_period_and_empty_input(self):
        articles = pd.DataFrame(
            {"date": ["2005-01-01", "2014-12-31"], "body": [None, ""]}
        )
        for frame, expected_articles in [(articles, 2), (articles.iloc[:0], 0)]:
            with self.subTest(empty=frame.empty), TemporaryDirectory() as temporary:
                output = Path(temporary)
                summary = write_training_corpora(
                    frame, output, text_col="body", date_col="date"
                )
                self.assertEqual(
                    (output / "corpus_P1.txt").read_bytes(), b"\n" * expected_articles
                )
                self.assertEqual((output / "corpus_P2.txt").read_bytes(), b"")
                self.assertEqual(summary["article_count"].tolist(), [expected_articles, 0])
                self.assertEqual(summary["token_count"].tolist(), [0, 0])

    def test_invalid_or_out_of_range_dates_fail_before_creating_output(self):
        for date in [None, "not a date", "2004-12-31", "2026-01-01"]:
            with self.subTest(date=date), TemporaryDirectory() as temporary:
                output = Path(temporary) / "corpora"
                articles = pd.DataFrame({"pub_date": [date], "text": ["news"]})
                with self.assertRaises(ValueError):
                    write_training_corpora(articles, output)
                self.assertFalse(output.exists())

    def test_either_existing_corpus_is_preserved(self):
        for name in ["corpus_P1.txt", "corpus_P2.txt"]:
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                output = Path(temporary)
                existing = output / name
                existing.write_bytes(b"existing content\n")
                with self.assertRaises(FileExistsError):
                    write_training_corpora(self.articles, output)
                self.assertEqual(existing.read_bytes(), b"existing content\n")
                self.assertEqual(list(output.iterdir()), [existing])

    def test_tokenizer_failure_and_keyboard_interrupt_leave_no_outputs(self):
        for failure in [RuntimeError("tokenizer failed"), KeyboardInterrupt()]:
            with self.subTest(failure=type(failure).__name__):
                with TemporaryDirectory() as temporary:
                    output = Path(temporary)
                    with patch(
                        "src.corpus_writer.preprocess_text",
                        side_effect=[preprocess_text(self.articles.iloc[0]["text"]), failure],
                    ):
                        with self.assertRaises(type(failure)):
                            write_training_corpora(self.articles, output)
                    self.assertEqual(list(output.iterdir()), [])

    def test_publication_conflict_rolls_back_only_our_files(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            real_link = __import__("os").link

            def competing_link(source, destination):
                destination = Path(destination)
                if destination.name == "corpus_P2.txt":
                    destination.write_bytes(b"another writer's content\n")
                real_link(source, destination)

            with patch("src.corpus_writer.os.link", side_effect=competing_link):
                with self.assertRaises(FileExistsError):
                    write_training_corpora(self.articles, output)

            self.assertFalse((output / "corpus_P1.txt").exists())
            self.assertEqual(
                (output / "corpus_P2.txt").read_bytes(), b"another writer's content\n"
            )
            self.assertFalse((output / ".corpus_write_in_progress").exists())

    def test_forced_exit_leaves_marker_and_blocks_another_write(self):
        script = """
import os
import sys
from unittest.mock import patch
import pandas as pd
from src.corpus_writer import write_training_corpora

real_link = os.link
def exit_after_first_link(source, destination):
    real_link(source, destination)
    os._exit(23)

articles = pd.DataFrame({
    "pub_date": ["2005-01-01", "2025-12-31"],
    "text": ["The first article", "The second article"],
})
with patch("src.corpus_writer.os.link", side_effect=exit_after_first_link):
    write_training_corpora(articles, sys.argv[1])
"""
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = subprocess.run(
                [sys.executable, "-B", "-c", script, str(output)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertTrue((output / ".corpus_write_in_progress").is_dir())
            self.assertEqual(
                (output / "corpus_P1.txt").read_bytes(), b"the first article\n"
            )
            self.assertFalse((output / "corpus_P2.txt").exists())
            with self.assertRaisesRegex(FileExistsError, "Incomplete or active"):
                write_training_corpora(self.articles, output)
            self.assertEqual(
                (output / "corpus_P1.txt").read_bytes(), b"the first article\n"
            )


if __name__ == "__main__":
    unittest.main()
