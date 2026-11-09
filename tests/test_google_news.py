"""Small offline Part 2.4 checks; no downloads, pretrained files, or training."""

from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from gensim.models import KeyedVectors
import numpy as np

from src.google_news import gn_lookup_candidates, load_gn_vectors, lookup_gn_key, main


class GoogleNewsLookupTests(unittest.TestCase):
    def test_exact_candidate_order_for_words_and_phrases(self):
        self.assertEqual(gn_lookup_candidates("us"), ("us", "Us", "US"))
        self.assertEqual(
            gn_lookup_candidates("new_york"),
            ("new_york", "New_york", "NEW_YORK", "New_York"),
        )
        self.assertEqual(
            gn_lookup_candidates("new_york_city"),
            ("new_york_city", "New_york_city", "NEW_YORK_CITY", "New_York_City"),
        )
        self.assertEqual(gn_lookup_candidates("2025"), ("2025",))

    def test_each_fallback_and_priority_when_multiple_versions_exist(self):
        variants = ("new_york", "New_york", "NEW_YORK", "New_York")
        for index, expected in enumerate(variants):
            with self.subTest(expected=expected):
                self.assertEqual(lookup_gn_key(set(variants[index:]), "new_york"), expected)
        for vocabulary, expected in [
            ({"apple", "Apple", "APPLE"}, "apple"),
            ({"Apple", "APPLE"}, "Apple"),
            ({"APPLE"}, "APPLE"),
        ]:
            with self.subTest(vocabulary=vocabulary):
                self.assertEqual(lookup_gn_key(vocabulary, "apple"), expected)

    def test_stops_at_first_match_and_does_not_scan_or_change_vocabulary(self):
        class MembershipOnly:
            def __init__(self):
                self.queries = []

            def __contains__(self, key):
                self.queries.append(key)
                return key == "US"

        vocabulary = MembershipOnly()
        self.assertEqual(lookup_gn_key(vocabulary, "us"), "US")
        self.assertEqual(vocabulary.queries, ["us", "Us", "US"])
        vocabulary = {"new_york": 1, "New_York": 2}
        before = dict(vocabulary)
        self.assertEqual(lookup_gn_key(vocabulary, "new_york"), "new_york")
        self.assertEqual(vocabulary, before)

    def test_missing_is_none_without_extra_spelling_or_phrase_guesses(self):
        self.assertIsNone(lookup_gn_key({}, "unknown"))
        self.assertIsNone(lookup_gn_key({"aPpLe"}, "apple"))
        self.assertIsNone(lookup_gn_key({"NewYork", "New York", "new", "york"}, "new_york"))
        self.assertIsNone(lookup_gn_key({"Don'T"}, "don't"))

    def test_invalid_input_is_distinct_from_missing(self):
        for token in [None, 2025, ["apple"]]:
            with self.subTest(token=token), self.assertRaises(TypeError):
                lookup_gn_key({}, token)
        for token in ["", " ", "new york", "new\tyork", "apple\n", "Apple", "NEW_YORK"]:
            with self.subTest(token=token), self.assertRaises(ValueError):
                lookup_gn_key({}, token)

    def test_keyedvectors_returns_exact_existing_vector_without_mutation(self):
        vectors = KeyedVectors(vector_size=3)
        keys = ["us", "Us", "US", "Apple", "New_York"]
        vectors.add_vectors(keys, np.arange(15, dtype=np.float32).reshape(5, 3))
        before = vectors.vectors.copy()
        keys_before = list(vectors.index_to_key)
        for token, expected in [("us", "us"), ("apple", "Apple"), ("new_york", "New_York")]:
            with self.subTest(token=token):
                key = lookup_gn_key(vectors, token)
                self.assertEqual(key, expected)
                np.testing.assert_array_equal(vectors[key], before[vectors.key_to_index[expected]])
        self.assertIsNone(lookup_gn_key(vectors, "missing"))
        np.testing.assert_array_equal(vectors.vectors, before)
        self.assertEqual(vectors.index_to_key, keys_before)


class GoogleNewsLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.vectors = KeyedVectors(vector_size=3)
        self.vectors.add_vectors(
            ["apple", "US", "New_York"],
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        )

    def test_binary_and_gzip_files_load_in_full_without_changing_source(self):
        binary = self.root / "tiny.bin"
        self.vectors.save_word2vec_format(str(binary), binary=True)
        compressed = self.root / "tiny.bin.gz"
        with gzip.open(compressed, "wb") as handle:
            handle.write(binary.read_bytes())
        for path in [binary, compressed]:
            with self.subTest(path=path):
                before = path.read_bytes()
                loaded = load_gn_vectors(path)
                self.assertEqual(loaded.index_to_key, self.vectors.index_to_key)
                np.testing.assert_array_equal(loaded.vectors, self.vectors.vectors)
                self.assertEqual(path.read_bytes(), before)

    def test_native_cache_uses_read_only_mmap_and_preserves_sidecars(self):
        native = self.root / "tiny.kv"
        self.vectors.save(str(native), separately=["vectors"])
        before = {path.name: path.read_bytes() for path in self.root.iterdir()}
        loaded = load_gn_vectors(native, model_format="keyedvectors")
        self.assertIsInstance(loaded.vectors, np.memmap)
        self.assertFalse(loaded.vectors.flags.writeable)
        np.testing.assert_array_equal(loaded.vectors, self.vectors.vectors)
        self.assertEqual(lookup_gn_key(loaded, "new_york"), "New_York")
        self.assertEqual({path.name: path.read_bytes() for path in self.root.iterdir()}, before)
        # Release the memory map before Windows TemporaryDirectory cleanup.
        del loaded

    def test_loader_uses_expected_apis_without_downloading_or_training(self):
        path = self.root / "placeholder"
        path.write_bytes(b"placeholder for mocked loader")
        with patch("src.google_news.KeyedVectors.load_word2vec_format") as binary:
            load_gn_vectors(path)
            binary.assert_called_once_with(str(path.resolve()), binary=True)
        with patch("src.google_news.KeyedVectors.load") as native:
            load_gn_vectors(path, model_format="keyedvectors")
            native.assert_called_once_with(str(path.resolve()), mmap="r")

    def test_bad_paths_and_formats_fail_before_loading(self):
        empty = self.root / "empty.bin"
        empty.touch()
        for path in [self.root / "absent.bin", empty, self.root]:
            with self.subTest(path=path), patch("src.google_news.KeyedVectors.load_word2vec_format") as load:
                with self.assertRaises(ValueError):
                    load_gn_vectors(path)
                load.assert_not_called()
        with self.assertRaisesRegex(ValueError, "model_format"):
            load_gn_vectors(empty, model_format="unknown")

    def test_cli_reports_matched_keys_and_explicit_missing_status(self):
        path = self.root / "tiny.bin"
        self.vectors.save_word2vec_format(str(path), binary=True)
        before = {item.name: item.read_bytes() for item in self.root.iterdir()}
        output = io.StringIO()
        with redirect_stdout(output):
            result = main([
                "--model", str(path), "--words", "apple", "us", "new_york", "missing",
            ])
        report = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(report["vocabulary_size"], 3)
        self.assertEqual(report["vector_size"], 3)
        self.assertEqual(
            [(item["token"], item["matched_key"], item["status"]) for item in report["lookups"]],
            [("apple", "apple", "found"), ("us", "US", "found"),
             ("new_york", "New_York", "found"), ("missing", None, "missing")],
        )
        self.assertEqual({item.name: item.read_bytes() for item in self.root.iterdir()}, before)

    def test_cli_rejects_invalid_tokens_before_loading(self):
        with patch("src.google_news.load_gn_vectors") as load:
            with self.assertRaisesRegex(ValueError, "lowercase"):
                main(["--model", str(self.root / "absent.bin"), "--words", "Apple"])
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
