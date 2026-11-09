"""Part 2.6 tests use tiny synthetic vectors/files only; no downloads or jobs."""

from contextlib import redirect_stdout
import gzip
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gensim.models import KeyedVectors, Word2Vec
import numpy as np

from src.google_news import lookup_gn_key
from src.gn_comparison import (
    analyze_gn_neighbors, build_restricted_gn_index, checked_paths,
    compare_neighborhoods, main, run_comparison,
)
from src.semantic_drift import TARGET_WORDS


def vectors(keys, values):
    values = np.asarray(values, dtype=np.float32)
    result = KeyedVectors(vector_size=values.shape[1])
    result.add_vectors(keys, values)
    return result


def fixture():
    labels = ("privacy", "internet", "phone", "election", "climate", "new_york",
              "apple", "us", "a", "d", "c", "b", "missinggn")
    angles = {word: index / 10 for index, word in enumerate(labels)}
    angles["only_m2"] = .001

    def local(keys):
        return SimpleNamespace(wv=vectors(keys, [[np.cos(angles[word]), np.sin(angles[word])] for word in keys]))

    m1 = local(labels)
    m2 = local(("privacy", "only_m2", *labels[1:]))
    mapping = {word: word for word in labels if word != "missinggn"}
    mapping.update(privacy="Privacy", new_york="New_York", a="A", d="D", b="B")
    gn_keys = list(mapping.values())
    gn_values = [[np.cos(angles[word]), np.sin(angles[word]), .02 * index]
                 for index, word in enumerate(mapping)]
    # A deliberate tie verifies alphabetical ordering of ORIGINAL labels.
    gn_values[gn_keys.index("D")] = gn_values[gn_keys.index("c")] = [.5, .5, .1]
    gn_keys += ["Only_GN", "only_m2", "Apple", "Us", "US"]
    gn_values += [[1, 0, 0]] * 5
    return m1, m2, vectors(gn_keys, gn_values), mapping


def neighborhood(words, *, gn=False):
    return {"status": "ok", "neighbors": [
        {"word": word, **({"gn_key": word.upper()} if gn else {})} for word in words
    ]}


class RestrictedSearchTests(unittest.TestCase):
    def test_pool_uses_shared_lookup_order_for_every_m1_candidate(self):
        m1, _, gn, mapping = fixture()
        with patch("src.gn_comparison.lookup_gn_key", wraps=lookup_gn_key) as lookup:
            index = build_restricted_gn_index(m1.wv, gn)
            self.assertEqual(lookup.call_count, len(m1.wv))
            index.neighbors(gn, "us")
            self.assertEqual(lookup.call_count, len(m1.wv) + 1)
        self.assertEqual(dict(zip(index.labels, index.gn_keys)), mapping)
        self.assertEqual(index.statistics["resolvable_m1_words"], 12)
        self.assertEqual(index.statistics["unresolved_m1_words"], 1)
        self.assertEqual(index.statistics["usable_candidates"], 12)
        self.assertEqual(index.neighbors(gn, "us")["matched_key"], "us")
        self.assertEqual(dict(zip(index.labels, index.gn_keys))["apple"], "apple")

    def test_search_is_restricted_before_ranking_and_excludes_query(self):
        m1, _, gn, _ = fixture()
        index = build_restricted_gn_index(m1.wv, gn)
        # Global GN search must never be called, even when non-M1 items are closer.
        with patch.object(gn, "most_similar", side_effect=AssertionError("No global GN search")):
            result = index.neighbors(gn, "privacy")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["matched_key"], "Privacy")
        self.assertEqual(len(result["neighbors"]), 10)
        self.assertEqual(result["neighbors"][0]["word"], "internet")
        for item in result["neighbors"]:
            self.assertIn(item["word"], m1.wv)
            self.assertNotIn(item["word"], {"privacy", "only_m2", "Only_GN", "missinggn"})
            self.assertNotEqual(item["gn_key"], "Privacy")

    def test_scores_and_order_match_brute_force_restricted_cosine(self):
        m1, _, gn, mapping = fixture()
        index = build_restricted_gn_index(m1.wv, gn)
        query = gn["Privacy"].astype(np.float64)
        expected = []
        for label, key in mapping.items():
            if label == "privacy":
                continue
            vector = gn[key].astype(np.float64)
            score = float(vector @ query / np.linalg.norm(vector) / np.linalg.norm(query))
            expected.append((label, score))
        expected.sort(key=lambda item: (-item[1], item[0]))
        actual = index.neighbors(gn, "privacy")["neighbors"]
        self.assertEqual([item["word"] for item in actual], [word for word, _ in expected[:10]])
        np.testing.assert_allclose([item["cosine_similarity"] for item in actual], [score for _, score in expected[:10]], atol=1e-12)
        labels = [item["word"] for item in actual]
        self.assertLess(labels.index("c"), labels.index("d"))

    def test_phrase_query_is_excluded_using_its_resolved_key(self):
        m1, _, gn, _ = fixture()
        result = build_restricted_gn_index(m1.wv, gn).neighbors(gn, "new_york")
        self.assertEqual(result["matched_key"], "New_York")
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("new_york", [item["word"] for item in result["neighbors"]])
        self.assertNotIn("New_York", [item["gn_key"] for item in result["neighbors"]])

    def test_query_aliases_cannot_return_the_query_under_another_label(self):
        labels = ["ss", "\u00df"] + [f"word{letter}" for letter in "abcdefghijkl"]
        m1 = vectors(labels, np.ones((len(labels), 2)))
        keys = ["SS"] + [label.capitalize() for label in labels[2:]]
        gn = vectors(keys, np.ones((len(keys), 3)))
        result = build_restricted_gn_index(m1, gn).neighbors(gn, "ss")
        self.assertEqual(result["matched_key"], "SS")
        self.assertEqual(result["status"], "ok")
        self.assertFalse({"ss", "\u00df"} & {item["word"] for item in result["neighbors"]})
        self.assertNotIn("SS", [item["gn_key"] for item in result["neighbors"]])

    def test_invalid_candidates_and_missing_queries_are_explicit(self):
        m1, _, gn, _ = fixture()
        gn["c"] = np.zeros(3)
        index = build_restricted_gn_index(m1.wv, gn)
        self.assertEqual(index.statistics["usable_candidates"], 11)
        self.assertEqual(index.statistics["undefined_candidates"], [
            {"word": "c", "gn_key": "c", "reason": "zero_or_nonfinite_vector"},
        ])
        self.assertNotIn("c", index.labels)
        self.assertTrue(index.neighbors(gn, "c")["status"].startswith("undefined:"))
        self.assertEqual(index.neighbors(gn, "missinggn"), {"status": "missing", "matched_key": None, "neighbors": None})

    def test_fewer_than_ten_candidates_does_not_change_the_denominator(self):
        m1 = vectors(["a", "b", "c"], np.ones((3, 2)))
        gn = vectors(["A", "B", "C"], np.ones((3, 3)))
        result = build_restricted_gn_index(m1, gn).neighbors(gn, "a")
        self.assertEqual(result["status"], "insufficient_candidates")
        self.assertEqual(result["available_candidates"], 2)
        comparison = compare_neighborhoods(result, neighborhood([f"n{i}" for i in range(10)]), "M1")
        self.assertIsNone(comparison["score"])

    def test_empty_pool_is_handled_without_guessing_words(self):
        m1 = vectors(["absent"], [[1, 0]])
        gn = vectors(["A"], [[1, 0, 0]])
        index = build_restricted_gn_index(m1, gn)
        self.assertEqual(index.normalized_vectors.shape, (0, 3))
        result = index.neighbors(gn, "a")
        self.assertEqual(result["status"], "insufficient_candidates")
        self.assertEqual(result["available_candidates"], 0)


class ComparisonTests(unittest.TestCase):
    def test_overlap_uses_original_labels_not_case_sensitive_gn_keys(self):
        left = [f"a{i}" for i in range(10)]
        right = left[-3:] + [f"b{i}" for i in range(7)]
        result = compare_neighborhoods(neighborhood(left, gn=True), neighborhood(right), "M1")
        self.assertEqual(result, {"score": .3, "shared_words": left[-3:], "status": "ok"})

    def test_full_local_vocabularies_partial_missing_and_unchanged_vectors(self):
        m1, m2, gn, _ = fixture()
        before = [model.vectors.copy() for model in (m1.wv, m2.wv, gn)]
        report = analyze_gn_neighbors(m1, m2, gn, targets=("privacy", "missinggn", "unknown", "only_m2"))
        rows = {row["word"]: row for row in report["words"]}
        self.assertIn("only_m2", [item["word"] for item in rows["privacy"]["m2"]["neighbors"]])
        self.assertNotIn("only_m2", [item["word"] for item in rows["privacy"]["gn"]["neighbors"]])
        self.assertEqual(rows["missinggn"]["missing_models"], ["GN"])
        self.assertIsNone(rows["missinggn"]["nn_gn_m1"])
        self.assertEqual(rows["missinggn"]["closer_to"], "unavailable")
        self.assertEqual(rows["unknown"]["missing_models"], ["M1", "M2", "GN"])
        self.assertEqual(rows["only_m2"]["missing_models"], ["M1"])
        self.assertIsNone(rows["only_m2"]["nn_gn_m1"])
        self.assertIsNotNone(rows["only_m2"]["nn_gn_m2"])
        for model, original in zip((m1.wv, m2.wv, gn), before):
            np.testing.assert_array_equal(model.vectors, original)
        json.dumps(report, allow_nan=False)

    def test_default_analysis_keeps_all_44_words_and_two_scores_each(self):
        m1, m2, gn, _ = fixture()
        report = analyze_gn_neighbors(m1, m2, gn)
        self.assertEqual(report["target_count"], 44)
        self.assertEqual([row["word"] for row in report["words"]], list(TARGET_WORDS))
        self.assertEqual(report["neighbor_count"], 10)
        for row in report["words"]:
            for field in ("nn_gn_m1", "nn_gn_m2"):
                self.assertIn(field, row)
                if row[field] is not None:
                    self.assertTrue(0 <= row[field] <= 1)
            if row["nn_gn_m1"] is not None and row["nn_gn_m2"] is not None:
                expected = "tie" if row["nn_gn_m1"] == row["nn_gn_m2"] else (
                    "M1" if row["nn_gn_m1"] > row["nn_gn_m2"] else "M2"
                )
                self.assertEqual(row["closer_to"], expected)


class ComparisonFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        m1, m2, gn, _ = fixture()
        self.inputs = (self.root / "m1.model", self.root / "m2.model", self.root / "gn.bin.gz")
        for wrapper, path in zip((m1, m2), self.inputs):
            model = Word2Vec(vector_size=wrapper.wv.vector_size, min_count=1, workers=1)
            model.build_vocab_from_freq({word: 10 for word in wrapper.wv.index_to_key})
            for word in wrapper.wv.index_to_key:
                model.wv[word] = wrapper.wv[word]
            model.save(str(path), separately=[], sep_limit=2**63)
        binary = self.root / "tiny.bin"
        gn.save_word2vec_format(str(binary), binary=True)
        with gzip.open(self.inputs[2], "wb") as handle:
            handle.write(binary.read_bytes())
        self.output = self.root / "results" / "gn_comparison.json"

    def test_cli_loads_gzip_and_writes_compact_json_without_model_changes(self):
        before = {path: path.read_bytes() for path in self.inputs}
        stdout = io.StringIO()
        with patch.object(Word2Vec, "train", side_effect=AssertionError("No training")) as train:
            with patch.object(Word2Vec, "save", side_effect=AssertionError("No model writes")) as save:
                with redirect_stdout(stdout):
                    result = main([
                        "--m1", str(self.inputs[0]), "--m2", str(self.inputs[1]),
                        "--gn", str(self.inputs[2]), "--gn-format", "word2vec", "--output", str(self.output),
                    ])
                train.assert_not_called()
                save.assert_not_called()
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(report["target_count"], 44)
        self.assertEqual(report["gn_format"], "word2vec")
        self.assertEqual(report["model_paths"]["GN"], str(self.inputs[2].resolve()))
        self.assertEqual(report["vocabulary_sizes"]["GN"], 17)
        self.assertLess(self.output.stat().st_size, 100_000)
        self.assertEqual({path.name for path in self.output.parent.iterdir()}, {self.output.name})
        self.assertEqual(json.loads(stdout.getvalue())["target_count"], 44)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_dry_run_never_loads_models_or_creates_results_directory(self):
        stdout = io.StringIO()
        with patch("src.gn_comparison.Word2Vec.load") as local, patch("src.gn_comparison.load_gn_vectors") as gn:
            with redirect_stdout(stdout):
                result = main([
                    "--m1", str(self.inputs[0]), "--m2", str(self.inputs[1]),
                    "--gn", str(self.inputs[2]), "--output", str(self.output), "--dry-run",
                ])
            local.assert_not_called()
            gn.assert_not_called()
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue())["target_count"], 44)
        self.assertFalse(self.output.parent.exists())

    def test_native_format_is_forwarded_to_existing_loader(self):
        _, _, gn, _ = fixture()
        with patch("src.gn_comparison.load_gn_vectors", return_value=gn) as load:
            report = run_comparison(*self.inputs, self.output, gn_format="keyedvectors", targets=("privacy",))
        load.assert_called_once_with(self.inputs[2].resolve(), model_format="keyedvectors")
        self.assertEqual(report["gn_format"], "keyedvectors")

    def test_existing_outputs_markers_and_model_paths_are_protected(self):
        self.output.parent.mkdir()
        marker = self.output.parent / f".{self.output.name}.in_progress"
        for path in (self.output, marker):
            path.write_bytes(b"preserve")
            with self.subTest(path=path), patch("src.gn_comparison.Word2Vec.load") as load:
                with self.assertRaises(FileExistsError):
                    run_comparison(*self.inputs, self.output)
                load.assert_not_called()
                self.assertEqual(path.read_bytes(), b"preserve")
            path.unlink()
        with self.assertRaisesRegex(ValueError, "input model"):
            checked_paths(*self.inputs, self.inputs[0])
        inside = Path(__file__).resolve().parents[1] / "results" / "forbidden.json"
        with self.assertRaisesRegex(ValueError, "outside"):
            checked_paths(*self.inputs, inside)
        with self.assertRaisesRegex(ValueError, "different"):
            checked_paths(self.inputs[0], self.inputs[0], self.inputs[2], self.output)

    def test_failure_does_not_publish_or_retry(self):
        with patch("src.gn_comparison.analyze_gn_neighbors", side_effect=RuntimeError("synthetic failure")) as analyze:
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                run_comparison(*self.inputs, self.output)
            analyze.assert_called_once()
        self.assertEqual(list(self.output.parent.iterdir()), [])

    def test_concurrent_result_is_neither_overwritten_nor_removed(self):
        real_link = os.link

        def compete(source, destination):
            Path(destination).write_bytes(b"another writer's result")
            real_link(source, destination)

        with patch("src.gn_comparison.os.link", side_effect=compete):
            with self.assertRaises(FileExistsError):
                run_comparison(*self.inputs, self.output, targets=("privacy",))
        self.assertEqual(self.output.read_bytes(), b"another writer's result")
        self.assertEqual({path.name for path in self.output.parent.iterdir()}, {self.output.name})


if __name__ == "__main__":
    unittest.main()
