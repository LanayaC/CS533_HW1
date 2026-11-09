"""Synthetic Part 2.5 checks; never read or analyze the assignment models."""

from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from gensim.models import KeyedVectors, Word2Vec
import numpy as np

from src.semantic_drift import (
    ADDITIONAL_WORDS, ANCHOR_COUNT, CONTROL_WORDS, DRIFT_CANDIDATES, OUTPUT_NAMES,
    TARGET_WORDS, analyze_models, checked_paths, cosine, fit_alignment,
    fit_procrustes, length_normalize_rows, main, measure_word, neighbor_overlap,
    rank_words, run_analysis, select_alignment_anchors, spearman_correlations,
)


ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def keyed_vectors(keys, values, counts=None):
    values = np.asarray(values, dtype=np.float32)
    vectors = KeyedVectors(vector_size=values.shape[1])
    vectors.add_vectors(keys, values)
    if counts is not None:
        for word, count in zip(keys, counts):
            vectors.set_vecattr(word, "count", count)
    return vectors


def synthetic_models():
    """Build small native models WITHOUT training; M3 has a known rotation."""
    keys = TARGET_WORDS + ("anchor_x", "anchor_y", "anchor_z", "filler")
    values = np.random.default_rng(533).normal(size=(len(keys), 3))
    for word, value in {
        "privacy": [1, 0, 0], "internet": [0, 1, 0], "phone": [1, 1, 0],
        "anchor_x": [1, 0, 0], "anchor_y": [0, 1, 0], "anchor_z": [0, 0, 1],
    }.items():
        values[keys.index(word)] = value
    models = []
    for index in range(3):
        chosen = [word for word in keys if not (index == 1 and word == "privacy")
                  and not (index == 2 and word == "booster")]
        counts = {word: 1_000_000 if word.startswith("anchor_") else 100 for word in chosen}
        model = Word2Vec(vector_size=3, min_count=1, workers=1)
        model.build_vocab_from_freq(counts)
        for word in chosen:
            vector = values[keys.index(word)]
            if index == 1:
                vector = vector + np.array([0.15, -0.1, 0.05])
            elif index == 2:
                vector = vector @ ROTATION
            model.wv[word] = vector
        models.append(model)
    return tuple(models)


class AlignmentTests(unittest.TestCase):
    def test_exact_word_list_and_production_anchor_count(self):
        expected = """zoom delta booster variant lockdown distancing mask quarantine
        positive spike remote essential trump pence meta ai chair stream streaming
        tweet platform podcast viral vaccine tesla uber apple cloud corona woke""".split()
        self.assertEqual(list(DRIFT_CANDIDATES), expected)
        self.assertEqual(CONTROL_WORDS, tuple("river piano mountain wine coffee bread doctor bridge baseball".split()))
        self.assertEqual(ADDITIONAL_WORDS, ("privacy", "internet", "phone", "election", "climate"))
        self.assertEqual(len(TARGET_WORDS), len(set(TARGET_WORDS)))
        self.assertEqual(len(TARGET_WORDS), 44)
        self.assertEqual(ANCHOR_COUNT, 20_000)

    def test_recovers_m3_to_m1_direction_and_correct_cosine(self):
        reference = np.array([[1., 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3]])
        source = reference @ ROTATION
        reference_before, source_before = reference.copy(), source.copy()
        fitted, diagnostics = fit_procrustes(source, reference)
        np.testing.assert_allclose(fitted, ROTATION.T, atol=1e-12)
        np.testing.assert_allclose(fitted.T @ fitted, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(source @ fitted, reference, atol=1e-12)
        self.assertLess(np.linalg.norm(source @ fitted - reference), 1e-10)
        self.assertGreater(np.linalg.norm(source @ fitted.T - reference), 1)
        self.assertAlmostEqual(cosine(reference[0], source[0]), 0)
        self.assertAlmostEqual(cosine(reference[0], source[0] @ fitted), 1)
        self.assertLess(diagnostics["orthogonality_error"], 1e-10)
        np.testing.assert_array_equal(reference, reference_before)
        np.testing.assert_array_equal(source, source_before)

    def test_fit_is_invariant_to_independent_anchor_lengths(self):
        random = np.random.default_rng(42)
        source, reference = random.normal(size=(8, 3)), random.normal(size=(8, 3))
        original, _ = fit_procrustes(source, reference)
        rescaled, _ = fit_procrustes(
            source * np.array([1, 1000, 0.1, 4, 0.5, 20, 2, 7])[:, None],
            reference * np.array([8, 0.2, 100, 1, 0.1, 5, 80, 3])[:, None],
        )
        np.testing.assert_allclose(original, rescaled, atol=1e-12)

    def test_orthogonal_reflections_are_allowed(self):
        reflection = np.diag([-1., 1., 1.])
        reference = np.eye(3)
        fitted, _ = fit_procrustes(reference @ reflection, reference)
        np.testing.assert_allclose(fitted, reflection, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(fitted), -1)

    def test_invalid_anchor_vectors_and_mismatched_shapes_fail_explicitly(self):
        for invalid in [[], [[0., 0, 0]], [[float("nan"), 1, 2]], [[float("inf"), 1, 2]]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                length_normalize_rows(invalid)
        with self.assertRaisesRegex(ValueError, "same shape"):
            fit_procrustes(np.eye(3), np.eye(2))

    def test_anchors_intersect_before_frequency_selection_and_tie_break(self):
        keys = ["absent_high", "low", "tie_first", "top", "tie_second"]
        m1 = keyed_vectors(keys, np.ones((5, 3)), [1000, 2, 10, 15, 10])
        m3 = keyed_vectors(keys[:0:-1], np.ones((4, 3)))
        self.assertEqual(select_alignment_anchors(m1, m3, 3), ("top", "tie_first", "tie_second"))
        with self.assertRaisesRegex(ValueError, "only 4"):
            select_alignment_anchors(m1, m3, 5)
        with self.assertRaisesRegex(ValueError, "positive"):
            select_alignment_anchors(m1, m3, 0)

    def test_missing_frequency_metadata_and_case_substitution_are_rejected(self):
        m1 = keyed_vectors(["apple"], [[1, 0, 0]])
        m3 = keyed_vectors(["Apple"], [[1, 0, 0]])
        with self.assertRaisesRegex(ValueError, "only 0"):
            select_alignment_anchors(m1, m3, 1)
        with self.assertRaisesRegex(ValueError, "frequencies"):
            select_alignment_anchors(m1, m1, 1)

    def test_alignment_uses_word_keys_even_when_vocabulary_orders_differ(self):
        m1, _, m3 = synthetic_models()
        rotation, report = fit_alignment(m1.wv, m3.wv, 3)
        np.testing.assert_allclose(rotation, ROTATION.T, atol=1e-12)
        self.assertEqual(report["anchor_count"], 3)
        self.assertEqual({item["word"] for item in report["anchors"]}, {"anchor_x", "anchor_y", "anchor_z"})
        self.assertEqual(report["direction"], "row_vector_M3 @ R -> M1")


class DriftMeasureTests(unittest.TestCase):
    def test_nn_overlap_is_exactly_set_intersection_over_ten(self):
        m1, m3 = MagicMock(), MagicMock()
        left = [f"a{i}" for i in range(10)]
        right = left[-3:] + [f"b{i}" for i in range(7)]
        for model, words in [(m1, left), (m3, right)]:
            model.__len__.return_value = 20
            model.__getitem__.return_value = np.array([1., 0, 0])
            model.most_similar.return_value = [(word, 0.9) for word in words]
        result = neighbor_overlap(m1, m3, "query")
        self.assertEqual(result["score"], 0.3)
        self.assertEqual(result["shared_neighbors"], left[-3:])
        self.assertEqual(result["m1_neighbors"], left)
        self.assertEqual(result["m3_neighbors"], right)
        m1.most_similar.assert_called_once_with("query", topn=10)
        m3.most_similar.assert_called_once_with("query", topn=10)

    def test_formulas_and_partial_missing_values_are_independent(self):
        m1, m2, m3 = synthetic_models()
        rotation, _ = fit_alignment(m1.wv, m3.wv, 3)
        row = measure_word("internet", m1.wv, m2.wv, m3.wv, rotation)
        left, right = m1.wv["internet"], m2.wv["internet"]
        self.assertAlmostEqual(row["cos_cont"], float(left @ right / np.linalg.norm(left) / np.linalg.norm(right)), places=6)
        self.assertAlmostEqual(row["raw_cos_m1_m3"], 0)
        self.assertAlmostEqual(row["cos_align"], 1)
        row = measure_word("privacy", m1.wv, m2.wv, m3.wv, rotation)
        self.assertEqual(row["missing_models"], ["M2"])
        self.assertIsNone(row["cos_cont"])
        self.assertEqual(row["cos_cont_status"], "missing: M2")
        self.assertAlmostEqual(row["cos_align"], 1)
        self.assertIsNotNone(row["nn_at_10"])
        row = measure_word("booster", m1.wv, m2.wv, m3.wv, rotation)
        self.assertEqual(row["missing_models"], ["M3"])
        self.assertIsNotNone(row["cos_cont"])
        self.assertIsNone(row["cos_align"])
        self.assertIsNone(row["nn_at_10"])
        row = measure_word("UNKNOWN", m1.wv, m2.wv, m3.wv, rotation)
        self.assertEqual(row["missing_models"], ["M1", "M2", "M3"])
        for field in ("cos_cont", "cos_align", "nn_at_10", "raw_cos_m1_m3"):
            self.assertIsNone(row[field])

    def test_zero_vectors_and_too_small_neighborhoods_are_undefined(self):
        vectors = keyed_vectors(["zero", "valid"], [[0, 0, 0], [1, 0, 0]], [10, 10])
        row = measure_word("zero", vectors, vectors, vectors, np.eye(3))
        self.assertIsNone(row["cos_cont"])
        self.assertTrue(row["cos_cont_status"].startswith("undefined:"))
        row = measure_word("valid", vectors, vectors, vectors, np.eye(3))
        self.assertAlmostEqual(row["cos_align"], 1)
        self.assertIsNone(row["nn_at_10"])
        self.assertIn("at least 11", row["nn_at_10_status"])

    def test_spearman_handles_ties_and_pairwise_missing_rows(self):
        rows = [
            dict(word="a", cos_cont=1., cos_align=1., nn_at_10=.3),
            dict(word="b", cos_cont=1., cos_align=2., nn_at_10=.2),
            dict(word="c", cos_cont=3., cos_align=3., nn_at_10=.1),
            dict(word="d", cos_cont=4., cos_align=4., nn_at_10=None),
            dict(word="e", cos_cont=None, cos_align=5., nn_at_10=0.),
        ]
        results = spearman_correlations(rows)
        self.assertEqual([record["n"] for record in results], [4, 3, 4])
        self.assertAlmostEqual(results[0]["rho"], 0.9486832980505138)
        self.assertAlmostEqual(results[1]["rho"], -0.8660254037844386)
        self.assertAlmostEqual(results[2]["rho"], -1.)
        self.assertEqual(results[1]["words"], ["a", "b", "c"])

    def test_undefined_correlations_use_json_null_not_nan(self):
        for rows, expected in [([], "insufficient_pairs"), ([
            dict(word="a", cos_cont=1., cos_align=1., nn_at_10=1.),
            dict(word="b", cos_cont=1., cos_align=1., nn_at_10=1.),
        ], "constant_input")]:
            records = spearman_correlations(rows)
            for record in records:
                self.assertEqual(record["status"], expected)
                self.assertIsNone(record["rho"])
            json.dumps(records, allow_nan=False)

    def test_rankings_use_aligned_cosine_five_each_and_alphabetical_ties(self):
        rows = [dict(word=f"w{i}", cos_align=i / 10, drift_rank=None) for i in range(10)]
        rows.append(dict(word="missing", cos_align=None, drift_rank=None))
        rankings = rank_words(list(reversed(rows)))
        self.assertEqual(rankings["most_drifted"], [f"w{i}" for i in range(5)])
        self.assertEqual(rankings["most_stable"], [f"w{i}" for i in range(9, 4, -1)])
        self.assertEqual(rankings["status"], "ok")
        self.assertIsNone(rows[-1]["drift_rank"])
        tied = [dict(word=word, cos_align=.5, drift_rank=None) for word in ["z", "b", "a"]]
        self.assertEqual(rank_words(tied, 1)["most_drifted"], ["a"])
        self.assertEqual(rank_words(tied, 1)["most_stable"], ["b"])
        shortage = rank_words(tied, 5)
        self.assertEqual(shortage["status"], "insufficient_rankable_words")
        self.assertFalse(set(shortage["most_drifted"]) & set(shortage["most_stable"]))

    def test_all_44_rows_examples_neighbors_and_vectors_are_preserved(self):
        models = synthetic_models()
        before = [model.wv.vectors.copy() for model in models]
        report = analyze_models(*models, anchor_count=3)
        self.assertEqual(report["target_count"], 44)
        self.assertEqual({row["word"] for row in report["words"]}, set(TARGET_WORDS))
        self.assertEqual(len(report["rankings"]["most_drifted"]), 5)
        self.assertEqual(len(report["rankings"]["most_stable"]), 5)
        self.assertEqual(len(report["selected_neighbors"]), 10)
        for entry in report["selected_neighbors"]:
            self.assertEqual(len(entry["m1"]["neighbors"]), 8)
            if entry["word"] == "privacy":
                self.assertEqual(entry["m2"]["status"], "missing")
            else:
                self.assertEqual(len(entry["m2"]["neighbors"]), 8)
        for example in report["raw_vs_aligned_examples"]:
            self.assertAlmostEqual(example["raw_cos_m1_m3"], 0)
            self.assertAlmostEqual(example["cos_align"], 1)
        defined = [row["cos_align"] for row in report["words"] if row["cos_align"] is not None]
        self.assertEqual(defined, sorted(defined))
        json.dumps(report, allow_nan=False)
        for model, original in zip(models, before):
            np.testing.assert_array_equal(model.wv.vectors, original)

    def test_selected_word_missing_m2_has_explicit_neighbor_status(self):
        models = synthetic_models()
        report = analyze_models(*models, targets=("privacy",), anchor_count=3, selection_count=1)
        self.assertEqual(report["selected_neighbors"][0]["m2"], {"status": "missing", "neighbors": None})
        self.assertEqual(report["rankings"]["status"], "insufficient_rankable_words")


class DriftFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = tuple(self.root / name for name in ("m1.model", "m2.model", "m3.model"))
        for model, path in zip(synthetic_models(), self.inputs):
            model.save(str(path), separately=[], sep_limit=2**63)
        self.output = self.root / "results"

    def test_dry_run_does_not_load_analyze_or_create_results(self):
        output = io.StringIO()
        with patch("src.semantic_drift.Word2Vec.load") as load, patch("src.semantic_drift.analyze_models") as analyze:
            with redirect_stdout(output):
                result = main([
                    "--m1", str(self.inputs[0]), "--m2", str(self.inputs[1]),
                    "--m3", str(self.inputs[2]), "--output-dir", str(self.output), "--dry-run",
                ])
            load.assert_not_called()
            analyze.assert_not_called()
        preview = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(preview["target_count"], 44)
        self.assertEqual(preview["anchor_count"], 20_000)
        self.assertFalse(self.output.exists())

    def test_saved_json_csv_match_and_input_models_are_byte_unchanged(self):
        before = {path: path.read_bytes() for path in self.inputs}
        with patch.object(Word2Vec, "train", side_effect=AssertionError("No training")) as train:
            with patch.object(Word2Vec, "save", side_effect=AssertionError("No model saving")) as save:
                report = run_analysis(*self.inputs, self.output, anchor_count=3)
            train.assert_not_called()
            save.assert_not_called()
        saved = json.loads((self.output / "semantic_drift.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, report)
        with (self.output / "semantic_drift.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 44)
        indexed = {row["word"]: row for row in rows}
        self.assertEqual(indexed["privacy"]["cos_cont"], "")
        self.assertEqual(indexed["privacy"]["missing_models"], "M2")
        self.assertEqual(indexed["booster"]["cos_align"], "")
        self.assertEqual(indexed["booster"]["cos_align_status"], "missing: M3")
        self.assertEqual({path.name for path in self.output.iterdir()}, set(OUTPUT_NAMES))
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_output_protection_and_input_validation_run_before_loading(self):
        inside = Path(__file__).resolve().parents[1] / "outputs" / "forbidden"
        with self.assertRaisesRegex(ValueError, "outside"):
            checked_paths(*self.inputs, inside)
        with self.assertRaisesRegex(ValueError, "different"):
            checked_paths(self.inputs[0], self.inputs[0], self.inputs[2], self.output)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            checked_paths(self.root / "absent", *self.inputs[1:], self.output)
        self.output.mkdir()
        for name in (*OUTPUT_NAMES, ".semantic_drift.in_progress"):
            existing = self.output / name
            existing.write_bytes(b"preserve")
            with self.subTest(name=name), patch("src.semantic_drift.Word2Vec.load") as load:
                with self.assertRaises(FileExistsError):
                    run_analysis(*self.inputs, self.output)
                self.assertEqual(existing.read_bytes(), b"preserve")
                load.assert_not_called()
            existing.unlink()

    def test_insufficient_real_anchors_fails_without_publishing_results(self):
        with self.assertRaisesRegex(ValueError, "20000 shared"):
            run_analysis(*self.inputs, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_publication_error_rolls_back_only_this_attempt(self):
        real_link = os.link

        def fail_second(source, destination):
            if str(destination).endswith(".json"):
                raise OSError("synthetic publication failure")
            real_link(source, destination)

        with patch("src.semantic_drift.os.link", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "publication failure"):
                run_analysis(*self.inputs, self.output, anchor_count=3)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_concurrent_result_is_not_overwritten_or_removed(self):
        real_link = os.link

        def competing_writer(source, destination):
            Path(destination).write_bytes(b"another writer's result")
            real_link(source, destination)

        with patch("src.semantic_drift.os.link", side_effect=competing_writer):
            with self.assertRaises(FileExistsError):
                run_analysis(*self.inputs, self.output, anchor_count=3)
        self.assertEqual((self.output / "semantic_drift.csv").read_bytes(), b"another writer's result")
        self.assertEqual({path.name for path in self.output.iterdir()}, {"semantic_drift.csv"})


if __name__ == "__main__":
    unittest.main()
