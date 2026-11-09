"""Part 2.8 tests use tiny temporary models; never analyze the actual models."""

from contextlib import redirect_stdout
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

from src.temporal_analogies import (
    QUERY_WORDS, analyze_temporal_models, checked_paths, emerging_words,
    load_part25_alignment, main, run_analysis, search_neighbors, validate_rotation,
)


ROTATION = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])


def vectors(keys, values, counts=None):
    values = np.asarray(values, dtype=np.float32)
    result = KeyedVectors(vector_size=values.shape[1])
    result.add_vectors(keys, values)
    if counts is not None:
        for word, count in zip(keys, counts):
            result.set_vecattr(word, "count", count)
    return result


def fixture():
    keys = ("anchor_x", "anchor_y", "anchor_z", "skype", "sars", "obama", "vine", "flickr",
            "old_a", "old_b", "old_c", "zoom", "far")
    random = np.random.default_rng(533)
    reference = {word: value for word, value in zip(keys, random.normal(size=(len(keys), 3)))}
    reference.update(anchor_x=np.array([1., 0, 0]), anchor_y=np.array([0., 1, 0]), anchor_z=np.array([0., 0, 1]),
                     skype=np.array([.9, .2, .1]), sars=np.array([-.8, .1, 0.]))
    reference["zoom"] = reference["skype"].copy()
    reference["far"] = -reference["sars"]
    continued = {**reference, **{word: random.normal(size=3) for word in QUERY_WORDS if word not in reference}}
    continued.update(covid=reference["sars"], zoom=reference["skype"], only_m2=reference["skype"])
    fresh = {word: value @ ROTATION for word, value in continued.items() if word != "only_m2"}
    # Check independent availability: absent in continued M2, present in fresh M3.
    del continued["brexit"]
    models = []
    for mapping in (reference, continued, fresh):
        labels = tuple(mapping)
        counts = [1_000_000 if word.startswith("anchor_") else 100 + index for index, word in enumerate(labels)]
        models.append(SimpleNamespace(wv=vectors(labels, list(mapping.values()), counts)))
    return tuple(models)


class EmergingWordsTests(unittest.TestCase):
    def test_top_twenty_uses_m3_counts_after_absence_check_without_filtering(self):
        m1 = vectors(["old"], [[1, 0, 0]], [1])
        labels = ["old", "123", "the"] + [f"new{index:02}" for index in range(23)]
        counts = [100_000, 1000, 900] + list(range(23))
        m3 = vectors(labels, np.ones((len(labels), 3)), counts)
        report = emerging_words(m1, m3)
        self.assertEqual(report["available_count"], 25)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(len(report["words"]), 20)
        self.assertEqual([row["word"] for row in report["words"][:3]], ["123", "the", "new22"])
        self.assertNotIn("old", [row["word"] for row in report["words"]])
        self.assertEqual([row["m3_count"] for row in report["words"]], [1000, 900] + list(range(22, 4, -1)))

    def test_ties_shortage_and_missing_frequency_metadata_are_explicit(self):
        m1 = vectors(["old"], [[1, 0, 0]])
        m3 = vectors(["z", "a", "old"], np.ones((3, 3)), [10, 10, 100])
        result = emerging_words(m1, m3)
        self.assertEqual(result["status"], "fewer_than_requested")
        self.assertEqual([row["word"] for row in result["words"]], ["a", "z"])
        with self.assertRaisesRegex(ValueError, "frequency"):
            emerging_words(m1, vectors(["new"], [[1, 0, 0]]))


class TemporalSearchTests(unittest.TestCase):
    def test_exact_ten_query_words(self):
        self.assertEqual(QUERY_WORDS, ("covid", "zoom", "trump", "biden", "tiktok", "instagram", "netflix", "isis", "pandemic", "brexit"))

    def test_both_temporal_lists_exclude_query_and_search_only_m1(self):
        m1, m2, m3 = fixture()
        for source, rotation in ((m2.wv, None), (m3.wv, ROTATION.T)):
            with self.subTest(rotation=rotation):
                result = search_neighbors(source, m1.wv, "zoom", rotation)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(len(result["neighbors"]), 5)
                labels = [item["word"] for item in result["neighbors"]]
                self.assertNotIn("zoom", labels)
                self.assertNotIn("only_m2", labels)
                self.assertTrue(all(label in m1.wv for label in labels))

    def test_temporal_candidates_are_only_m1_even_when_query_is_absent_from_m1(self):
        m1, m2, _ = fixture()
        self.assertNotIn("covid", m1.wv)
        result = search_neighbors(m2.wv, m1.wv, "covid")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["neighbors"][0]["word"], "sars")
        self.assertTrue(all(item["word"] in m1.wv for item in result["neighbors"]))
        result = search_neighbors(m2.wv, m1.wv, "zoom")
        self.assertNotIn("zoom", [item["word"] for item in result["neighbors"]])
        self.assertNotIn("only_m2", [item["word"] for item in result["neighbors"]])

    def test_fresh_query_uses_correct_row_rotation_direction(self):
        m1, _, m3 = fixture()
        aligned = search_neighbors(m3.wv, m1.wv, "covid", ROTATION.T)
        reversed_direction = search_neighbors(m3.wv, m1.wv, "covid", ROTATION)
        self.assertEqual(aligned["neighbors"][0]["word"], "sars")
        self.assertEqual(reversed_direction["neighbors"][0]["word"], "far")

    def test_missing_zero_queries_and_small_candidate_sets_have_statuses(self):
        m1, m2, _ = fixture()
        self.assertEqual(search_neighbors(m2.wv, m1.wv, "ZOOM")["status"], "missing_query_word")
        tiny = vectors(["zero", "one"], [[0, 0, 0], [1, 0, 0]])
        self.assertTrue(search_neighbors(tiny, m1.wv, "zero")["status"].startswith("undefined:"))
        self.assertEqual(search_neighbors(tiny, tiny, "one")["status"], "insufficient_candidates")

    def test_invalid_rotation_is_rejected_instead_of_refitted(self):
        models = fixture()
        for matrix in (np.eye(2), np.eye(3) * 2, np.full((3, 3), np.nan), np.empty((0, 0))):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                analyze_temporal_models(*models, matrix)
        np.testing.assert_array_equal(validate_rotation(ROTATION.T), ROTATION.T)

    def test_pipeline_has_only_two_lists_per_query_and_preserves_model_vectors(self):
        models = fixture()
        before = [model.wv.vectors.copy() for model in models]
        rotation = ROTATION.T.copy()
        report = analyze_temporal_models(*models, rotation)
        self.assertEqual([row["word"] for row in report["queries"]], list(QUERY_WORDS))
        indexed = {row["word"]: row for row in report["queries"]}
        self.assertEqual(indexed["brexit"]["m1_from_m2"]["status"], "missing_query_word")
        self.assertEqual(indexed["brexit"]["m1_from_m3_aligned"]["status"], "ok")
        self.assertIn("M1", indexed["covid"]["missing_query_models"])
        self.assertEqual(indexed["covid"]["m1_from_m2"]["neighbors"][0]["word"], "sars")
        self.assertEqual(indexed["covid"]["m1_from_m3_aligned"]["neighbors"][0]["word"], "sars")
        self.assertNotIn("hypotheses", report)
        for row in report["queries"]:
            self.assertEqual(set(row), {"word", "missing_query_models", "m1_from_m2", "m1_from_m3_aligned"})
        np.testing.assert_array_equal(rotation, ROTATION.T)
        for model, original in zip(models, before):
            np.testing.assert_array_equal(model.wv.vectors, original)
        json.dumps(report, allow_nan=False)


class TemporalFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = tuple(self.root / name for name in ("m1.model", "m2.model", "m3.model"))
        for wrapper, path in zip(fixture(), self.inputs):
            model = Word2Vec(vector_size=3, min_count=1, workers=1)
            model.build_vocab_from_freq({word: int(wrapper.wv.get_vecattr(word, "count")) for word in wrapper.wv.index_to_key})
            for word in wrapper.wv.index_to_key:
                model.wv[word] = wrapper.wv[word]
            model.save(str(path), separately=[], sep_limit=2**63)
        self.output = self.root / "results" / "temporal_analogies.json"
        self.alignment = self.root / "semantic_drift.json"
        self.alignment.write_text(json.dumps({
            "model_paths": {"M1": str(self.inputs[0]), "M3": str(self.inputs[2])},
            "alignment": {
                "direction": "row_vector_M3 @ R -> M1", "anchor_count": 3,
                "rotation": ROTATION.T.tolist(),
            },
        }), encoding="utf-8")

    def test_dry_run_only_checks_paths_and_prints_plan(self):
        stdout = io.StringIO()
        with patch("src.temporal_analogies.Word2Vec.load") as load, patch("src.temporal_analogies.analyze_temporal_models") as analyze:
            with redirect_stdout(stdout):
                result = main([
                    "--m1", str(self.inputs[0]), "--m2", str(self.inputs[1]), "--m3", str(self.inputs[2]),
                    "--alignment", str(self.alignment), "--output", str(self.output), "--dry-run",
                ])
            load.assert_not_called()
            analyze.assert_not_called()
        preview = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(preview["emerging_count"], 20)
        self.assertEqual(preview["neighbors_per_search"], 5)
        self.assertEqual(preview["alignment"]["report_path"], str(self.alignment))
        self.assertEqual(len(preview["searches"]), 2)
        self.assertNotIn("hypotheses", preview)
        self.assertFalse(self.output.parent.exists())

    def test_compact_saved_results_preserve_all_model_bytes_without_training(self):
        before = {path: path.read_bytes() for path in (*self.inputs, self.alignment)}
        with patch.object(Word2Vec, "train", side_effect=AssertionError("No training")) as train:
            with patch.object(Word2Vec, "save", side_effect=AssertionError("No model writes")) as save:
                with patch("numpy.linalg.svd", side_effect=AssertionError("Reuse saved R")) as svd:
                    report = run_analysis(*self.inputs, self.output, alignment_path=self.alignment)
                    svd.assert_not_called()
            train.assert_not_called()
            save.assert_not_called()
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), report)
        self.assertEqual(report["alignment"]["report_path"], str(self.alignment))
        self.assertNotIn("hypotheses", report)
        self.assertLess(self.output.stat().st_size, 50_000)
        self.assertEqual({path.name for path in self.output.parent.iterdir()}, {self.output.name})
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_invalid_and_existing_paths_are_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, "input model"):
            checked_paths(*self.inputs, self.inputs[0], alignment_path=self.alignment)
        with self.assertRaisesRegex(ValueError, "alignment report"):
            checked_paths(*self.inputs, self.alignment, alignment_path=self.alignment)
        inside = Path(__file__).resolve().parents[1] / "results" / "forbidden.json"
        with self.assertRaisesRegex(ValueError, "outside"):
            checked_paths(*self.inputs, inside, alignment_path=self.alignment)
        self.output.parent.mkdir()
        marker = self.output.parent / f".{self.output.name}.in_progress"
        for path in (self.output, marker):
            path.write_bytes(b"preserve")
            with self.subTest(path=path), patch("src.temporal_analogies.Word2Vec.load") as load:
                with self.assertRaises(FileExistsError):
                    run_analysis(*self.inputs, self.output, alignment_path=self.alignment)
                load.assert_not_called()
                self.assertEqual(path.read_bytes(), b"preserve")
            path.unlink()

    def test_analysis_failure_does_not_publish_results_or_retry(self):
        with patch("src.temporal_analogies.analyze_temporal_models", side_effect=ValueError("synthetic failure")) as analyze:
            with self.assertRaisesRegex(ValueError, "synthetic failure"):
                run_analysis(*self.inputs, self.output, alignment_path=self.alignment)
            analyze.assert_called_once()
        self.assertEqual(list(self.output.parent.iterdir()), [])

    def test_concurrent_result_is_not_overwritten_or_removed(self):
        real_link = os.link

        def compete(source, destination):
            Path(destination).write_bytes(b"another writer's result")
            real_link(source, destination)

        with patch("src.temporal_analogies.os.link", side_effect=compete):
            with self.assertRaises(FileExistsError):
                run_analysis(*self.inputs, self.output, alignment_path=self.alignment)
        self.assertEqual(self.output.read_bytes(), b"another writer's result")
        self.assertEqual({path.name for path in self.output.parent.iterdir()}, {self.output.name})

    def test_saved_rotation_direction_and_source_model_paths_are_validated(self):
        rotation, metadata = load_part25_alignment(self.alignment, self.inputs[0], self.inputs[2])
        np.testing.assert_array_equal(rotation, ROTATION.T)
        self.assertEqual(metadata["direction"], "row_vector_M3 @ R -> M1")
        with self.assertRaisesRegex(ValueError, "M3 path"):
            load_part25_alignment(self.alignment, self.inputs[0], self.inputs[1])
        data = json.loads(self.alignment.read_text(encoding="utf-8"))
        data["alignment"]["direction"] = "row_vector_M1 @ R -> M3"
        self.alignment.write_text(json.dumps(data), encoding="utf-8")
        with patch("src.temporal_analogies.Word2Vec.load") as load:
            with self.assertRaisesRegex(ValueError, "map M3"):
                run_analysis(*self.inputs, self.output, alignment_path=self.alignment)
            load.assert_not_called()
        self.assertFalse(self.output.parent.exists())


if __name__ == "__main__":
    unittest.main()
