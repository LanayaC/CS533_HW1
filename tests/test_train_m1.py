"""Part 2.1/2.3 fresh-training checks using only tiny synthetic corpora."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from gensim.models import Word2Vec

from src.train_m1 import assignment_settings, main, nearest_neighbors, train_m1


class TrainM1Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "tiny.txt"
        words = "government economy baseball music technology policy budget market jobs growth trade taxes finance business industry health"
        self.corpus.write_text((words + "\n") * 40, encoding="utf-8")
        self.model_path = self.root / "models" / "w2v_P1.model"

    def test_exact_assignment_settings_and_cpu_worker_count(self):
        with patch("src.train_m1.os.cpu_count", return_value=3):
            self.assertEqual(assignment_settings(), {
                "vector_size": 300, "window": 5, "sg": 1, "negative": 5,
                "min_count": 10, "sample": 1e-3, "epochs": 5, "seed": 42,
                "workers": 3,
            })
        with patch("src.train_m1.os.cpu_count", return_value=None):
            with self.assertRaises(RuntimeError):
                assignment_settings()

    def test_dry_run_never_trains_or_creates_model_directory(self):
        words = ["government", "economy", "baseball", "music", "technology"]
        for word_args in [[], ["--words", *words]]:
            with self.subTest(word_args=word_args):
                output = io.StringIO()
                with patch("src.train_m1.Word2Vec") as constructor, redirect_stdout(output):
                    result = main([
                        "--corpus", str(self.corpus), "--model", str(self.model_path),
                        "--dry-run", *word_args,
                    ])
                self.assertEqual(result, 0)
                preview = json.loads(output.getvalue())
                self.assertTrue(preview["dry_run"])
                self.assertEqual(preview["example_words"], words)
                constructor.assert_not_called()
                self.assertFalse(self.model_path.parent.exists())

    def test_existing_model_sidecar_report_or_marker_is_not_overwritten(self):
        self.model_path.parent.mkdir()
        names = [
            "w2v_P1.model", "w2v_P1.model.wv.vectors.npy",
            "w2v_P1.model.training.json", ".w2v_P1.model.training",
        ]
        for name in names:
            existing = self.model_path.parent / name
            existing.write_bytes(b"preserve")
            with self.subTest(name=name), patch("src.train_m1.Word2Vec") as constructor:
                with self.assertRaises(FileExistsError):
                    train_m1(self.corpus, self.model_path)
                constructor.assert_not_called()
            self.assertEqual(existing.read_bytes(), b"preserve")
            existing.unlink()

    def test_models_inside_repository_and_stopword_examples_are_rejected(self):
        inside = Path(__file__).resolve().parents[1] / "models" / "forbidden.model"
        with self.assertRaisesRegex(ValueError, "outside"):
            train_m1(self.corpus, inside)
        with self.assertRaisesRegex(ValueError, "non-stop"):
            train_m1(self.corpus, self.model_path, neighbor_words=("the", "economy", "baseball", "music", "technology"))
        for words in [
            ("government", "economy"),
            ("government", "economy", "baseball", "music", "music"),
        ]:
            with self.subTest(words=words), self.assertRaisesRegex(ValueError, "five different"):
                train_m1(self.corpus, self.model_path, neighbor_words=words)
        self.assertFalse(self.model_path.parent.exists())

    def test_neighbors_preserve_direct_top_eight_without_filtering(self):
        model = MagicMock()
        model.wv.__contains__.return_value = True
        model.wv.__len__.return_value = 15
        model.wv.get_vecattr.return_value = 100
        terms = ["the", "and", "123", "x", "policy", "budget", "market", "jobs"]
        candidates = [(word, 0.95 - i / 100) for i, word in enumerate(terms)]
        model.wv.most_similar.return_value = candidates
        result = nearest_neighbors(model, ["government"])[0]
        self.assertEqual(result["corpus_frequency"], 100)
        self.assertEqual(
            [(row["word"], row["cosine_similarity"]) for row in result["neighbors"]],
            candidates,
        )
        model.wv.most_similar.assert_called_once_with("government", topn=8)

    def test_tiny_real_training_is_reloadable_and_reports_correct_metadata(self):
        corpus_before = self.corpus.read_bytes()
        with patch("src.train_m1.os.cpu_count", return_value=2):
            report = train_m1(self.corpus, self.model_path)
        loaded = Word2Vec.load(str(self.model_path))
        for name, expected in {
            "vector_size": 300, "window": 5, "sg": 1, "negative": 5,
            "min_count": 10, "sample": 1e-3, "epochs": 5, "seed": 42, "workers": 2,
        }.items():
            self.assertEqual(getattr(loaded, name), expected)
        self.assertEqual(report["vocabulary_size"], 16)
        self.assertEqual(report["corpus_articles"], 40)
        self.assertEqual(report["corpus_tokens"], 640)
        self.assertGreater(report["training_seconds"], 0)
        self.assertEqual(self.corpus.read_bytes(), corpus_before)
        self.assertEqual(
            {path.name for path in self.model_path.parent.iterdir()},
            {"w2v_P1.model", "w2v_P1.model.training.json"},
        )
        persisted = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(persisted, report)
        self.assertEqual(
            [example["word"] for example in report["nearest_neighbors"]],
            ["government", "economy", "baseball", "music", "technology"],
        )
        self.assertEqual(report["neighbor_filter"], "none; direct model.wv.most_similar(word, topn=8)")
        for example in report["nearest_neighbors"]:
            self.assertEqual(example["corpus_frequency"], 40)
            self.assertEqual(len(example["neighbors"]), 8)
            expected = loaded.wv.most_similar(example["word"], topn=8)
            self.assertEqual([row["word"] for row in example["neighbors"]], [word for word, _ in expected])
            for row, (_, score) in zip(example["neighbors"], expected):
                self.assertAlmostEqual(row["cosine_similarity"], float(score))

    def test_training_failures_and_interrupts_leave_no_final_artifacts(self):
        for error in [RuntimeError("training failed"), KeyboardInterrupt()]:
            with self.subTest(error=type(error).__name__):
                with patch("src.train_m1.Word2Vec", side_effect=error):
                    with self.assertRaises(type(error)):
                        train_m1(self.corpus, self.model_path)
                self.assertEqual(list(self.model_path.parent.iterdir()), [])

    def test_p2_fresh_cli_trains_only_p2_and_preserves_existing_models(self):
        p2 = self.root / "corpus_P2.txt"
        words = "government economy baseball music technology innovation robotics computing discovery newtoken"
        p2.write_text((words + "\n") * 40, encoding="utf-8")
        p2_before = p2.read_bytes()
        self.model_path.parent.mkdir()
        # Existing-model sentinels must never be loaded, modified, or replaced.
        preserved = {
            self.model_path: b"preserve M1",
            Path(str(self.model_path) + ".training.json"): b"preserve M1 report",
            self.model_path.with_name("w2v_P2_continued.model"): b"preserve continued model",
            self.model_path.with_name("w2v_P2_continued.model.training.json"): b"preserve continued report",
        }
        for path, content in preserved.items():
            path.write_bytes(content)
        fresh = self.model_path.with_name("w2v_P2_fresh.model")
        output = io.StringIO()
        with patch("src.train_m1.os.cpu_count", return_value=2):
            settings = assignment_settings()
            with patch.object(Word2Vec, "load", side_effect=AssertionError("Fresh training must not load a model")) as load:
                with patch("src.train_m1.logging.basicConfig"), redirect_stdout(output):
                    result = main(["--corpus", str(p2), "--model", str(fresh)])
                load.assert_not_called()
        self.assertEqual(result, 0)
        report = json.loads(output.getvalue())
        model = Word2Vec.load(str(fresh))
        self.assertEqual(set(model.wv.key_to_index), set(words.split()))
        self.assertNotIn("health", model.wv)  # Present in the synthetic P1 fixture only.
        self.assertEqual(model.train_count, 1)
        self.assertEqual(report["vocabulary_size"], 10)
        self.assertEqual(report["corpus_articles"], 40)
        self.assertEqual(report["corpus_tokens"], 400)
        self.assertEqual(report["corpus_path"], str(p2.resolve()))
        self.assertEqual(report["model_path"], str(fresh.resolve()))
        self.assertEqual(report["settings"], settings)
        for name, expected in settings.items():
            self.assertEqual(getattr(model, name), expected)
        self.assertGreater(report["training_seconds"], 0)
        self.assertEqual(
            report["timing_definition"],
            "vocabulary building plus 5 epochs; excludes neighbors and saving",
        )
        self.assertEqual(json.loads(Path(report["report_path"]).read_text(encoding="utf-8")), report)
        self.assertFalse((fresh.parent / ("." + fresh.name + ".training")).exists())
        self.assertEqual(p2.read_bytes(), p2_before)
        for path, content in preserved.items():
            self.assertEqual(path.read_bytes(), content)
        for destination in [self.model_path, fresh.with_name("w2v_P2_continued.model"), fresh]:
            with self.subTest(destination=destination), patch("src.train_m1.Word2Vec") as constructor:
                with self.assertRaises(FileExistsError):
                    train_m1(p2, destination)
                constructor.assert_not_called()

    def test_p2_fresh_dry_run_resolves_paths_without_training_or_output(self):
        p2 = self.corpus.rename(self.root / "corpus_P2.txt")
        fresh = self.model_path.with_name("w2v_P2_fresh.model")
        output = io.StringIO()
        with patch("src.train_m1.Word2Vec") as constructor, redirect_stdout(output):
            result = main(["--corpus", str(p2), "--model", str(fresh), "--dry-run"])
        self.assertEqual(result, 0)
        preview = json.loads(output.getvalue())
        self.assertTrue(preview["dry_run"])
        self.assertEqual(preview["corpus_path"], str(p2.resolve()))
        self.assertEqual(preview["model_path"], str(fresh.resolve()))
        self.assertEqual(preview["settings"], assignment_settings())
        constructor.assert_not_called()
        self.assertFalse(fresh.parent.exists())

    def test_constructor_uses_corpus_file_and_serialization_disables_recursive_sidecars(self):
        fake = MagicMock()
        fake.wv.__len__.return_value = 16
        fake.corpus_count, fake.corpus_total_words = 40, 640
        fake.estimate_memory.return_value = {"total": 1000}

        def save_model(path, **kwargs):
            self.assertEqual(kwargs, {"separately": [], "sep_limit": 2**63})
            Path(path).write_bytes(b"complete synthetic model")

        fake.save.side_effect = save_model
        with patch("src.train_m1.Word2Vec", return_value=fake) as constructor:
            with patch("src.train_m1.nearest_neighbors", return_value=[]):
                train_m1(self.corpus, self.model_path)
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs.pop("corpus_file"), str(self.corpus.resolve()))
        self.assertEqual(kwargs, assignment_settings())


if __name__ == "__main__":
    unittest.main()
