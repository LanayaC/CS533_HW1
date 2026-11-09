"""Part 2.2 tests use temporary synthetic inputs, never the real M1 or P2."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, call, patch

from gensim.models import Word2Vec
import numpy as np

from src.train_m1 import assignment_settings
from src.train_m2 import main, train_m2


class TrainM2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "w2v_P1.model"
        self.source.write_bytes(b"synthetic source placeholder for mocked loading")
        self.corpus = self.root / "corpus_P2.txt"
        self.corpus.write_text(
            "government economy baseball technology\n" * 40
            + "government rareword\n" * 5, encoding="utf-8",
        )
        self.output = self.root / "models" / "w2v_P2_continued.model"

    def fake_model(self):
        model = MagicMock()
        for name, value in assignment_settings().items():
            setattr(model, name, value)
        model.wv.__len__.return_value = 13
        model.corpus_count, model.corpus_total_words = 40, 520

        def update_vocabulary(**kwargs):
            model.wv.__len__.return_value = 15
            model.corpus_count, model.corpus_total_words = 45, 170

        def save_model(path, **kwargs):
            self.assertEqual(kwargs, {"separately": [], "sep_limit": 2**63})
            Path(path).write_bytes(b"complete continued synthetic model")

        model.build_vocab.side_effect = update_vocabulary
        model.save.side_effect = save_model
        return model

    def test_exact_update_and_train_calls_use_updated_p2_counts(self):
        model = self.fake_model()
        source_before = self.source.read_bytes()
        with patch("src.train_m2.Word2Vec.load", return_value=model) as load:
            with patch("src.train_m2.perf_counter", side_effect=[100.0, 109.0]):
                report = train_m2(self.source, self.corpus, self.output)
        load.assert_called_once_with(str(self.source.resolve()))
        self.assertEqual(model.method_calls[:2], [
            call.build_vocab(corpus_file=str(self.corpus.resolve()), update=True),
            call.train(
                corpus_file=str(self.corpus.resolve()), total_examples=45,
                total_words=170, epochs=5,
            ),
        ])
        model.save.assert_called_once()
        self.assertEqual(report["original_vocabulary_size"], 13)
        self.assertEqual(report["updated_vocabulary_size"], 15)
        self.assertEqual(report["new_words_added"], 2)
        self.assertEqual(report["training_seconds"], 9.0)
        self.assertEqual(report["settings"], assignment_settings())
        self.assertEqual(self.source.read_bytes(), source_before)

    def test_dry_run_loads_but_never_updates_trains_saves_or_creates_output(self):
        model = self.fake_model()
        output = io.StringIO()
        with patch("src.train_m2.Word2Vec.load", return_value=model), redirect_stdout(output):
            result = main([
                "--source-model", str(self.source), "--corpus", str(self.corpus),
                "--model", str(self.output), "--dry-run",
            ])
        preview = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertTrue(preview["dry_run"])
        self.assertEqual(preview["original_vocabulary_size"], 13)
        self.assertEqual(preview["settings"], assignment_settings())
        model.build_vocab.assert_not_called()
        model.train.assert_not_called()
        model.save.assert_not_called()
        self.assertFalse(self.output.parent.exists())

    def test_nonempty_source_and_corpus_are_required_before_loading(self):
        empty = self.root / "empty"
        empty.touch()
        missing = self.root / "missing"
        for source, corpus in [
            (missing, self.corpus), (empty, self.corpus),
            (self.source, missing), (self.source, empty),
        ]:
            with self.subTest(source=source, corpus=corpus):
                with patch("src.train_m2.Word2Vec.load") as load:
                    with self.assertRaises(ValueError):
                        train_m2(source, corpus, self.output)
                    load.assert_not_called()
        self.assertFalse(self.output.parent.exists())

    def test_m1_cannot_be_the_output_and_repository_outputs_are_rejected(self):
        source_before = self.source.read_bytes()
        inside = Path(__file__).resolve().parents[1] / "models" / "forbidden.model"
        for destination, message in [(self.source, "different path"), (inside, "outside")]:
            with self.subTest(destination=destination), patch("src.train_m2.Word2Vec.load") as load:
                with self.assertRaisesRegex(ValueError, message):
                    train_m2(self.source, self.corpus, destination)
                load.assert_not_called()
        self.assertEqual(self.source.read_bytes(), source_before)

    def test_existing_output_sidecar_report_or_marker_is_preserved(self):
        self.output.parent.mkdir()
        for name in [
            self.output.name, self.output.name + ".wv.vectors.npy",
            self.output.name + ".training.json", "." + self.output.name + ".training",
        ]:
            existing = self.output.parent / name
            existing.write_bytes(b"preserve")
            with self.subTest(name=name), patch("src.train_m2.Word2Vec.load") as load:
                with self.assertRaises(FileExistsError):
                    train_m2(self.source, self.corpus, self.output)
                load.assert_not_called()
                self.assertEqual(existing.read_bytes(), b"preserve")
            existing.unlink()

    def test_failures_and_interrupts_do_not_publish_or_retry(self):
        source_before = self.source.read_bytes()
        for stage in ["build_vocab", "train", "save"]:
            for error in [RuntimeError("synthetic failure"), KeyboardInterrupt()]:
                model = self.fake_model()
                getattr(model, stage).side_effect = error
                with self.subTest(stage=stage, error=type(error).__name__):
                    with patch("src.train_m2.Word2Vec.load", return_value=model) as load:
                        with self.assertRaises(type(error)):
                            train_m2(self.source, self.corpus, self.output)
                    load.assert_called_once()
                    getattr(model, stage).assert_called_once()
                    self.assertEqual(list(self.output.parent.iterdir()), [])
                    self.assertEqual(self.source.read_bytes(), source_before)

    def test_publication_failure_rolls_back_this_attempt(self):
        model = self.fake_model()
        real_link = os.link

        def fail_report_publication(source, destination):
            if str(destination).endswith(".training.json"):
                raise OSError("synthetic report publication failure")
            real_link(source, destination)

        with patch("src.train_m2.Word2Vec.load", return_value=model):
            with patch("src.train_m2.os.link", side_effect=fail_report_publication):
                with self.assertRaisesRegex(OSError, "publication failure"):
                    train_m2(self.source, self.corpus, self.output)
        self.assertEqual(list(self.output.parent.iterdir()), [])

    def test_real_continuation_is_reloadable_and_preserves_m1_and_corpus(self):
        p1 = self.root / "corpus_P1.txt"
        words = "government economy policy budget market jobs growth trade taxes finance business industry health"
        p1.write_text((words + "\n") * 40, encoding="utf-8")
        with patch("src.train_m1.os.cpu_count", return_value=1):
            settings = assignment_settings()
        original = Word2Vec(corpus_file=str(p1), **settings)
        original.save(str(self.source), separately=[], sep_limit=2**63)
        source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        corpus_before = self.corpus.read_bytes()
        report = train_m2(self.source, self.corpus, self.output)
        continued = Word2Vec.load(str(self.output))
        preserved = Word2Vec.load(str(self.source))
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), source_hash)
        self.assertEqual(self.corpus.read_bytes(), corpus_before)
        self.assertEqual(len(preserved.wv), 13)
        np.testing.assert_array_equal(preserved.wv.vectors, original.wv.vectors)
        self.assertTrue(set(original.wv.key_to_index).issubset(continued.wv.key_to_index))
        self.assertEqual(set(continued.wv.key_to_index) - set(original.wv.key_to_index), {"baseball", "technology"})
        self.assertNotIn("rareword", continued.wv)
        self.assertFalse(np.array_equal(continued.wv["government"], original.wv["government"]))
        for name, expected in settings.items():
            self.assertEqual(getattr(continued, name), expected)
        self.assertEqual(report["original_vocabulary_size"], 13)
        self.assertEqual(report["updated_vocabulary_size"], 15)
        self.assertEqual(report["new_words_added"], 2)
        self.assertEqual(report["corpus_articles"], 45)
        self.assertEqual(report["corpus_tokens"], 170)
        self.assertGreater(report["training_seconds"], 0)
        self.assertEqual(report["model_file_bytes"], self.output.stat().st_size)
        self.assertEqual(json.loads(Path(report["report_path"]).read_text(encoding="utf-8")), report)
        self.assertEqual(
            {path.name for path in self.output.parent.iterdir()},
            {self.output.name, self.output.name + ".training.json"},
        )


if __name__ == "__main__":
    unittest.main()
