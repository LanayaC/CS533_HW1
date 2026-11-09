"""Part 2.2: continue a saved M1 on P2 and save a separate native model.

Use --dry-run to validate paths and inspect M1 without updating its vocabulary,
training, or creating output files. See notes/m2_training.md for commands.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
from time import perf_counter

import gensim
from gensim.models import Word2Vec, word2vec

from src.train_m1 import _checked_paths, hardware_info


def _continuation_paths(source_model_path, corpus_path, model_path):
    source = Path(source_model_path).resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"M1 must be a nonempty existing model file: {source}")
    if source == Path(model_path).resolve():
        raise ValueError("The continued model must use a different path from M1.")
    corpus, model, marker = _checked_paths(corpus_path, model_path)
    return source, corpus, model, marker


def _inherited_settings(model):
    return {
        name: getattr(model, name)
        for name in (
            "vector_size", "window", "sg", "negative", "min_count", "sample",
            "epochs", "seed", "workers",
        )
    }


def train_m2(source_model_path, corpus_path, model_path) -> dict:
    """Load M1 into memory, update on P2, and publish a new model and report.

    Inherit M1's settings without overrides. Follow the screenshot's calls,
    including total_words, which gensim requires for corpus_file training.
    Time vocabulary updating plus training, excluding loading and saving.
    Use M1's staging/publication pattern to refuse existing output artifacts.
    The source model is loaded only; saving always uses the new output path.
    """
    source, corpus, model_path, marker = _continuation_paths(
        source_model_path, corpus_path, model_path,
    )
    if word2vec.CORPUSFILE_VERSION < 0:
        raise RuntimeError("This gensim installation lacks compiled corpus_file training.")
    model = Word2Vec.load(str(source))
    original_vocabulary_size = len(model.wv)
    settings = _inherited_settings(model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    marker.mkdir()
    report_path = Path(str(model_path) + ".training.json")
    staged_paths = {model_path: marker / model_path.name, report_path: marker / report_path.name}
    try:
        before = hardware_info()
        started = perf_counter()
        model.build_vocab(corpus_file=str(corpus), update=True)
        updated_vocabulary_size = len(model.wv)
        model.train(
            corpus_file=str(corpus), total_examples=model.corpus_count,
            total_words=model.corpus_total_words, epochs=model.epochs,
        )
        training_seconds = perf_counter() - started
        report = {
            "source_model_path": str(source),
            "corpus_path": str(corpus),
            "model_path": str(model_path),
            "report_path": str(report_path),
            "settings": settings,
            "gensim_version": gensim.__version__,
            "original_vocabulary_size": original_vocabulary_size,
            "updated_vocabulary_size": updated_vocabulary_size,
            "new_words_added": updated_vocabulary_size - original_vocabulary_size,
            "training_seconds": training_seconds,
            "timing_definition": "P2 vocabulary update plus training; excludes M1 loading and saving",
            "corpus_articles": int(model.corpus_count),
            "corpus_tokens": int(model.corpus_total_words),
            "hardware_before_training": before,
            "hardware_after_training": hardware_info(),
            "reproducibility": "inherited seed; workers>1 makes training nondeterministic",
        }
        model.save(str(staged_paths[model_path]), separately=[], sep_limit=2**63)
        if {path.name for path in marker.iterdir()} != {model_path.name}:
            raise RuntimeError("Unexpected model sidecars; refusing incomplete publication.")
        with staged_paths[model_path].open("r+b") as handle:
            os.fsync(handle.fileno())
        report["model_file_bytes"] = staged_paths[model_path].stat().st_size
        with staged_paths[report_path].open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for final, staged in staged_paths.items():
            os.link(staged, final)
    except BaseException:
        for final, staged in staged_paths.items():
            if staged.exists() and final.exists() and final.samefile(staged):
                final.unlink()
        # Remove only this attempt's exclusively created staging directory.
        shutil.rmtree(marker)
        raise
    shutil.rmtree(marker)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True, help="Existing M1; read only")
    parser.add_argument("--corpus", type=Path, required=True, help="P2 corpus file")
    parser.add_argument("--model", type=Path, required=True, help="New model outside the repository")
    parser.add_argument("--dry-run", action="store_true", help="Inspect M1 and paths without training")
    args = parser.parse_args(argv)
    if args.dry_run:
        source, corpus, model_path, _ = _continuation_paths(
            args.source_model, args.corpus, args.model,
        )
        if word2vec.CORPUSFILE_VERSION < 0:
            raise RuntimeError("This gensim installation lacks compiled corpus_file training.")
        model = Word2Vec.load(str(source))
        print(json.dumps({
            "dry_run": True,
            "source_model_path": str(source), "corpus_path": str(corpus),
            "model_path": str(model_path),
            "original_vocabulary_size": len(model.wv),
            "settings": _inherited_settings(model),
            "compiled_corpusfile_version": word2vec.CORPUSFILE_VERSION,
        }, indent=2))
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(train_m2(args.source_model, args.corpus, args.model), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
