"""Train only Part 2.1's M1, using the assignment's fixed Word2Vec settings.

Preview without training or creating output directories::

    python -m src.train_m1 --corpus ../data/corpora/corpus_P1.txt \
        --model ../data/models/w2v_P1.model --dry-run

Remove --dry-run only after approving the full training run. See notes/README.md
for result locations and CLI entry points. Existing models, sidecars, and
training reports are never replaced. All model artifacts must be outside this
repository.
"""

import argparse
import ctypes
import json
import logging
import os
from pathlib import Path
import platform
import shutil
from time import perf_counter

import gensim
from gensim.models import Word2Vec, word2vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


DEFAULT_NEIGHBOR_WORDS = ("government", "economy", "baseball", "music", "technology")


def assignment_settings() -> dict:
    """Return the exact required settings; workers equals detected CPU count."""
    cores = os.cpu_count()
    if cores is None or cores < 1:
        raise RuntimeError("Cannot determine CPU count for the required workers setting.")
    return {
        "vector_size": 300, "window": 5, "sg": 1, "negative": 5,
        "min_count": 10, "sample": 1e-3, "epochs": 5, "seed": 42,
        "workers": cores,
    }


def hardware_info() -> dict:
    """Read CPU/RAM metadata without requiring WMI or installing dependencies.

    psutil is optional; Windows has a standard-library native RAM fallback.
    Available memory is a snapshot, not a reservation or peak-memory estimate.
    """
    name = platform.processor() or platform.machine()
    if os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        except OSError:
            pass

    total = available = physical = None
    try:
        import psutil
        memory = psutil.virtual_memory()
        total, available = memory.total, memory.available
        physical = psutil.cpu_count(logical=False)
    except (ImportError, OSError):
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended", ctypes.c_ulonglong),
                ]
            memory = MemoryStatus()
            memory.length = ctypes.sizeof(memory)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                total, available = memory.total_phys, memory.avail_phys
    return {
        "cpu": name,
        "physical_cores": physical,
        "logical_cpu_count": os.cpu_count(),
        "ram_total_bytes": total,
        "ram_available_bytes": available,
        "ram_total_gib": total / 2**30 if total is not None else None,
        "ram_available_gib": available / 2**30 if available is not None else None,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _checked_paths(corpus_path, model_path):
    corpus, model = Path(corpus_path).resolve(), Path(model_path).resolve()
    repository = Path(__file__).resolve().parents[1]
    if model.is_relative_to(repository):
        raise ValueError("Save trained models outside the Git repository.")
    if not corpus.is_file() or corpus.stat().st_size == 0:
        raise ValueError(f"Corpus must be a nonempty existing file: {corpus}")
    marker = model.parent / f".{model.name}.training"
    if os.path.lexists(marker):
        raise FileExistsError(f"Incomplete or active training marker: {marker}")
    if model.parent.exists():
        for path in model.parent.iterdir():
            if path.name == model.name or path.name.startswith(model.name + "."):
                raise FileExistsError(f"Refusing to overwrite model/report/sidecar: {path}")
    return corpus, model, marker


def _checked_words(words):
    words = tuple(words)
    if len(words) != 5 or len(set(words)) != 5:
        raise ValueError("Choose exactly five different example words.")
    if any(not isinstance(word, str) or not word.isalpha() or len(word) < 2
           or word.lower() in ENGLISH_STOP_WORDS for word in words):
        raise ValueError("Example words must be alphabetic, non-stop words of length >= 2.")
    return tuple(word.lower() for word in words)


def nearest_neighbors(model, words) -> list[dict]:
    """Report the eight cosine neighbors returned by gensim without filtering."""
    records = []
    for word in words:
        if word not in model.wv:
            raise ValueError(f"Example word is absent from the trained vocabulary: {word}")
        if len(model.wv) - 1 < 8:
            raise ValueError(f"Fewer than 8 vocabulary neighbors available for {word}.")
        neighbors = [
            {"word": term, "cosine_similarity": float(score)}
            for term, score in model.wv.most_similar(word, topn=8)
        ]
        records.append({
            "word": word,
            "corpus_frequency": int(model.wv.get_vecattr(word, "count")),
            "neighbors": neighbors,
        })
    return records


def train_m1(corpus_path, model_path, neighbor_words=DEFAULT_NEIGHBOR_WORDS) -> dict:
    """Train M1 once and publish a complete native gensim model plus JSON report.

    Use Word2Vec(corpus_file=...) with all fixed assignment settings and no
    sampling of corpus articles. Training time includes vocabulary building
    and the five epochs; persistence and neighbor calculation are excluded.
    Retain all stop words during training and report neighbors without filtering.
    Multiworker training is not deterministic even with seed=42.

    Save a single native model containing its training state, never a vectors-
    only export. Staging and exclusive hard-link publication prevent replacing
    existing artifacts or exposing a partly written final model. The staging
    marker remains after an uncatchable termination and blocks another run.
    Caught failures roll back this attempt's published files. A valid output
    has the model and .training.json report with no marker. Requires hard-link
    support (available on this Windows NTFS filesystem).
    """
    corpus, model_path, marker = _checked_paths(corpus_path, model_path)
    words = _checked_words(neighbor_words)
    settings = assignment_settings()
    if word2vec.CORPUSFILE_VERSION < 0:
        raise RuntimeError("This gensim installation lacks compiled corpus_file training.")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    marker.mkdir()  # exclusive: never clean up another training attempt's marker
    report_path = Path(str(model_path) + ".training.json")
    staged_paths = {model_path: marker / model_path.name, report_path: marker / report_path.name}
    try:
        before = hardware_info()
        started = perf_counter()
        model = Word2Vec(corpus_file=str(corpus), **settings)
        training_seconds = perf_counter() - started
        neighbors = nearest_neighbors(model, words)
        report = {
            "corpus_path": str(corpus),
            "model_path": str(model_path),
            "report_path": str(report_path),
            "settings": settings,
            "gensim_version": gensim.__version__,
            "compiled_fast_version": word2vec.FAST_VERSION,
            "compiled_corpusfile_version": word2vec.CORPUSFILE_VERSION,
            "training_seconds": training_seconds,
            "timing_definition": "vocabulary building plus 5 epochs; excludes neighbors and saving",
            "vocabulary_size": len(model.wv),
            "corpus_articles": int(model.corpus_count),
            "corpus_tokens": int(model.corpus_total_words),
            "estimated_model_memory_bytes": model.estimate_memory(),
            "hardware_before_training": before,
            "hardware_after_training": hardware_info(),
            "nearest_neighbors": neighbors,
            "neighbor_filter": "none; direct model.wv.most_similar(word, topn=8)",
            "reproducibility": "seed=42; workers>1 makes training nondeterministic",
        }
        # separately=[] alone does not disable automatic splitting in recursive
        # gensim objects. A huge sep_limit also keeps their arrays in this file.
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
        # marker is a private subdirectory under the resolved external model
        # parent, created exclusively above; remove only this attempt's files.
        shutil.rmtree(marker)
        raise
    shutil.rmtree(marker)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--words", nargs=5, default=DEFAULT_NEIGHBOR_WORDS)
    parser.add_argument("--dry-run", action="store_true", help="Validate and show settings/resources only")
    args = parser.parse_args(argv)
    if args.dry_run:
        corpus, model, _ = _checked_paths(args.corpus, args.model)
        print(json.dumps({
            "dry_run": True, "corpus_path": str(corpus), "model_path": str(model),
            "settings": assignment_settings(), "example_words": _checked_words(args.words),
            "hardware": hardware_info(),
            "compiled_corpusfile_version": word2vec.CORPUSFILE_VERSION,
        }, indent=2))
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(train_m1(args.corpus, args.model, args.words), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
