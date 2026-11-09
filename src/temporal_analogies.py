"""Part 2.8: emerging vocabulary and temporal-role nearest-neighbor analogues.

Read saved M1, continued M2, and fresh M3 only. --dry-run checks paths and
validates the saved Part 2.5 alignment without loading models or retrieving words.
"""

import argparse
import json
import os
from pathlib import Path
import shutil

import gensim
from gensim.models import Word2Vec
import numpy as np

from src.semantic_drift import length_normalize_rows


QUERY_WORDS = ("covid", "zoom", "trump", "biden", "tiktok", "instagram", "netflix", "isis", "pandemic", "brexit")


def emerging_words(m1, m3, count=20):
    """Rank M3-only words by M3's saved frequency; alphabetical ties."""
    if count < 1:
        raise ValueError("Emerging-word count must be positive.")
    new = [word for word in m3.index_to_key if word not in m1]
    try:
        new.sort(key=lambda word: (-int(m3.get_vecattr(word, "count")), word))
    except KeyError as error:
        raise ValueError("M3 word frequency counts are required.") from error
    return {
        "requested_count": count, "available_count": len(new),
        "status": "ok" if len(new) >= count else "fewer_than_requested",
        "words": [{"rank": rank, "word": word, "m3_count": int(m3.get_vecattr(word, "count"))}
                  for rank, word in enumerate(new[:count], 1)],
    }


def search_neighbors(source, candidates, word, rotation=None):
    """Five cosine neighbors of a source query in the candidate vocabulary.

    Source and candidates may be different models. For fresh M3 queries,
    pass the M3-to-M1 row-vector rotation. Exclude the literal query label
    even when querying with a vector, which Gensim does not auto-exclude.
    """
    if word not in source:
        return {"status": "missing_query_word", "neighbors": None}
    try:
        query = np.asarray(source[word], dtype=np.float64)
        if rotation is not None:
            query = query @ rotation
        query = length_normalize_rows([query], f"query {word}")[0]
        available = len(candidates) - int(word in candidates)
        if available < 5:
            return {"status": "insufficient_candidates", "available_candidates": available, "neighbors": None}
        # At most one returned label is excluded, so six candidates suffice.
        matches = candidates.most_similar(positive=[query], topn=min(6, len(candidates)))
        matches = [(label, score) for label, score in matches if label != word][:5]
        if len(matches) != 5 or len({label for label, _ in matches}) != 5:
            raise ValueError("Expected five distinct non-query neighbors.")
        if not all(np.isfinite(score) for _, score in matches):
            raise ValueError("Nonfinite neighbor cosine.")
        return {"status": "ok", "neighbors": [
            {"word": label, "cosine_similarity": float(np.clip(score, -1., 1.))}
            for label, score in matches
        ]}
    except ValueError as error:
        return {"status": "undefined: " + str(error), "neighbors": None}


def validate_rotation(rotation, dimension=None):
    rotation = np.asarray(rotation, dtype=np.float64)
    if (rotation.ndim != 2 or rotation.shape[0] == 0
            or rotation.shape[0] != rotation.shape[1] or not np.isfinite(rotation).all()):
        raise ValueError("Saved alignment must be a finite square matrix.")
    if dimension is not None and rotation.shape != (dimension, dimension):
        raise ValueError("Saved alignment dimensions do not match the models.")
    if not np.allclose(rotation.T @ rotation, np.eye(rotation.shape[0]), rtol=0, atol=1e-8):
        raise ValueError("Saved alignment is not orthogonal.")
    return rotation


def load_part25_alignment(path, m1_path, m3_path):
    """Reuse the actual saved Part 2.5 R; validate direction and model paths."""
    path = Path(path).expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    alignment = report.get("alignment", {})
    if alignment.get("direction") != "row_vector_M3 @ R -> M1":
        raise ValueError("Part 2.5 alignment must map M3 @ R onto M1.")
    for label, model_path in (("M1", m1_path), ("M3", m3_path)):
        recorded = report.get("model_paths", {}).get(label)
        if recorded is None or Path(recorded).expanduser().resolve() != Path(model_path).expanduser().resolve():
            raise ValueError(f"Part 2.5 alignment {label} path does not match the selected model.")
    if "rotation" not in alignment:
        raise ValueError("Part 2.5 report has no saved rotation.")
    rotation = validate_rotation(alignment["rotation"])
    metadata = {key: value for key, value in alignment.items() if key not in {"rotation", "anchors"}}
    metadata["report_path"] = str(path)
    return rotation, metadata


def analyze_temporal_models(m1, m2, m3, rotation, *, queries=QUERY_WORDS, emerging_count=20):
    queries = tuple(queries)
    if len(set(queries)) != len(queries):
        raise ValueError("Query words must be unique.")
    if len({model.wv.vector_size for model in (m1, m2, m3)}) != 1:
        raise ValueError("All three models must have equal vector dimensions.")
    rotation = validate_rotation(rotation, m1.wv.vector_size)
    new_words = emerging_words(m1.wv, m3.wv, emerging_count)
    rows = []
    for word in queries:
        rows.append({
            "word": word,
            "missing_query_models": [label for label, model in zip(("M1", "M2", "M3"), (m1, m2, m3)) if word not in model.wv],
            "m1_from_m2": search_neighbors(m2.wv, m1.wv, word),
            "m1_from_m3_aligned": search_neighbors(m3.wv, m1.wv, word, rotation),
        })
    return {
        "definitions": {
            "emerging": "Top 20 M3 words absent from M1; descending saved M3 count, alphabetical ties; no filtering",
            "m1_from_m2": "Five M1 words closest to continued M2[w], excluding w (professor's temporal analogues)",
            "m1_from_m3_aligned": "Five M1 words closest to fresh M3[w] @ R, excluding w",
            "missing": "Exact keys only; no spelling/case fallback or substituted query/answer",
        },
        "interpretation_notes": (
            "Both lists investigate which older M1 words occupied similar semantic roles. "
            "Discuss plausible and implausible analogues from the actual retrieved lists, considering polysemy and corpus context. "
            "No expected answers, extra analogy tests, or automatic success/failure labels are imposed."
        ),
        "vocabulary_sizes": {label: len(model.wv) for label, model in zip(("M1", "M2", "M3"), (m1, m2, m3))},
        "emerging_words": new_words, "queries": rows,
    }


def checked_paths(m1_path, m2_path, m3_path, output_path, *, alignment_path):
    inputs = tuple(Path(path).expanduser().resolve() for path in (m1_path, m2_path, m3_path))
    if len(set(inputs)) != 3:
        raise ValueError("M1, continued M2, and fresh M3 must be different files.")
    for path in inputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Model must be a nonempty existing file: {path}")
    alignment = Path(alignment_path).expanduser().resolve()
    if not alignment.is_file() or alignment.stat().st_size == 0:
        raise ValueError(f"Part 2.5 alignment report must be a nonempty existing file: {alignment}")
    output = Path(output_path).expanduser().resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Save results outside the repository.")
    if output in (*inputs, alignment):
        raise ValueError("Results must not overwrite an input model or alignment report.")
    marker = output.parent / f".{output.name}.in_progress"
    for path in (output, marker):
        if os.path.lexists(path):
            raise FileExistsError(f"Refusing existing result or active/incomplete marker: {path}")
    return inputs, alignment, output, marker


def run_analysis(m1_path, m2_path, m3_path, output_path, *, alignment_path, **analysis_options):
    inputs, alignment, output, marker = checked_paths(
        m1_path, m2_path, m3_path, output_path, alignment_path=alignment_path,
    )
    rotation, metadata = load_part25_alignment(alignment, inputs[0], inputs[2])
    output.parent.mkdir(parents=True, exist_ok=True)
    marker.mkdir()  # private, exclusive staging under the resolved external parent
    staged = marker / output.name
    try:
        models = [Word2Vec.load(str(path)) for path in inputs]
        report = analyze_temporal_models(*models, rotation, **analysis_options)
        report["alignment"] = metadata
        report["model_paths"] = dict(zip(("M1", "M2", "M3"), map(str, inputs)))
        report["versions"] = {"gensim": gensim.__version__, "numpy": np.__version__}
        with staged.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
        os.link(staged, output)  # exclusive publication; never replace an existing result
    except BaseException:
        shutil.rmtree(marker)  # remove only this attempt's exclusively created directory
        raise
    shutil.rmtree(marker)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--m2", type=Path, required=True, help="Continued P2 model")
    parser.add_argument("--m3", type=Path, required=True, help="Fresh P2 model")
    parser.add_argument("--alignment", type=Path, required=True, help="Part 2.5 semantic_drift.json containing saved R")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the repository")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and saved alignment without loading models")
    args = parser.parse_args(argv)
    if args.dry_run:
        inputs, alignment, output, _ = checked_paths(
            args.m1, args.m2, args.m3, args.output, alignment_path=args.alignment,
        )
        _, metadata = load_part25_alignment(alignment, inputs[0], inputs[2])
        print(json.dumps({
            "dry_run": True, "model_paths": list(map(str, inputs)), "output": str(output),
            "emerging_count": 20, "query_words": QUERY_WORDS, "neighbors_per_search": 5,
            "alignment": metadata,
            "searches": ["M1 neighbors of M2[w]", "M1 neighbors of M3[w] @ saved R"],
        }, indent=2))
        return 0
    report = run_analysis(args.m1, args.m2, args.m3, args.output, alignment_path=args.alignment)
    print(json.dumps({
        "output": str(args.output.resolve()), "emerging_words": report["emerging_words"],
        "missing_queries": [{"word": row["word"], "missing_query_models": row["missing_query_models"]}
                            for row in report["queries"] if row["missing_query_models"]],
        "queries": report["queries"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
