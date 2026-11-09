"""Part 2.6: compare M1/continued M2 neighbors with GN restricted to M1 words.

Offline loading only. No training, model saving, downloading, or job submission.
--dry-run validates paths without loading models or creating outputs.
"""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil

import gensim
from gensim.models import Word2Vec
import numpy as np

from src.google_news import gn_lookup_candidates, load_gn_vectors, lookup_gn_key
from src.semantic_drift import TARGET_GROUPS, TARGET_WORDS, length_normalize_rows


@dataclass(frozen=True)
class RestrictedGNIndex:
    labels: tuple[str, ...]
    gn_keys: tuple[str, ...]
    normalized_vectors: np.ndarray
    statistics: dict

    def neighbors(self, gn, word):
        """Rank only indexed M1 labels; exclude query label AND query GN key."""
        key = lookup_gn_key(gn, word)
        result = {"status": "missing", "matched_key": key, "neighbors": None}
        if key is None:
            return result
        try:
            query = length_normalize_rows([gn[key]], f"GN query {word}")[0]
        except ValueError as error:
            result["status"] = "undefined: " + str(error)
            return result
        eligible = np.fromiter(
            (label != word and candidate_key != key for label, candidate_key in zip(self.labels, self.gn_keys)),
            dtype=bool, count=len(self.labels),
        )
        indices = np.flatnonzero(eligible)
        if len(indices) < 10:
            result.update(status="insufficient_candidates", available_candidates=int(len(indices)))
            return result
        scores = np.clip(self.normalized_vectors @ query, -1.0, 1.0)
        # Sort only the restricted candidates; ties use original M1 labels.
        labels = np.asarray(self.labels, dtype=object)
        order = np.lexsort((labels[indices], -scores[indices]))[:10]
        result.update(status="ok", neighbors=[
            {"word": self.labels[index], "gn_key": self.gn_keys[index],
             "cosine_similarity": float(scores[index])}
            for index in indices[order]
        ])
        return result


def build_restricted_gn_index(m1, gn):
    """Resolve every M1 label with the shared rule; copy only its GN vectors.

    Do not build normalized vectors for the entire GN vocabulary. Undefined
    cosine candidates (zero/nonfinite vectors) are explicitly counted/listed.
    Different original labels remain distinct even if they share a GN key.
    """
    labels, keys, indices = [], [], []
    missing = 0
    for word in m1.index_to_key:
        key = lookup_gn_key(gn, word)
        if key is None:
            missing += 1
            continue
        labels.append(word)
        keys.append(key)
        indices.append(gn.key_to_index[key])
    undefined = []
    if indices:
        vectors = np.asarray(gn.vectors[np.asarray(indices)], dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(vectors).all(axis=1) & np.isfinite(norms) & (norms > 0)
        undefined = [
            {"word": word, "gn_key": key, "reason": "zero_or_nonfinite_vector"}
            for word, key, good in zip(labels, keys, valid) if not good
        ]
        normalized = vectors[valid] / norms[valid, None]
        labels = [word for word, good in zip(labels, valid) if good]
        keys = [key for key, good in zip(keys, valid) if good]
    else:
        normalized = np.empty((0, gn.vector_size), dtype=np.float64)
    return RestrictedGNIndex(tuple(labels), tuple(keys), normalized, {
        "m1_vocabulary_size": len(m1),
        "resolvable_m1_words": len(indices), "unresolved_m1_words": missing,
        "usable_candidates": len(labels), "undefined_candidates": undefined,
        "restriction": "Only original M1 words resolvable by lookup_gn_key",
        "self_exclusion": "Exclude original query label and all candidates mapped to its GN key",
        "ties": "Descending GN cosine, then original M1 label alphabetically",
    })


def local_neighbors(vectors, word):
    """Ten neighbors from a local model's own full vocabulary, unfiltered."""
    result = {"status": "missing", "neighbors": None}
    if word not in vectors:
        return result
    try:
        length_normalize_rows([vectors[word]], f"local query {word}")
        if len(vectors) < 11:
            result.update(status="insufficient_candidates", available_candidates=max(0, len(vectors) - 1))
            return result
        neighbors = vectors.most_similar(word, topn=10)
        labels = [label for label, _ in neighbors]
        if len(labels) != 10 or len(set(labels)) != 10 or word in labels:
            raise ValueError("Expected ten distinct non-query neighbors.")
        result.update(status="ok", neighbors=[
            {"word": label, "cosine_similarity": float(score)} for label, score in neighbors
        ])
    except ValueError as error:
        result["status"] = "undefined: " + str(error)
    return result


def compare_neighborhoods(gn_result, local_result, local_name):
    if gn_result["status"] != "ok" or local_result["status"] != "ok":
        return {
            "score": None, "shared_words": None,
            "status": f"unavailable: GN={gn_result['status']}; {local_name}={local_result['status']}",
        }
    gn_words = {item["word"] for item in gn_result["neighbors"]}
    local_words = {item["word"] for item in local_result["neighbors"]}
    if len(gn_words) != 10 or len(local_words) != 10:
        raise ValueError("NN@10 requires ten distinct original word labels in each neighborhood.")
    shared = sorted(gn_words & local_words)
    return {"score": len(shared) / 10, "shared_words": shared, "status": "ok"}


def analyze_gn_neighbors(m1, m2, gn, *, targets=TARGET_WORDS):
    """Two NN@10 measures per target; no cross-model vector alignment/cosine."""
    targets = tuple(targets)
    if len(set(targets)) != len(targets):
        raise ValueError("Target words must be unique.")
    for word in targets:
        gn_lookup_candidates(word)
    index = build_restricted_gn_index(m1.wv, gn)
    rows = []
    for word in targets:
        gn_result = index.neighbors(gn, word)
        first, second = local_neighbors(m1.wv, word), local_neighbors(m2.wv, word)
        overlap1 = compare_neighborhoods(gn_result, first, "M1")
        overlap2 = compare_neighborhoods(gn_result, second, "M2")
        closer = "unavailable"
        if overlap1["score"] is not None and overlap2["score"] is not None:
            closer = "M1" if overlap1["score"] > overlap2["score"] else (
                "M2" if overlap2["score"] > overlap1["score"] else "tie"
            )
        rows.append({
            "word": word, "group": TARGET_GROUPS.get(word, "custom"),
            "in_m1": word in m1.wv, "in_m2": word in m2.wv,
            "gn_key": gn_result["matched_key"],
            "missing_models": [label for label, present in (
                ("M1", word in m1.wv), ("M2", word in m2.wv),
                ("GN", gn_result["matched_key"] is not None),
            ) if not present],
            "m1": first, "m2": second, "gn": gn_result,
            "nn_gn_m1": overlap1["score"], "nn_gn_m1_status": overlap1["status"],
            "nn_gn_m2": overlap2["score"], "nn_gn_m2_status": overlap2["status"],
            "shared_gn_m1": overlap1["shared_words"], "shared_gn_m2": overlap2["shared_words"],
            "closer_to": closer,
        })
    return {
        "target_count": len(rows), "neighbor_count": 10,
        "vocabulary_sizes": {"M1": len(m1.wv), "M2": len(m2.wv), "GN": len(gn)},
        "candidate_pool": index.statistics,
        "definitions": {
            "gn_neighbors": "GN cosine over resolvable M1 vocabulary only, excluding query label/key",
            "local_neighbors": "Ten unfiltered neighbors in each local model's full vocabulary",
            "nn_gn_m1": "|GN restricted top10 labels intersect M1 top10 labels| / 10",
            "nn_gn_m2": "|GN restricted top10 labels intersect continued M2 top10 labels| / 10",
            "labels": "Original M1 labels for GN neighbors; record actual matched GN keys separately",
            "closer_to": "Larger NN@10 overlap wins; equal scores are a tie; missing scores are unavailable",
        },
        "report_notes": (
            "Inspect both overlap scores and neighbor lists to discuss resemblance and disagreements. "
            "Case fallback can merge meanings; use verified GN training provenance when discussing its period. "
            "No disagreement threshold, aggregate correlation, or cross-space alignment is imposed."
        ),
        "words": rows,
    }


def checked_paths(m1_path, m2_path, gn_path, output_path):
    inputs = tuple(Path(path).expanduser().resolve() for path in (m1_path, m2_path, gn_path))
    if len(set(inputs)) != 3:
        raise ValueError("M1, continued M2, and GN must be different files.")
    for path in inputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Model must be a nonempty existing file: {path}")
    output = Path(output_path).expanduser().resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Save comparison results outside the repository.")
    if output in inputs:
        raise ValueError("Results must not overwrite an input model.")
    marker = output.parent / f".{output.name}.in_progress"
    for path in (output, marker):
        if os.path.lexists(path):
            raise FileExistsError(f"Refusing existing result or active/incomplete marker: {path}")
    return inputs, output, marker


def run_comparison(m1_path, m2_path, gn_path, output_path, *, gn_format="word2vec", targets=TARGET_WORDS):
    if gn_format not in {"word2vec", "keyedvectors"}:
        raise ValueError("GN format must be word2vec or keyedvectors.")
    inputs, output, marker = checked_paths(m1_path, m2_path, gn_path, output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    marker.mkdir()  # exclusive private staging under the resolved output parent
    staged = marker / output.name
    try:
        m1, m2 = (Word2Vec.load(str(path)) for path in inputs[:2])
        gn = load_gn_vectors(inputs[2], model_format=gn_format)
        report = analyze_gn_neighbors(m1, m2, gn, targets=targets)
        report["model_paths"] = dict(zip(("M1", "M2", "GN"), map(str, inputs)))
        report["gn_format"] = gn_format
        report["versions"] = {"gensim": gensim.__version__, "numpy": np.__version__}
        with staged.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
        os.link(staged, output)  # exclusive publication; preserves concurrent results
    except BaseException:
        shutil.rmtree(marker)  # only this attempt's exclusively created private directory
        raise
    shutil.rmtree(marker)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--m2", type=Path, required=True, help="Continued P2 model")
    parser.add_argument("--gn", type=Path, required=True)
    parser.add_argument("--gn-format", choices=["word2vec", "keyedvectors"], default="word2vec")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the repository")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths without loading models or analysis")
    args = parser.parse_args(argv)
    if args.dry_run:
        inputs, output, _ = checked_paths(args.m1, args.m2, args.gn, args.output)
        print(json.dumps({
            "dry_run": True, "model_paths": list(map(str, inputs)), "output_path": str(output),
            "gn_format": args.gn_format, "target_count": len(TARGET_WORDS), "neighbor_count": 10,
        }, indent=2))
        return 0
    report = run_comparison(args.m1, args.m2, args.gn, args.output, gn_format=args.gn_format)
    print(json.dumps({
        "target_count": report["target_count"], "output": str(args.output.resolve()),
        "missing_words": [{"word": row["word"], "missing_models": row["missing_models"]}
                          for row in report["words"] if row["missing_models"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
