"""Part 2.5 analysis of saved M1, continued M2, and fresh M3; never train/save models.

Use --dry-run to validate paths and print the plan without loading models.
See notes/README.md for CLI entry points and saved measurement definitions.
"""

import argparse
import csv
from itertools import combinations
import json
import os
from pathlib import Path
import shutil

import gensim
from gensim.models import Word2Vec
import numpy as np
import scipy
from scipy.stats import spearmanr

from src.train_m1 import nearest_neighbors


DRIFT_CANDIDATES = tuple("""
zoom delta booster variant lockdown distancing mask quarantine positive spike remote
essential trump pence meta ai chair stream streaming tweet platform podcast viral
vaccine tesla uber apple cloud corona woke
""".split())
CONTROL_WORDS = tuple("river piano mountain wine coffee bread doctor bridge baseball".split())
ADDITIONAL_WORDS = ("privacy", "internet", "phone", "election", "climate")
TARGET_WORDS = DRIFT_CANDIDATES + CONTROL_WORDS + ADDITIONAL_WORDS
TARGET_GROUPS = {
    **dict.fromkeys(DRIFT_CANDIDATES, "drift_candidate"),
    **dict.fromkeys(CONTROL_WORDS, "control"),
    **dict.fromkeys(ADDITIONAL_WORDS, "additional"),
}
ANCHOR_COUNT = 20_000
RAW_EXAMPLE_WORDS = ("privacy", "internet", "phone")
MEASURES = ("cos_cont", "cos_align", "nn_at_10")
CSV_FIELDS = (
    "word", "group", "in_m1", "in_m2", "in_m3", "missing_models", "drift_rank",
    "cos_cont", "cos_cont_status", "cos_align", "cos_align_status",
    "nn_at_10", "nn_at_10_status", "raw_cos_m1_m3", "raw_cos_status",
)
OUTPUT_NAMES = ("semantic_drift.csv", "semantic_drift.json")
RANKING_CRITERION = (
    "Ascending cos_align: lower similarity means greater drift; alphabetical ties. "
    "Stable words use descending cos_align with alphabetical ties. "
    "Only defined cos_align values are eligible; the two selections are disjoint."
)


def length_normalize_rows(matrix, label="vectors"):
    """Normalize independent float64 copies, never stored model vectors."""
    matrix = np.array(matrix, dtype=np.float64, copy=True)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError(f"{label}: expected a nonempty finite matrix.")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError(f"{label}: zero or nonfinite vector norm.")
    return matrix / norms[:, None]


def fit_procrustes(source, reference):
    """Minimize ||unit(source) @ R - unit(reference)||_F for ROW vectors.

    SVD of source.T @ reference gives R = U @ Vt. No mean centering,
    learned scale, determinant correction, or transpose of the resulting R.
    """
    source = length_normalize_rows(source, "Procrustes source")
    reference = length_normalize_rows(reference, "Procrustes reference")
    if source.shape != reference.shape:
        raise ValueError("Procrustes source and reference must have the same shape.")
    u, singular_values, vt = np.linalg.svd(source.T @ reference)
    rotation = u @ vt
    return rotation, {
        "singular_values": singular_values.tolist(),
        "normalized_anchor_rmse": float(np.sqrt(np.mean((source @ rotation - reference) ** 2))),
        "orthogonality_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(rotation.shape[0]))),
        "determinant": float(np.linalg.det(rotation)),
    }


def select_alignment_anchors(m1, m3, count=ANCHOR_COUNT):
    """Take highest M1 counts AFTER intersection with M3; ties by M1 index."""
    if count < 1:
        raise ValueError("Anchor count must be positive.")
    shared = [word for word in m1.index_to_key if word in m3]
    if len(shared) < count:
        raise ValueError(f"Need {count} shared alignment anchors; only {len(shared)} exist.")
    try:
        shared.sort(key=lambda word: (-int(m1.get_vecattr(word, "count")), m1.key_to_index[word]))
    except KeyError as error:
        raise ValueError("M1 word frequencies ('count') are required for anchor selection.") from error
    return tuple(shared[:count])


def fit_alignment(m1, m3, anchor_count=ANCHOR_COUNT):
    if m1.vector_size != m3.vector_size:
        raise ValueError("M1 and M3 must have equal vector dimensions.")
    anchors = select_alignment_anchors(m1, m3, anchor_count)
    rotation, diagnostics = fit_procrustes(
        np.stack([m3[word] for word in anchors]),
        np.stack([m1[word] for word in anchors]),
    )
    return rotation, {
        "source": "M3 (fresh)", "reference": "M1",
        "direction": "row_vector_M3 @ R -> M1",
        "fit": "SVD(unit(M3 anchors).T @ unit(M1 anchors)); R = U @ Vt; no centering",
        "anchor_count": len(anchors),
        "anchor_selection": "Top M1 counts among words present in M3; ties by M1 vocabulary index",
        "anchors": [{"word": word, "m1_count": int(m1.get_vecattr(word, "count"))} for word in anchors],
        "rotation": rotation.tolist(),
        **diagnostics,
    }


def cosine(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError("Cosine requires equal one-dimensional vector shapes.")
    unit = length_normalize_rows(np.stack([left, right]), "cosine")
    return float(np.clip(unit[0] @ unit[1], -1.0, 1.0))


def neighbor_overlap(m1, m3, word):
    """Unaligned top-10 cosine neighbors from each FULL model vocabulary."""
    neighborhoods = []
    for label, vectors in [("M1", m1), ("M3", m3)]:
        length_normalize_rows([vectors[word]], f"{label}[{word}]")
        if len(vectors) < 11:
            raise ValueError(f"{label} needs at least 11 words for ten neighbors.")
        neighbors = [term for term, _ in vectors.most_similar(word, topn=10)]
        if len(neighbors) != 10 or len(set(neighbors)) != 10 or word in neighbors:
            raise ValueError(f"{label} did not return ten distinct non-query neighbors.")
        neighborhoods.append(neighbors)
    shared = sorted(set(neighborhoods[0]) & set(neighborhoods[1]))
    return {
        "score": len(shared) / 10,
        "m1_neighbors": neighborhoods[0], "m3_neighbors": neighborhoods[1],
        "shared_neighbors": shared,
    }


def measure_word(word, m1, m2, m3, rotation):
    vectors = {"M1": m1, "M2": m2, "M3": m3}
    row = {"word": word, "group": TARGET_GROUPS.get(word, "custom"), "drift_rank": None}
    row.update({f"in_{label.lower()}": word in model for label, model in vectors.items()})
    row["missing_models"] = [label for label, model in vectors.items() if word not in model]

    def compute(field, required, calculation):
        missing = [label for label in required if word not in vectors[label]]
        row[field] = None
        if missing:
            return "missing: " + ", ".join(missing)
        try:
            row[field] = calculation()
        except ValueError as error:
            return "undefined: " + str(error)
        return "ok"

    row["cos_cont_status"] = compute("cos_cont", ("M1", "M2"), lambda: cosine(m1[word], m2[word]))
    row["cos_align_status"] = compute(
        "cos_align", ("M1", "M3"), lambda: cosine(m1[word], np.asarray(m3[word], dtype=np.float64) @ rotation),
    )
    row["raw_cos_status"] = compute("raw_cos_m1_m3", ("M1", "M3"), lambda: cosine(m1[word], m3[word]))
    overlap = {}

    def compute_overlap():
        overlap.update(neighbor_overlap(m1, m3, word))
        return overlap["score"]

    row["nn_at_10_status"] = compute("nn_at_10", ("M1", "M3"), compute_overlap)
    row["neighbor_overlap_details"] = overlap or None
    return row


def spearman_correlations(rows):
    """Pairwise complete observations across candidates, controls, and extras."""
    results = []
    for left, right in combinations(MEASURES, 2):
        eligible = [row for row in rows if row[left] is not None and row[right] is not None]
        record = {
            "measure_a": left, "measure_b": right, "n": len(eligible),
            "words": [row["word"] for row in eligible], "rho": None, "p_value": None,
        }
        x, y = [row[left] for row in eligible], [row[right] for row in eligible]
        if len(eligible) < 2:
            record["status"] = "insufficient_pairs"
        elif len(set(x)) == 1 or len(set(y)) == 1:
            record["status"] = "constant_input"
        else:
            result = spearmanr(x, y)
            record.update(rho=float(result.statistic), status="ok")
            if np.isfinite(result.pvalue):
                record["p_value"] = float(result.pvalue)
        results.append(record)
    return results


def rank_words(rows, count=5):
    if count < 1:
        raise ValueError("Selection count must be positive.")
    ranked = sorted((row for row in rows if row["cos_align"] is not None), key=lambda row: (row["cos_align"], row["word"]))
    for rank, row in enumerate(ranked, 1):
        row["drift_rank"] = rank
    drifted = [row["word"] for row in ranked[:count]]
    stable = [row["word"] for row in sorted(ranked, key=lambda row: (-row["cos_align"], row["word"])) if row["word"] not in drifted][:count]
    return {
        "criterion": RANKING_CRITERION, "eligible_count": len(ranked),
        "most_drifted": drifted, "most_stable": stable,
        "requested_per_group": count,
        "status": "ok" if len(ranked) >= 2 * count else "insufficient_rankable_words",
    }


def _selected_neighbors(model, word):
    if word not in model.wv:
        return {"status": "missing", "neighbors": None}
    try:
        length_normalize_rows([model.wv[word]], f"neighbor query {word}")
        return {"status": "ok", **nearest_neighbors(model, [word])[0]}
    except ValueError as error:
        return {"status": "undefined: " + str(error), "neighbors": None}


def analyze_models(m1, m2, m3, *, targets=TARGET_WORDS, anchor_count=ANCHOR_COUNT, selection_count=5):
    """Analyze in-memory models without calling train, build_vocab, or save."""
    if len(set(targets)) != len(targets):
        raise ValueError("Target words must be unique.")
    if len({model.wv.vector_size for model in (m1, m2, m3)}) != 1:
        raise ValueError("All three models must have equal vector dimensions.")
    rotation, alignment = fit_alignment(m1.wv, m3.wv, anchor_count)
    rows = [measure_word(word, m1.wv, m2.wv, m3.wv, rotation) for word in targets]
    correlations = spearman_correlations(rows)
    rankings = rank_words(rows, selection_count)
    selected = [
        {"word": word, "selection": selection, "m1": _selected_neighbors(m1, word), "m2": _selected_neighbors(m2, word)}
        for selection in ("most_drifted", "most_stable") for word in rankings[selection]
    ]
    indexed = {row["word"]: row for row in rows}
    raw_examples = []
    for word in RAW_EXAMPLE_WORDS:
        row = indexed.get(word)
        raw_examples.append({
            "word": word,
            "raw_cos_m1_m3": row["raw_cos_m1_m3"] if row else None,
            "cos_align": row["cos_align"] if row else None,
            "raw_status": row["raw_cos_status"] if row else "not_in_target_list",
            "aligned_status": row["cos_align_status"] if row else "not_in_target_list",
        })
    rows.sort(key=lambda row: (row["drift_rank"] is None, row["drift_rank"] or 0, row["word"]))
    return {
        "definitions": {
            "cos_cont": "cosine(M1[w], M2_continued[w]); no alignment",
            "cos_align": "cosine(M1[w], M3_fresh[w] @ R)",
            "nn_at_10": "|top10(M1,w) intersect top10(M3,w)| / 10; full vocabularies, no filtering or alignment",
            "missing": "Exact membership only; unavailable measures are null (blank in CSV), with status fields",
            "correlations": "Spearman rho on pairwise defined measures; report n, words and undefined cases",
            "raw_examples": "Fixed words privacy, internet, phone; no substitution if missing",
        },
        "interpretation": (
            "Independently trained embeddings have arbitrary coordinate orientations. "
            "Rotations/reflections can preserve within-model similarities while changing "
            "raw cross-model cosine, so unaligned M1-M3 cosine does not measure semantic drift. "
            "Procrustes establishes shared axes using common normalized anchors. "
            "Continued M2 starts from M1, making direct cosine meaningful. "
            "The same coordinate mismatch applies to raw M1-GN cosine; GN numerical examples are deferred to Leylak."
        ),
        "model_vocabulary_sizes": {label: len(model.wv) for label, model in zip(("M1", "M2", "M3"), (m1, m2, m3))},
        "target_count": len(targets), "alignment": alignment, "words": rows,
        "missing_words": [{"word": row["word"], "missing_models": row["missing_models"]} for row in rows if row["missing_models"]],
        "rankings": rankings, "spearman_correlations": correlations,
        "selected_neighbors": selected, "raw_vs_aligned_examples": raw_examples,
    }


def checked_paths(m1_path, m2_path, m3_path, output_dir):
    inputs = tuple(Path(path).expanduser().resolve() for path in (m1_path, m2_path, m3_path))
    if len(set(inputs)) != 3:
        raise ValueError("M1, continued M2, and fresh M3 must be three different model files.")
    for path in inputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Model must be a nonempty existing file: {path}")
    output = Path(output_dir).expanduser().resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Save results outside the repository.")
    if output.exists() and not output.is_dir():
        raise ValueError("Output must be a directory.")
    marker = output / ".semantic_drift.in_progress"
    for path in [marker, *(output / name for name in OUTPUT_NAMES)]:
        if os.path.lexists(path):
            raise FileExistsError(f"Refusing existing result or active/incomplete marker: {path}")
    return inputs, output, marker


def run_analysis(m1_path, m2_path, m3_path, output_dir, **analysis_options):
    inputs, output, marker = checked_paths(m1_path, m2_path, m3_path, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    marker.mkdir()  # exclusive staging under the resolved external output directory
    staged = {output / name: marker / name for name in OUTPUT_NAMES}
    try:
        models = [Word2Vec.load(str(path)) for path in inputs]
        report = analyze_models(*models, **analysis_options)
        report["model_paths"] = dict(zip(("M1", "M2", "M3"), map(str, inputs)))
        report["versions"] = {"gensim": gensim.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
        with staged[output / "semantic_drift.json"].open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        with staged[output / "semantic_drift.csv"].open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in report["words"]:
                record = {field: row[field] for field in CSV_FIELDS}
                record["missing_models"] = ";".join(row["missing_models"])
                writer.writerow(record)
        for final, temporary in staged.items():
            os.link(temporary, final)  # refuses overwrite, including concurrent writers
    except BaseException:
        for final, temporary in staged.items():
            if temporary.exists() and final.exists() and final.samefile(temporary):
                final.unlink()
        shutil.rmtree(marker)  # only this attempt's exclusively created staging directory
        raise
    shutil.rmtree(marker)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--m2", type=Path, required=True, help="Continued model")
    parser.add_argument("--m3", type=Path, required=True, help="Fresh P2 model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Check paths and plan without loading models or analysis")
    args = parser.parse_args(argv)
    if args.dry_run:
        inputs, output, _ = checked_paths(args.m1, args.m2, args.m3, args.output_dir)
        print(json.dumps({
            "dry_run": True, "model_paths": list(map(str, inputs)),
            "output_dir": str(output), "output_files": OUTPUT_NAMES,
            "anchor_count": ANCHOR_COUNT, "target_count": len(TARGET_WORDS),
            "target_words": TARGET_WORDS, "ranking_criterion": RANKING_CRITERION,
        }, indent=2))
        return 0
    report = run_analysis(args.m1, args.m2, args.m3, args.output_dir)
    print(json.dumps({
        "target_count": report["target_count"], "missing_words": report["missing_words"],
        "rankings": report["rankings"], "spearman_correlations": report["spearman_correlations"],
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
