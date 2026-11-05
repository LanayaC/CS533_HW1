"""Reconstruct and validate known-item qrels only, without writing datasets.

Reproduce from the repository root with configurable paths::

    python -m src.known_item_qrels --dataset ../data/articles_mapped.h5 \
        --official-dir ../data/nyt-tc-2026

The command reuses cleaning, collection reconstruction, global eligibility,
and MD5 query selection. Only afterward does it read qrels_ki_P1.txt and
qrels_ki_P2.txt for validation. No topic queries or topic qrels are generated.
"""

import argparse
from pathlib import Path
import re
from time import perf_counter

import pandas as pd

from .document_collections import (
    EXPECTED_CLEANED_COUNT,
    EXPECTED_COLLECTION_COUNTS,
    build_collections,
    clean_articles,
)
from .known_item_eligibility import check_known_item_eligibility
from .known_item_queries import select_known_item_queries


QRELS_COLUMNS = ["qid", "iteration", "docno", "relevance"]
_OFFICIAL_FILES = {"C1": "qrels_ki_P1.txt", "C2": "qrels_ki_P2.txt"}


def _check_qrels(frame: pd.DataFrame, label: str) -> None:
    missing = set(QRELS_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if frame[QRELS_COLUMNS].isna().any().any():
        raise ValueError(f"{label}: missing judgment values.")
    if frame["qid"].duplicated().any():
        raise ValueError(f"{label}: each known-item query must have exactly one judgment.")
    for column in ["qid", "docno"]:
        valid = frame[column].map(
            lambda value: isinstance(value, str) and bool(value) and not re.search(r"\s", value)
        )
        if not valid.all():
            raise ValueError(f"{label}: {column} must contain nonempty, whitespace-free strings.")
    for column in ["iteration", "relevance"]:
        if not pd.api.types.is_integer_dtype(frame[column]):
            raise ValueError(f"{label}: {column} must contain integers.")


def build_known_item_qrels(
    queries: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Create one judgment per reconstructed known-item query in C1/C2.

    Use only qid and target docno from select_known_item_queries() results.
    Preserve query row order; set iteration=0 and relevance=1 for every row.
    Return C1/C2 frames with qid, iteration, docno, relevance columns, ready
    for headerless, space-separated TREC serialization. Duplicate query IDs,
    missing IDs/targets, or identifiers containing whitespace raise ValueError.
    Do not mutate the input, read official files, or create any files.
    """
    judgments = {}
    for name in _OFFICIAL_FILES:
        frame = queries[name].loc[:, ["qid", "docno"]].copy().reset_index(drop=True)
        frame.insert(1, "iteration", 0)
        frame["relevance"] = 1
        _check_qrels(frame, f"Reconstructed {name}")
        judgments[name] = frame.loc[:, QRELS_COLUMNS]
    return judgments


def format_known_item_qrels(judgments: pd.DataFrame) -> str:
    """Return TREC text: qid 0 docno 1, one row per line, with no header.

    Return an empty string for an empty frame. This function does not write
    files. Use the frames returned by build_known_item_qrels(); reject other
    iteration/relevance values so the known-item format is unambiguous.
    """
    _check_qrels(judgments, "Known-item serialization")
    if not judgments["iteration"].eq(0).all() or not judgments["relevance"].eq(1).all():
        raise ValueError("Known-item output requires iteration=0 and relevance=1.")
    return judgments.loc[:, QRELS_COLUMNS].to_csv(
        sep=" ", header=False, index=False, lineterminator="\n"
    )


def compare_known_item_qrels(
    judgments: dict[str, pd.DataFrame], official_dir: str | Path
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """Validate against official known-item qrels, matching judgments by qid.

    Missing judgments have official-only IDs; extras have reconstructed-only
    IDs. A shared ID with a different iteration, docno, or relevance is one
    mismatched record, even when multiple fields differ. Reject duplicate qids
    rather than silently merging them. Ignore physical row order and whitespace
    between TREC fields. Return a summary and per-period detailed differences.
    No input frames or official files are modified; no topic qrels are read.
    """
    records, details = [], {}
    for name, filename in _OFFICIAL_FILES.items():
        reconstructed = judgments[name]
        _check_qrels(reconstructed, f"Reconstructed {name}")
        official = pd.read_csv(
            Path(official_dir) / filename,
            sep=r"\s+",
            header=None,
            names=QRELS_COLUMNS,
            dtype={"qid": str, "iteration": "int64", "docno": str, "relevance": "int64"},
            keep_default_na=False,
        )
        _check_qrels(official, f"Official {name}")
        left, right = reconstructed.set_index("qid"), official.set_index("qid")
        missing = official.loc[
            ~official["qid"].isin(left.index), QRELS_COLUMNS
        ].reset_index(drop=True)
        extra = reconstructed.loc[
            ~reconstructed["qid"].isin(right.index), QRELS_COLUMNS
        ].reset_index(drop=True)
        common = left.index.intersection(right.index, sort=False)
        fields = ["iteration", "docno", "relevance"]
        joined = left.loc[common, fields].add_suffix("_reconstructed").join(
            right.loc[common, fields].add_suffix("_official")
        )
        masks = pd.DataFrame(
            {
                field: joined[f"{field}_reconstructed"].ne(joined[f"{field}_official"])
                for field in fields
            },
            index=joined.index,
        )
        mismatched = joined.loc[masks.any(axis=1)].copy()
        mismatched["mismatched_fields"] = [
            ", ".join(field for field in fields if row[field])
            for _, row in masks.loc[mismatched.index].iterrows()
        ]
        details[name] = {
            "missing": missing,
            "extra": extra,
            "mismatched": mismatched.reset_index(),
        }
        records.append(
            {
                "collection": name,
                "reconstructed_count": len(reconstructed),
                "official_count": len(official),
                "missing": len(missing),
                "extra": len(extra),
                "mismatched": len(mismatched),
                "total_differences": len(missing) + len(extra) + len(mismatched),
            }
        )
    return pd.DataFrame(records).set_index("collection"), details


def main(argv: list[str] | None = None) -> int:
    """Rebuild query targets independently and validate known-item judgments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Original HDF5 path")
    parser.add_argument("--official-dir", type=Path, required=True, help="Official qrels directory")
    parser.add_argument("--hdf-key", default="data", help="HDF5 key (default: data)")
    parser.add_argument("--examples", type=int, default=5, help="Examples per difference type")
    args = parser.parse_args(argv)
    if args.examples < 0:
        parser.error("--examples must be nonnegative")

    start = perf_counter()
    print(f"Loading original dataset: {args.dataset.resolve()}", flush=True)
    original = pd.read_hdf(args.dataset, key=args.hdf_key)
    cleaned, report = clean_articles(original)
    del original
    print("\nPart 1.2 cleaning (in order):", flush=True)
    print(report.to_string(), flush=True)
    cleaned_count = len(cleaned)
    print(f"Final cleaned article count: {cleaned_count:,}", flush=True)
    print("Reconstructing collections and global known-item eligibility...", flush=True)
    collections = build_collections(cleaned)
    eligibility = check_known_item_eligibility(cleaned)
    del cleaned
    queries = select_known_item_queries(collections, eligibility)
    judgments = build_known_item_qrels(queries)
    print("Reconstructed one known-item judgment per selected query.", flush=True)

    print(f"\nComparing with official known-item qrels: {args.official_dir.resolve()}", flush=True)
    summary, details = compare_known_item_qrels(judgments, args.official_dir)
    print(summary.to_string(), flush=True)
    for name, kinds in details.items():
        for kind, frame in kinds.items():
            if not frame.empty and args.examples:
                print(f"\n{name} {kind} examples:", flush=True)
                print(frame.head(args.examples).to_string(index=False), flush=True)

    passed = (
        cleaned_count == EXPECTED_CLEANED_COUNT
        and all(len(collections[name]) == count for name, count in EXPECTED_COLLECTION_COUNTS.items())
        and summary["reconstructed_count"].eq(1000).all()
        and summary["official_count"].eq(1000).all()
        and summary["total_differences"].eq(0).all()
    )
    print(f"\nKnown-item qrels validation {'PASSED' if passed else 'FAILED'}", flush=True)
    print(f"Elapsed time: {perf_counter() - start:.1f} seconds", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
