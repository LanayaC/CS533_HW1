"""Appendix A.5 topic qrels reconstructed from independent C1/C2 collections.

Reproduce validation from the repository root with configurable paths::

    python -m src.topic_qrels --dataset ../data/articles_mapped.h5 \
        --official-dir ../data/nyt-tc-2026

Reuse the existing cleaning and collection builders. Only after constructing
judgments, read qrels_topics_P1.txt and qrels_topics_P2.txt for validation.
No generated datasets are written and no notebook is executed or modified.
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


# Appendix Table 2, explicitly transcribed; never loaded from official files.
TOPIC_IDS = {
    "africa": "T01",
    "art": "T02",
    "asia": "T03",
    "baseball": "T04",
    "basketball": "T05",
    "books": "T06",
    "dance": "T07",
    "economy": "T08",
    "europe": "T09",
    "fashion": "T10",
    "food": "T11",
    "football": "T12",
    "golf": "T13",
    "health": "T14",
    "hockey": "T15",
    "latin america": "T16",
    "media": "T17",
    "middle east": "T18",
    "movies": "T19",
    "music": "T20",
    "olympics": "T21",
    "personal finance": "T22",
    "politics": "T23",
    "real estate": "T24",
    "science": "T25",
    "soccer": "T26",
    "technology": "T27",
    "television": "T28",
    "tennis": "T29",
    "theater": "T30",
    "travel": "T31",
    "weddings": "T32",
}
QRELS_COLUMNS = ["tid", "iteration", "docno", "relevance"]
_OFFICIAL_FILES = {"C1": "qrels_topics_P1.txt", "C2": "qrels_topics_P2.txt"}


def _check_qrels(frame: pd.DataFrame, label: str) -> None:
    missing = set(QRELS_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if frame[QRELS_COLUMNS].isna().any().any():
        raise ValueError(f"{label}: missing judgment values.")
    if frame["docno"].duplicated().any():
        raise ValueError(f"{label}: each article must have exactly one topic judgment.")
    for column in ["tid", "docno"]:
        valid = frame[column].map(
            lambda value: isinstance(value, str) and bool(value) and not re.search(r"\s", value)
        )
        if not valid.all():
            raise ValueError(f"{label}: {column} must contain nonempty, whitespace-free strings.")
    for column in ["iteration", "relevance"]:
        if not pd.api.types.is_integer_dtype(frame[column]):
            raise ValueError(f"{label}: {column} must contain integers.")


def build_topic_qrels(collections: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Create one tid 0 docno 1 judgment per C1/C2 article, following A.5.

    Read only docno and topic from the independently reconstructed collections.
    Map exact topic labels to T01-T32 using Table 2's TOPIC_IDS, set iteration
    to 0 and relevance to 1, and preserve article row order. Both books URL
    paths already have the collection label books and map to T06.

    Return C1/C2 frames with tid, iteration, docno, relevance columns. Many
    documents can share a tid; each docno must occur once per collection.
    Unknown/missing labels or invalid/duplicate document IDs raise ValueError,
    so articles cannot silently disappear. Empty collections remain empty.
    Do not modify inputs, read official files, or generate topic-query files.
    """
    judgments = {}
    for name in _OFFICIAL_FILES:
        source = collections[name].loc[:, ["docno", "topic"]].copy().reset_index(drop=True)
        unknown = source.loc[~source["topic"].isin(TOPIC_IDS), "topic"].drop_duplicates()
        if not unknown.empty:
            raise ValueError(f"{name}: unknown or missing topic labels: {unknown.tolist()}")
        frame = pd.DataFrame(
            {
                "tid": source["topic"].map(TOPIC_IDS),
                "iteration": 0,
                "docno": source["docno"],
                "relevance": 1,
            }
        )
        _check_qrels(frame, f"Reconstructed {name}")
        judgments[name] = frame.loc[:, QRELS_COLUMNS]
    return judgments


def format_topic_qrels(judgments: pd.DataFrame) -> str:
    """Return headerless, space-separated TREC text, without writing files.

    Each line is tid 0 docno 1 with an LF newline; an empty frame produces
    an empty string. Reject non-Table-2 IDs and nonbinary topic judgments.
    """
    _check_qrels(judgments, "Topic serialization")
    if not judgments["tid"].isin(TOPIC_IDS.values()).all():
        raise ValueError("Topic IDs must be T01-T32 from Table 2.")
    if not judgments["iteration"].eq(0).all() or not judgments["relevance"].eq(1).all():
        raise ValueError("Topic output requires iteration=0 and relevance=1.")
    return judgments.loc[:, QRELS_COLUMNS].to_csv(
        sep=" ", header=False, index=False, lineterminator="\n"
    )


def compare_topic_qrels(
    judgments: dict[str, pd.DataFrame], official_dir: str | Path
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """Compare reconstructed judgments with official topic qrels, read only.

    A.5 assigns each article exactly one topic, so match on docno. Missing
    judgments have official-only docnos; extras have reconstructed-only docnos.
    A shared docno with a different tid, iteration, or relevance is mismatched,
    counted once even if several fields disagree. This identifies a wrong
    topic assignment as a mismatch rather than a missing-plus-extra pair.
    Duplicate docnos raise ValueError; repeated topic IDs are expected.

    Return a C1/C2 summary and per-period missing/extra/mismatched detail
    frames. Ignore physical row order and inter-field whitespace. Do not
    mutate inputs, write files, or read official topic definitions to build
    any judgments. Compare all records, not merely aggregate counts.
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
            dtype={"tid": str, "iteration": "int64", "docno": str, "relevance": "int64"},
            keep_default_na=False,
        )
        _check_qrels(official, f"Official {name}")
        left, right = reconstructed.set_index("docno"), official.set_index("docno")
        missing = official.loc[
            ~official["docno"].isin(left.index), QRELS_COLUMNS
        ].reset_index(drop=True)
        extra = reconstructed.loc[
            ~reconstructed["docno"].isin(right.index), QRELS_COLUMNS
        ].reset_index(drop=True)
        common = left.index.intersection(right.index, sort=False)
        fields = ["tid", "iteration", "relevance"]
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
    """Independently rebuild and validate topic judgments; write no files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Original HDF5 path")
    parser.add_argument("--official-dir", type=Path, required=True, help="Official topic qrels directory")
    parser.add_argument("--hdf-key", default="data", help="HDF5 key (default: data)")
    parser.add_argument("--examples", type=int, default=5, help="Examples per difference type")
    args = parser.parse_args(argv)
    if args.examples < 0:
        parser.error("--examples must be nonnegative")

    start = perf_counter()
    print(f"Loading original dataset: {args.dataset.resolve()}", flush=True)
    original = pd.read_hdf(args.dataset, key=args.hdf_key)
    cleaned, cleaning_report = clean_articles(original)
    del original
    print("\nPart 1.2 cleaning (in order):", flush=True)
    print(cleaning_report.to_string(), flush=True)
    cleaned_count = len(cleaned)
    print(f"Final cleaned article count: {cleaned_count:,}", flush=True)
    print("Reconstructing C1/C2 independently...", flush=True)
    collections = build_collections(cleaned)
    del cleaned
    judgments = build_topic_qrels(collections)
    for name, frame in judgments.items():
        print(
            f"{name}: {len(frame):,} judgments covering {frame['tid'].nunique()} topic IDs",
            flush=True,
        )
    print(f"\nComparing with official topic qrels: {args.official_dir.resolve()}", flush=True)
    summary, details = compare_topic_qrels(judgments, args.official_dir)
    print(summary.to_string(), flush=True)
    for name, kinds in details.items():
        for kind, frame in kinds.items():
            if not frame.empty and args.examples:
                print(f"\n{name} {kind} examples:", flush=True)
                print(frame.head(args.examples).to_string(index=False), flush=True)

    passed = (
        cleaned_count == EXPECTED_CLEANED_COUNT
        and all(
            len(collections[name]) == count and len(judgments[name]) == count
            for name, count in EXPECTED_COLLECTION_COUNTS.items()
        )
        and summary["total_differences"].eq(0).all()
    )
    print(f"\nTopic qrels validation {'PASSED' if passed else 'FAILED'}", flush=True)
    print(f"Elapsed time: {perf_counter() - start:.1f} seconds", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
