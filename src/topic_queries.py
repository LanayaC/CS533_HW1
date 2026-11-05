"""Independently reconstruct and validate the shared Table 2 topic queries.

From the repository root, validate without loading the article dataset::

    python -m src.topic_queries --official-dir ../data/nyt-tc-2026

Appendix A.5 uses the SAME 32 definitions for C1/P1 and C2/P2. Only relevance
judgments change between collections. No topic definitions depend on a period.
"""

import argparse
from pathlib import Path

import pandas as pd

from .document_collections import TOPIC_PATHS
from .topic_qrels import TOPIC_IDS


TOPIC_COLUMNS = ["tid", "query", "paths"]


def build_topic_queries() -> pd.DataFrame:
    """Reconstruct the 32 Table 2 definitions without reading official files.

    Reuse the existing independently transcribed topic-ID and URL-path maps.
    The query is the exact Table 2 topic label; paths is a space-separated
    list of its exact URL paths. Both books paths belong to T06. Return a new
    frame sorted T01-T32, with columns tid, query, paths. Use this one common
    definition table for BOTH periods; do not infer definitions from documents,
    official topics.tsv, or period-specific topic frequencies.
    """
    expected_ids = {f"T{i:02d}" for i in range(1, 33)}
    if len(TOPIC_IDS) != 32 or set(TOPIC_IDS.values()) != expected_ids:
        raise ValueError("The independently transcribed Table 2 IDs must be T01-T32.")
    if set(TOPIC_IDS) != set(TOPIC_PATHS.values()):
        raise ValueError("The independent topic-ID and URL-path maps disagree.")
    records = [
        {
            "tid": tid,
            "query": topic,
            "paths": " ".join(path for path, label in TOPIC_PATHS.items() if label == topic),
        }
        for topic, tid in TOPIC_IDS.items()
    ]
    return pd.DataFrame(records, columns=TOPIC_COLUMNS).sort_values("tid").reset_index(drop=True)


def _check_definitions(frame: pd.DataFrame, label: str) -> None:
    missing = set(TOPIC_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if frame[TOPIC_COLUMNS].isna().any().any():
        raise ValueError(f"{label}: missing topic definition values.")
    for column in TOPIC_COLUMNS:
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{label}: {column} must contain nonempty strings.")
    if frame["tid"].duplicated().any():
        raise ValueError(f"{label}: duplicate topic IDs prevent unambiguous comparison.")


def compare_topic_queries(
    definitions: pd.DataFrame, official_dir: str | Path
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Read topics.tsv only to validate already-reconstructed definitions.

    Match on exact tid. Missing IDs occur only officially; extra IDs occur
    only in the reconstruction. A shared ID with different query text or
    associated path SETS is one mismatch, even when both fields differ. Query
    text, path spelling, and case are compared exactly. Path order and spacing
    between listed paths are not significant; TSV row order is ignored.

    Return a one-row summary indexed by topics and missing/extra/mismatched
    detail frames. Duplicate IDs and missing/blank values raise ValueError.
    Do not modify the input or official file. This validation applies equally
    to the common topic definitions used for P1 and P2.
    """
    _check_definitions(definitions, "Reconstructed topics")
    official = pd.read_csv(
        Path(official_dir) / "topics.tsv", sep="\t", dtype=str, keep_default_na=False
    )
    _check_definitions(official, "Official topics")
    left, right = definitions.set_index("tid"), official.set_index("tid")
    missing = official.loc[
        ~official["tid"].isin(left.index), TOPIC_COLUMNS
    ].reset_index(drop=True)
    extra = definitions.loc[
        ~definitions["tid"].isin(right.index), TOPIC_COLUMNS
    ].reset_index(drop=True)
    common = left.index.intersection(right.index, sort=False)
    joined = left.loc[common, ["query", "paths"]].add_suffix("_reconstructed").join(
        right.loc[common, ["query", "paths"]].add_suffix("_official")
    )
    masks = pd.DataFrame(
        {
            "query": joined["query_reconstructed"].ne(joined["query_official"]),
            "paths": joined["paths_reconstructed"].map(lambda value: frozenset(value.split())).ne(
                joined["paths_official"].map(lambda value: frozenset(value.split()))
            ),
        },
        index=joined.index,
    )
    mismatched = joined.loc[masks.any(axis=1)].copy()
    mismatched["mismatched_fields"] = [
        ", ".join(field for field in ["query", "paths"] if row[field])
        for _, row in masks.loc[mismatched.index].iterrows()
    ]
    details = {
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched.reset_index(),
    }
    summary = pd.DataFrame(
        {
            "reconstructed_count": [len(definitions)],
            "official_count": [len(official)],
            "missing": [len(missing)],
            "extra": [len(extra)],
            "mismatched": [len(mismatched)],
            "total_differences": [len(missing) + len(extra) + len(mismatched)],
        },
        index=pd.Index(["topics"], name="validation"),
    )
    return summary, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, required=True, help="Directory containing topics.tsv")
    args = parser.parse_args(argv)
    definitions = build_topic_queries()
    summary, details = compare_topic_queries(definitions, args.official_dir)
    print(summary.to_string())
    print("The same 32 topic-query definitions apply to both P1 and P2.")
    for kind, frame in details.items():
        if not frame.empty:
            print(f"\n{kind} definitions:")
            print(frame.to_string(index=False))
    passed = (
        summary["reconstructed_count"].eq(32).all()
        and summary["official_count"].eq(32).all()
        and summary["total_differences"].eq(0).all()
    )
    print(f"Topic-query validation {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
