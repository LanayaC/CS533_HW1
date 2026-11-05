"""Select and validate Appendix A.4 known-item queries without writing datasets.

Run from the repository root with configurable input paths::

    python -m src.known_item_queries --dataset ../data/articles_mapped.h5 \
        --official-dir ../data/nyt-tc-2026

See notes/known_item_query_validation.md for the complete workflow.
"""

import argparse
import hashlib
from numbers import Integral
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


QUERY_COLUMNS = ["qid", "query", "docno", "year"]
_PERIODS = {
    "C1": ("KI1", "queries_ki_P1.tsv"),
    "C2": ("KI2", "queries_ki_P2.tsv"),
}
_ALPHABETIC_TOKEN = re.compile(r"[a-z]+")


def _url_hash(web_url: str) -> int:
    return int(hashlib.md5(web_url.encode("utf-8")).hexdigest()[:12], 16)


def _query_from_slug(slug: str) -> str:
    return " ".join(
        token for token in slug.lower().split("-") if _ALPHABETIC_TOKEN.fullmatch(token)
    )


def select_known_item_queries(
    collections: dict[str, pd.DataFrame],
    eligibility: pd.DataFrame,
    n_queries: int = 1000,
) -> dict[str, pd.DataFrame]:
    """Select eligible C1/C2 articles by the 12-digit MD5 prefix of the FULL URL.

    collections must contain reconstructed C1/C2 frames with docno, web_url,
    and year. eligibility must be the existing check_known_item_eligibility()
    result computed on ALL cleaned articles, including those outside C1/C2.
    All collection URLs must have eligibility records. Inputs are not modified.

    For each collection, retain eligible members, sort by the integer value of
    the first 12 hexadecimal digits of MD5(web_url.encode('utf-8')), and take
    n_queries. Stable sorting preserves collection order for equal hashes;
    Appendix A.4 does not specify a tie-breaking rule. Raise ValueError if
    either collection has insufficient eligible candidates. The default is
    1,000; a smaller positive n_queries supports synthetic examples/tests.

    Query text uses the lowercased slug split on hyphens, retaining whole
    [a-z]+ tokens, INCLUDING stop words and single-letter words. Do not use
    preprocess_text() or apply eligibility's stop-word filter to query text.

    Return a dict keyed C1/C2 with exactly qid, query, docno, year columns.
    IDs are KI1-0001... or KI2-0001..., in increasing hash order. Nothing is
    saved, no official files are read, and no relevance judgments are generated.
    """
    if isinstance(n_queries, bool) or not isinstance(n_queries, Integral) or n_queries < 1:
        raise ValueError("n_queries must be a positive integer.")
    status = eligibility.loc[:, ["web_url", "slug", "eligible"]].copy()
    if status["eligible"].isna().any() or not pd.api.types.is_bool_dtype(status["eligible"]):
        raise ValueError("Eligibility flags must be nonmissing booleans.")
    status = status.loc[status["web_url"].notna()]
    if status["web_url"].duplicated().any():
        raise ValueError("Eligibility contains duplicate web_url values.")
    status = status.set_index("web_url")

    selected_queries = {}
    for name, (prefix, _) in _PERIODS.items():
        frame = collections[name].loc[:, ["docno", "web_url", "year"]].copy()
        if frame.isna().any().any():
            raise ValueError(f"{name} contains missing collection values.")
        if frame["web_url"].duplicated().any() or frame["docno"].duplicated().any():
            raise ValueError(f"{name} contains duplicate URLs or document IDs.")
        if not frame["web_url"].isin(status.index).all():
            raise ValueError(f"Eligibility does not cover every {name} article.")
        candidates = frame.join(status, on="web_url", validate="one_to_one")
        candidates = candidates.loc[candidates["eligible"]].copy()
        if len(candidates) < n_queries:
            raise ValueError(
                f"{name} has {len(candidates)} eligible articles; {n_queries} are required."
            )
        if not candidates["slug"].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Eligible {name} articles must have extracted slugs.")
        candidates["_hash"] = candidates["web_url"].map(_url_hash)
        selected = candidates.sort_values("_hash", kind="stable").head(n_queries).copy()
        selected["qid"] = [f"{prefix}-{i:04d}" for i in range(1, n_queries + 1)]
        selected["query"] = selected["slug"].map(_query_from_slug)
        selected["year"] = selected["year"].astype("int64")
        selected_queries[name] = selected.loc[:, QUERY_COLUMNS].reset_index(drop=True)
    return selected_queries


def _check_queries(frame: pd.DataFrame, label: str) -> None:
    missing = set(QUERY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if frame[QUERY_COLUMNS].isna().any().any():
        raise ValueError(f"{label}: missing query record values.")
    for column in ["qid", "docno"]:
        if frame[column].duplicated().any():
            raise ValueError(f"{label}: duplicate {column} values.")


def compare_known_item_queries(
    queries: dict[str, pd.DataFrame], official_dir: str | Path
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """Read official TSVs ONLY to validate independently selected query frames.

    Match by qid: missing records are official-only IDs; extra records are
    selected-only IDs. Shared IDs with different query, docno, or year values
    are mismatched, counted once even if multiple fields differ. Return a
    C1/C2 summary and missing/extra/mismatched detail frames for each period.
    Row order is ignored because qid itself records the selection order.
    Neither the inputs nor official files are modified.
    """
    records, details = [], {}
    for name, (_, filename) in _PERIODS.items():
        selected = queries[name]
        _check_queries(selected, f"Selected {name}")
        official = pd.read_csv(
            Path(official_dir) / filename,
            sep="\t",
            dtype={"qid": str, "query": str, "docno": str, "year": "int64"},
            keep_default_na=False,
        )
        _check_queries(official, f"Official {name}")
        left, right = selected.set_index("qid"), official.set_index("qid")
        missing = official.loc[
            ~official["qid"].isin(left.index), QUERY_COLUMNS
        ].reset_index(drop=True)
        extra = selected.loc[
            ~selected["qid"].isin(right.index), QUERY_COLUMNS
        ].reset_index(drop=True)
        common = left.index.intersection(right.index, sort=False)
        fields = ["query", "docno", "year"]
        joined = left.loc[common, fields].add_suffix("_selected").join(
            right.loc[common, fields].add_suffix("_official")
        )
        masks = pd.DataFrame(
            {
                field: joined[f"{field}_selected"].ne(joined[f"{field}_official"])
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
                "selected_count": len(selected),
                "official_count": len(official),
                "missing": len(missing),
                "extra": len(extra),
                "mismatched": len(mismatched),
                "total_differences": len(missing) + len(extra) + len(mismatched),
            }
        )
    return pd.DataFrame(records).set_index("collection"), details


def main(argv: list[str] | None = None) -> int:
    """Run selection and validation independently of Jupyter; write no files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Original HDF5 path")
    parser.add_argument("--official-dir", type=Path, required=True, help="Official TSV directory")
    parser.add_argument("--hdf-key", default="data", help="HDF5 key (default: data)")
    parser.add_argument("--examples", type=int, default=5, help="Examples per difference type")
    args = parser.parse_args(argv)
    if args.examples < 0:
        parser.error("--examples must be nonnegative")

    start = perf_counter()
    print(f"Loading original dataset: {args.dataset.resolve()}", flush=True)
    original = pd.read_hdf(args.dataset, key=args.hdf_key)
    print(f"Original article count: {len(original):,}", flush=True)
    cleaned, cleaning_report = clean_articles(original)
    del original
    print("\nPart 1.2 cleaning (in order):", flush=True)
    print(cleaning_report.to_string(), flush=True)
    cleaned_count = len(cleaned)
    print(f"Final cleaned article count: {cleaned_count:,}", flush=True)

    print("\nReconstructing C1/C2 from cleaned articles...", flush=True)
    collections = build_collections(cleaned)
    for name, frame in collections.items():
        print(f"{name} document count: {len(frame):,}", flush=True)
    print("Checking slug eligibility across ALL cleaned articles...", flush=True)
    eligibility = check_known_item_eligibility(cleaned)
    del cleaned
    print("Selecting 1,000 queries per collection by full-URL MD5...", flush=True)
    queries = select_known_item_queries(collections, eligibility)
    for name, frame in queries.items():
        print(
            f"{name}: {len(frame):,} queries selected "
            f"({frame['qid'].iloc[0]} through {frame['qid'].iloc[-1]})",
            flush=True,
        )

    print(f"\nComparing with official query files: {args.official_dir.resolve()}", flush=True)
    summary, details = compare_known_item_queries(queries, args.official_dir)
    print(summary.to_string(), flush=True)
    for name, difference_types in details.items():
        for kind, frame in difference_types.items():
            if not frame.empty and args.examples:
                print(f"\n{name} {kind} examples:", flush=True)
                print(frame.head(args.examples).to_string(index=False), flush=True)

    passed = (
        cleaned_count == EXPECTED_CLEANED_COUNT
        and all(len(collections[name]) == count for name, count in EXPECTED_COLLECTION_COUNTS.items())
        and summary["selected_count"].eq(1000).all()
        and summary["official_count"].eq(1000).all()
        and summary["total_differences"].eq(0).all()
    )
    print(f"\nKnown-item validation {'PASSED' if passed else 'FAILED'}", flush=True)
    print(f"Elapsed time: {perf_counter() - start:.1f} seconds", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
