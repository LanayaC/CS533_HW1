"""Reconstruct and validate C1/C2 using Appendix A of Assignment 1.

Run from the repository root (paths are configurable)::

    python -m src.document_collections --dataset ../data/articles_mapped.h5 \
        --official-dir ../data/nyt-tc-2026

The command loads HDF5, applies Part 1.2 cleaning, reconstructs collections,
and ONLY THEN loads the official TSVs for validation. It writes no files.
See notes/part_1_5_validation.md for definitions and reproduction instructions.
"""

import argparse
from pathlib import Path
import re
from time import perf_counter

import pandas as pd


# Appendix A.2, copied exactly. The URL's date is not used for period selection.
URL_PATTERN = re.compile(
    r"^https://www\.nytimes\.com/(\d{4})/(\d{2})/(\d{2})/"
    r"(?P<path>.+)/(?P<slug>[^/]+?)(?:\.html)?$"
)

# Table 2: 32 topic names and 33 EXACT paths. No official files define this map.
TOPIC_PATHS = {
    "world/africa": "africa",
    "arts/design": "art",
    "world/asia": "asia",
    "sports/baseball": "baseball",
    "sports/basketball": "basketball",
    "books": "books",
    "books/review": "books",
    "arts/dance": "dance",
    "business/economy": "economy",
    "world/europe": "europe",
    "fashion": "fashion",
    "dining": "food",
    "sports/football": "football",
    "sports/golf": "golf",
    "health": "health",
    "sports/hockey": "hockey",
    "world/americas": "latin america",
    "business/media": "media",
    "world/middleeast": "middle east",
    "movies": "movies",
    "arts/music": "music",
    "sports/olympics": "olympics",
    "your-money": "personal finance",
    "us/politics": "politics",
    "realestate": "real estate",
    "science": "science",
    "sports/soccer": "soccer",
    "technology": "technology",
    "arts/television": "television",
    "sports/tennis": "tennis",
    "theater": "theater",
    "travel": "travel",
    "fashion/weddings": "weddings",
}
COLLECTION_COLUMNS = ["docno", "web_url", "year", "topic"]
PERIODS = {"C1": (2005, 2014), "C2": (2015, 2025)}
OFFICIAL_FILENAMES = {"C1": "collection_P1.tsv", "C2": "collection_P2.tsv"}
EXPECTED_CLEANED_COUNT = 970_720
EXPECTED_COLLECTION_COUNTS = {"C1": 173_603, "C2": 210_664}


def clean_articles(articles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Part 1.2's three steps in order, without mutating the input.

    Remove text with fewer than 10 whitespace-separated tokens, then duplicate
    text, then duplicate web_url values. Each duplicate step keeps the first
    surviving occurrence in input order. No tokenizer or extra cleaning is used.
    Return the cleaned frame and a per-step report of removed/remaining rows.
    """
    short = articles["text"].fillna("").str.count(r"\S+") < 10
    cleaned = articles.loc[~short]
    records = [("short_text", int(short.sum()), len(cleaned))]

    for step, column in [("duplicate_text", "text"), ("duplicate_url", "web_url")]:
        duplicates = cleaned[column].duplicated(keep="first")
        removed = int(duplicates.sum())
        cleaned = cleaned.loc[~duplicates]
        records.append((step, removed, len(cleaned)))

    report = pd.DataFrame(records, columns=["step", "removed", "remaining"])
    return cleaned.copy(), report.set_index("step")


def _topic_from_url(url: object) -> str | None:
    if not isinstance(url, str):
        return None
    match = URL_PATTERN.match(url)
    return TOPIC_PATHS.get(match.group("path")) if match else None


def build_collections(clean_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build C1/C2 from cleaned articles, independently of official files.

    Follow Appendix A.1-A.3: derive year from pub_date, match the exact A.2
    regex, and look up the extracted path in Table 2. Do not use category
    metadata, the URL's year, prefix matching, or URL normalization.

    Sort each period by full publication timestamp, then web_url. Assign
    zero-based C1-/C2- docnos with at least six digits. Return exactly docno,
    web_url, year, topic columns with a fresh RangeIndex. Invalid/missing
    dates cannot belong to a period. Missing/nonmatching URLs and unlisted
    paths are excluded. Input rows are not modified or cleaned again.
    """
    source = clean_df.loc[:, ["pub_date", "web_url"]].copy()
    if source["web_url"].dropna().duplicated().any():
        raise ValueError("clean_df contains duplicate URLs; apply Part 1.2 first.")
    source["pub_date"] = pd.to_datetime(
        source["pub_date"], errors="coerce", format="mixed"
    )
    source["year"] = source["pub_date"].dt.year
    source["topic"] = source["web_url"].map(_topic_from_url)

    collections = {}
    for name, (first, last) in PERIODS.items():
        selected = source.loc[
            source["year"].between(first, last) & source["topic"].notna()
        ].sort_values(["pub_date", "web_url"], kind="stable").reset_index(drop=True)
        selected.insert(0, "docno", [f"{name}-{i:06d}" for i in range(len(selected))])
        selected["year"] = selected["year"].astype("int64")
        collections[name] = selected.loc[:, COLLECTION_COLUMNS].copy()
    return collections


def _check_collection(frame: pd.DataFrame, label: str) -> None:
    missing_columns = set(COLLECTION_COLUMNS).difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{label}: missing columns {sorted(missing_columns)}")
    if frame[COLLECTION_COLUMNS].isna().any().any():
        raise ValueError(f"{label}: collection records contain missing values.")
    for column in ["web_url", "docno"]:
        if frame[column].duplicated().any():
            raise ValueError(f"{label}: duplicate {column} values prevent comparison.")


def compare_collections(
    collections: dict[str, pd.DataFrame], official_dir: str | Path
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """Validate already-reconstructed collections against official TSV files.

    Match on web_url. Missing rows occur only in the official collection;
    extra rows occur only in the reconstruction. Shared URLs with a different
    docno, year, or topic are mismatched, counted ONCE per affected record.
    Detail frames retain the records and both versions of mismatched fields.
    TSV row order and index labels do not themselves count as differences.
    Duplicate docnos/URLs or missing required values raise ValueError.

    Return a summary indexed by C1/C2 and details[name][missing/extra/mismatched].
    Neither input collections nor official files are modified.
    """
    records = []
    details = {}
    for name, filename in OFFICIAL_FILENAMES.items():
        rebuilt = collections[name]
        _check_collection(rebuilt, f"Reconstructed {name}")
        official = pd.read_csv(
            Path(official_dir) / filename,
            sep="\t",
            dtype={"docno": str, "web_url": str, "year": "int64", "topic": str},
            keep_default_na=False,
        )
        _check_collection(official, f"Official {name}")
        left = rebuilt.set_index("web_url")
        right = official.set_index("web_url")
        missing = official.loc[
            ~official["web_url"].isin(left.index), COLLECTION_COLUMNS
        ].reset_index(drop=True)
        extra = rebuilt.loc[
            ~rebuilt["web_url"].isin(right.index), COLLECTION_COLUMNS
        ].reset_index(drop=True)

        common = left.index.intersection(right.index, sort=False)
        fields = ["docno", "year", "topic"]
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
        mismatched = mismatched.reset_index()
        details[name] = {"missing": missing, "extra": extra, "mismatched": mismatched}
        records.append(
            {
                "collection": name,
                "reconstructed_count": len(rebuilt),
                "official_count": len(official),
                "missing": len(missing),
                "extra": len(extra),
                "mismatched": len(mismatched),
                "total_differences": len(missing) + len(extra) + len(mismatched),
            }
        )
    return pd.DataFrame(records).set_index("collection"), details


def main(argv: list[str] | None = None) -> int:
    """Run the reproducible, read-only validation; return 0 only if all checks pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Original HDF5 path")
    parser.add_argument(
        "--official-dir", type=Path, required=True, help="Directory of official TSV files"
    )
    parser.add_argument("--hdf-key", default="data", help="HDF5 key (default: data)")
    parser.add_argument(
        "--examples", type=int, default=5, help="Maximum examples per difference type"
    )
    args = parser.parse_args(argv)
    if args.examples < 0:
        parser.error("--examples must be nonnegative")

    start = perf_counter()
    print(f"Loading original dataset: {args.dataset.resolve()}", flush=True)
    articles = pd.read_hdf(args.dataset, key=args.hdf_key)
    print(f"Original article count: {len(articles):,}", flush=True)
    cleaned, cleaning_report = clean_articles(articles)
    del articles
    print("\nPart 1.2 cleaning (in order):", flush=True)
    print(cleaning_report.to_string(), flush=True)
    cleaned_count = len(cleaned)
    print(
        f"\nFinal cleaned article count: {cleaned_count:,} "
        f"(expected {EXPECTED_CLEANED_COUNT:,})",
        flush=True,
    )
    print("Reconstructing C1/C2 from cleaned articles only...", flush=True)
    collections = build_collections(cleaned)
    del cleaned
    for name, frame in collections.items():
        print(
            f"{name} document count: {len(frame):,} "
            f"(expected {EXPECTED_COLLECTION_COUNTS[name]:,})",
            flush=True,
        )
    print(f"\nValidating against: {args.official_dir.resolve()}", flush=True)
    summary, details = compare_collections(collections, args.official_dir)
    print(summary.to_string(), flush=True)
    for name, difference_types in details.items():
        for kind, frame in difference_types.items():
            if not frame.empty and args.examples:
                print(f"\n{name} {kind} examples:", flush=True)
                print(frame.head(args.examples).to_string(index=False), flush=True)

    passed = (
        cleaned_count == EXPECTED_CLEANED_COUNT
        and all(
            len(collections[name]) == expected
            for name, expected in EXPECTED_COLLECTION_COUNTS.items()
        )
        and summary["total_differences"].eq(0).all()
    )
    print(f"\nValidation {'PASSED' if passed else 'FAILED'}", flush=True)
    print(f"Elapsed time: {perf_counter() - start:.1f} seconds", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
