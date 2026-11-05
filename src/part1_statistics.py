"""Reporting-only statistics from official Part 1 test-collection files.

These functions do not reconstruct collections, select queries, or derive
relevance judgments. Official metadata is used only for descriptive reporting.
"""

from pathlib import Path

import pandas as pd


_PERIODS = {
    "P1": ("collection_P1.tsv", "queries_ki_P1.tsv", 2005, 2014),
    "P2": ("collection_P2.tsv", "queries_ki_P2.tsv", 2015, 2025),
}


def _check_table(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if frame.loc[:, required].isna().any().any():
        raise ValueError(f"{label}: missing required values.")
    for column in required:
        if column != "year" and frame[column].map(
            lambda value: not isinstance(value, str) or not value.strip()
        ).any():
            raise ValueError(f"{label}: {column} must contain nonempty strings.")


def _query_length_statistics(queries: pd.DataFrame) -> dict:
    lengths = queries["query"].str.split().str.len().astype("int64")
    description = lengths.describe()
    result = {}
    for name, value in description.items():
        if pd.isna(value):
            result[name] = None
        elif name in {"count", "min", "max"}:
            result[name] = int(value)
        else:
            result[name] = float(value)
    return result


def official_reporting_statistics(official_dir: str | Path) -> dict:
    """Return JSON-serializable topic distributions and known-item summaries.

    Read official topics.tsv, collection_P1/P2.tsv and queries_ki_P1/P2.tsv for
    REPORTING ONLY. Reconstruction must use the independent builders elsewhere.
    Include every declared topic in both periods, even when its count is zero.
    Query lengths count whitespace-separated words in the original query text,
    including stop words. Standard deviation uses ddof=1; quartiles use pandas'
    default linear interpolation. Undefined summaries (e.g. std for one query)
    become None rather than nonstandard JSON NaN values.

    Choose the first five complete queries by ascending qid for examples.
    Validate unique IDs/URLs, known topic labels, period years, target-document
    coverage, and agreement between query and target years. Do not modify any
    files or silently drop inconsistent metadata. The helper accepts smaller
    synthetic topic/query tables as well as the full official collection.
    """
    official_dir = Path(official_dir)
    topics = pd.read_csv(official_dir / "topics.tsv", sep="\t", dtype=str, keep_default_na=False)
    _check_table(topics, ["tid", "query", "paths"], "topics.tsv")
    if topics["tid"].duplicated().any() or topics["query"].duplicated().any():
        raise ValueError("topics.tsv: topic IDs and query labels must be unique.")
    topics = topics.sort_values("tid", kind="stable")

    result = {
        "provenance": "Official metadata files, used for descriptive reporting only.",
        "measurement_definitions": {
            "query_length": "Whitespace-separated original query words, including stop words.",
            "standard_deviation": "Sample standard deviation (ddof=1).",
            "quartiles": "Pandas default linear interpolation.",
            "examples": "First five records by ascending qid, with full untruncated query text.",
        },
        "topic_distribution": [
            {"tid": row.tid, "topic": row.query, "P1": 0, "P2": 0}
            for row in topics.itertuples(index=False)
        ],
        "query_length_statistics": {},
        "known_item_examples": {},
        "collection_sizes": {},
        "known_item_query_counts": {},
    }
    for period, (collection_name, query_name, first, last) in _PERIODS.items():
        collection = pd.read_csv(
            official_dir / collection_name,
            sep="\t",
            dtype={"docno": str, "web_url": str, "year": "int64", "topic": str},
            keep_default_na=False,
        )
        queries = pd.read_csv(
            official_dir / query_name,
            sep="\t",
            dtype={"qid": str, "query": str, "docno": str, "year": "int64"},
            keep_default_na=False,
        )
        _check_table(collection, ["docno", "web_url", "year", "topic"], collection_name)
        _check_table(queries, ["qid", "query", "docno", "year"], query_name)
        for column in ["docno", "web_url"]:
            if collection[column].duplicated().any():
                raise ValueError(f"{collection_name}: duplicate {column} values.")
        for column in ["qid", "docno"]:
            if queries[column].duplicated().any():
                raise ValueError(f"{query_name}: duplicate {column} values.")
        if not collection["topic"].isin(topics["query"]).all():
            raise ValueError(f"{collection_name}: unknown topic labels.")
        if not collection["year"].between(first, last).all():
            raise ValueError(f"{collection_name}: year outside {period}.")
        if not queries["year"].between(first, last).all():
            raise ValueError(f"{query_name}: year outside {period}.")
        targets = collection.set_index("docno")["year"]
        if not queries["docno"].isin(targets.index).all():
            raise ValueError(f"{query_name}: query target absent from the collection.")
        if not queries["year"].eq(queries["docno"].map(targets)).all():
            raise ValueError(f"{query_name}: query and target publication years disagree.")

        counts = collection["topic"].value_counts()
        for row in result["topic_distribution"]:
            row[period] = int(counts.get(row["topic"], 0))
        result["collection_sizes"][period] = int(len(collection))
        result["known_item_query_counts"][period] = int(len(queries))
        result["query_length_statistics"][period] = _query_length_statistics(queries)
        examples = queries.sort_values("qid", kind="stable").head(5).copy()
        examples["length"] = examples["query"].str.split().str.len().astype("int64")
        result["known_item_examples"][period] = examples.to_dict("records")
    return result
