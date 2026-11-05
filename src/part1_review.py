"""One-HDF5-load Part 1 audit: reuse reconstruction, summarize, verify corpora.

Run from the repository root; all input/output paths are configurable::

    python -m src.part1_review --dataset ../data/articles_mapped.h5 \
        --official-dir ../data/nyt-tc-2026 --corpus-dir ../data/corpora \
        --output-dir ../data/part1_review --verify-corpus-text

The JSON summary is created exclusively outside the repository. Neither official
files, existing corpora, nor notebooks are written. Official data are loaded only
after independent reconstruction, and separately for reporting-only statistics.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import pandas as pd

from .document_collections import (
    EXPECTED_CLEANED_COUNT, EXPECTED_COLLECTION_COUNTS,
    build_collections, clean_articles, compare_collections,
)
from .known_item_eligibility import check_known_item_eligibility
from .known_item_queries import select_known_item_queries, compare_known_item_queries
from .known_item_qrels import build_known_item_qrels, compare_known_item_qrels
from .monthly_counts import monthly_article_counts
from .part1_statistics import official_reporting_statistics
from .preprocessing import preprocess_text
from .topic_qrels import build_topic_qrels, compare_topic_qrels
from .topic_queries import build_topic_queries, compare_topic_queries


def raw_data_statistics(articles: pd.DataFrame) -> dict:
    """Describe original, uncleaned rows; do not mutate the DataFrame."""
    dates = pd.to_datetime(articles["pub_date"], errors="coerce", format="mixed")
    years = dates.dt.year
    year_category = pd.crosstab(years, articles["mapped_section"])
    year_category.index.name = "year"
    period = pd.Series(None, index=articles.index, dtype=object)
    period.loc[years.between(2005, 2014)] = "P1"
    period.loc[years.between(2015, 2025)] = "P2"
    category_period = pd.crosstab(articles["mapped_section"], period).reindex(
        columns=["P1", "P2"], fill_value=0
    )
    category_period.index.name = "category"
    monthly = monthly_article_counts(articles, [2009, 2025]).reset_index()
    sports = monthly_article_counts(
        articles, [2023, 2024, 2025], categories=["Sports & Athletics"]
    ).reset_index()
    monthly["month"] = monthly["month"].dt.strftime("%Y-%m")
    sports["month"] = sports["month"].dt.strftime("%Y-%m")
    return {
        "source": "reproduced from original HDF5, before cleaning",
        "article_count": len(articles),
        "date_min": dates.min().isoformat() if dates.notna().any() else None,
        "date_max": dates.max().isoformat() if dates.notna().any() else None,
        "year_category_counts": year_category.reset_index().to_dict("records"),
        "category_period_counts": category_period.reset_index().to_dict("records"),
        "monthly_2009_2025": monthly.to_dict("records"),
        "sports_monthly_2023_2025": sports.to_dict("records"),
    }


def inspect_corpus(path: str | Path, expected_texts=None) -> dict:
    """Stream an existing UTF-8 corpus, optionally checking every article's text.

    Counts come from persisted tokens. If expected_texts is supplied in source
    article order, compare every line with the CURRENT preprocess_text() output.
    Keep only one tokenized article and the corpus vocabulary in memory. Never
    write the corpus. Source article texts need not be materialized as a list.
    """
    digest = hashlib.sha256()
    vocabulary = set()
    articles = tokens = empty_lines = bad_newlines = byte_count = mismatches = 0
    marker = object()
    expected = iter(expected_texts) if expected_texts is not None else None
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            articles += 1
            encoded = line.encode("utf-8")
            digest.update(encoded)
            byte_count += len(encoded)
            words = line.split()
            tokens += len(words)
            vocabulary.update(words)
            empty_lines += not words
            bad_newlines += "\r" in line or not line.endswith("\n")
            if expected is not None:
                text = next(expected, marker)
                if text is marker or line != " ".join(preprocess_text(text)) + "\n":
                    mismatches += 1
    if expected is not None:
        mismatches += sum(1 for _ in expected)
    return {
        "article_count": articles,
        "total_tokens": tokens,
        "distinct_tokens": len(vocabulary),
        "empty_lines": empty_lines,
        "non_lf_or_unterminated_lines": bad_newlines,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "current_tokenizer_text_checked": expected is not None,
        "mismatched_article_lines": mismatches if expected is not None else None,
    }


def query_target_statistics(queries, collections, clean_df) -> dict:
    """Reproduce processed query/target overlap with both explicit conventions.

    Token occurrence: count each processed query token occurrence. Unique term:
    use distinct processed query terms. Compute a fraction PER QUERY and then
    average across every query, including zero-token queries with fraction zero.
    The document side is a set for membership, using the same preprocess_text().
    """
    text_lookup = clean_df.loc[:, ["web_url", "text"]].set_index("web_url")["text"]
    result = {}
    for name, frame in queries.items():
        targets = collections[name].set_index("docno")["web_url"]
        records, acronym_examples = [], []
        for row in frame.itertuples(index=False):
            target_url = targets.loc[row.docno]
            target_text = text_lookup.loc[target_url]
            query_tokens = preprocess_text(row.query)
            target_tokens = set(preprocess_text(target_text))
            unique = set(query_tokens)
            occurrences = sum(token in target_tokens for token in query_tokens)
            occurrence_fraction = occurrences / len(query_tokens) if query_tokens else 0.0
            unique_fraction = len(unique & target_tokens) / len(unique) if unique else 0.0
            records.append({
                "qid": row.qid,
                "token_occurrence_fraction": occurrence_fraction,
                "unique_term_fraction": unique_fraction,
            })
            for token, dotted in [("nba", r"N\.B\.A\."), ("fbi", r"F\.B\.I\."), ("epa", r"E\.P\.A\.")]:
                if token not in query_tokens or not isinstance(target_text, str):
                    continue
                match = re.search(dotted, target_text, flags=re.IGNORECASE)
                if match and not any(example["term"] == token for example in acronym_examples):
                    acronym_examples.append({
                        "qid": row.qid, "query": row.query, "term": token,
                        "target_excerpt": target_text[max(0, match.start()-40):match.end()+60],
                        "dotted_form_tokens": preprocess_text(match.group()),
                        "term_in_processed_target": token in target_tokens,
                    })
        n = len(records)
        result[name] = {
            "query_count": n,
            "mean_token_occurrence_overlap_pct": (
                100 * sum(r["token_occurrence_fraction"] for r in records) / n if n else None
            ),
            "mean_unique_term_overlap_pct": (
                100 * sum(r["unique_term_fraction"] for r in records) / n if n else None
            ),
            "definition": "unweighted mean of per-query fractions; same shared tokenizer on both sides",
            "per_query": records,
            "acronym_examples": acronym_examples,
        }
    return result


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hdf-key", default="data")
    parser.add_argument("--verify-corpus-text", action="store_true")
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    if output_dir.is_relative_to(repository) or output_dir.is_relative_to(args.official_dir.resolve()):
        parser.error("Summary output must be outside the repository and official directory.")
    output_path = output_dir / "part1_results.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing summary: {output_path}")
    if (args.corpus_dir / ".corpus_write_in_progress").exists():
        raise RuntimeError("An incomplete corpus-write marker exists; inspect it before review.")

    start = perf_counter()
    print(f"Loading HDF5 ONCE: {args.dataset.resolve()}", flush=True)
    original = pd.read_hdf(args.dataset, key=args.hdf_key)
    raw_stats = raw_data_statistics(original)
    cleaned, cleaning = clean_articles(original)
    del original
    print(f"Cleaned articles: {len(cleaned):,}", flush=True)
    years = pd.to_datetime(cleaned["pub_date"], errors="coerce", format="mixed").dt.year
    period_counts = {"P1": int(years.between(2005, 2014).sum()), "P2": int(years.between(2015, 2025).sum())}

    print("Independently reconstructing all Part 1.5 artifacts...", flush=True)
    collections = build_collections(cleaned)
    eligibility = check_known_item_eligibility(cleaned)
    queries = select_known_item_queries(collections, eligibility)
    known_qrels = build_known_item_qrels(queries)
    topic_qrels = build_topic_qrels(collections)
    topic_queries = build_topic_queries()
    validations = {}
    for label, comparator, artifact in [
        ("collections", compare_collections, collections),
        ("known_item_queries", compare_known_item_queries, queries),
        ("known_item_qrels", compare_known_item_qrels, known_qrels),
        ("topic_qrels", compare_topic_qrels, topic_qrels),
        ("topic_queries", compare_topic_queries, topic_queries),
    ]:
        summary, _ = comparator(artifact, args.official_dir)
        validations[label] = summary.reset_index().to_dict("records")
        print(f"\n{label}:\n{summary.to_string()}", flush=True)

    print("Computing reporting statistics and processed query/target overlap...", flush=True)
    reporting = official_reporting_statistics(args.official_dir)
    overlap = query_target_statistics(queries, collections, cleaned)
    print("Reading existing corpora; never overwriting them...", flush=True)
    corpus_stats = {}
    for period, limits in [("P1", (2005, 2014)), ("P2", (2015, 2025))]:
        expected = cleaned.loc[years.between(*limits), "text"] if args.verify_corpus_text else None
        corpus_stats[period] = inspect_corpus(args.corpus_dir / f"corpus_{period}.txt", expected)
        print(f"{period} corpus: {corpus_stats[period]}", flush=True)

    passed = (
        len(cleaned) == EXPECTED_CLEANED_COUNT
        and all(len(collections[name]) == count for name, count in EXPECTED_COLLECTION_COUNTS.items())
        and len(topic_queries) == 32
        and all(row["total_differences"] == 0 for rows in validations.values() for row in rows)
        and sum(period_counts.values()) == len(cleaned)
        and all(
            stats["article_count"] == period_counts[period]
            and stats["non_lf_or_unterminated_lines"] == 0
            and stats["mismatched_article_lines"] in (None, 0)
            for period, stats in corpus_stats.items()
        )
    )
    result = {
        "passed": bool(passed),
        "elapsed_seconds": perf_counter() - start,
        "source_paths": {"dataset": str(args.dataset.resolve()), "official_dir": str(args.official_dir.resolve()), "corpus_dir": str(args.corpus_dir.resolve())},
        "provenance": {
            "raw_data": "reproduced from original HDF5 in this single-load run",
            "reconstruction": "independent builders before official validation; no official construction inputs",
            "corpus_statistics": "streamed persisted corpora; current tokenizer comparison indicated per corpus",
            "official_reporting": "official files used ONLY for reporting tables after reconstruction",
            "overlap": "reproduced from independent known-item queries and cleaned target text",
        },
        "raw": raw_stats,
        "cleaning": cleaning.reset_index().to_dict("records"),
        "cleaned_count": len(cleaned),
        "training_article_counts": period_counts,
        "corpora": corpus_stats,
        "validations": validations,
        "official_reporting": reporting,
        "processed_overlap": overlap,
        "preprocessing_examples": [
            {"original": text, "tokens": preprocess_text(text)}
            for text in ["The U.S. President\u2019s NEW policy in 2025!", "The N.B.A. team's state-of-the-art plan costs 3.14 dollars."]
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
        handle.write("\n")
    print(f"\nPart 1 review {'PASSED' if passed else 'FAILED'}; summary: {output_path}", flush=True)
    print(f"Elapsed: {result['elapsed_seconds']:.1f} seconds", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
