"""Part 4.1 metrics and Part 4.3 past-to-future parameter transfer.

Run only on existing CSVs, without loading indexes, embeddings, or article text::

    python -m src.evaluation --official-dir ../data/nyt-tc-2026 \
        --results-dir ../data/part3_results --output-dir ../data/part4_results

Relevance is binary (official grade >= 1); CSV rank is authoritative. Precision
uses the requested cutoff even for short runs. Assignment AP@k divides by
min(Rq,k). Native trec_eval map_cut divides by Rq, so report both its native
discrepancy and the independently computed score rescaled to this assignment.
Use --experiment past_future for QE/hybrid tuning using saved CSVs only.
Use --experiment main_results for Part 4.2 tables with frozen P1 settings.
Use --experiment temporal for Part 4.4 period-by-model and publication-year analysis.
Use --experiment per_query for Part 4.5 vocabulary, drift, topic and hybrid examples.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from .document_collections import EXPECTED_COLLECTION_COUNTS
from .retrieval import FUSION_ALPHAS, external_path, publish_summary, read_table, sha256_file, write_json


KNOWN_METRICS = ("RR@10", "Success@1", "Success@10", "Recall@100")
TOPIC_METRICS = ("P@10", "P@100", "AP@100")
MEAN_NAMES = {"RR@10": "MRR@10", "AP@100": "MAP@100"}
EXTERNAL_NAMES = {"RR@10": "recip_rank", "Success@1": "success_1", "Success@10": "success_10",
                  "Recall@100": "recall_100", "P@10": "P_10", "P@100": "P_100", "AP@100": "map_cut_100"}
TREC_MAP_SOURCE = "https://github.com/usnistgov/trec_eval/blob/master/m_map_cut.c"


def _relevance(ranking, relevant, k):
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("Metric cutoffs must be positive integers")
    ranking = list(ranking)
    if any(not isinstance(docno, str) or not docno for docno in ranking):
        raise ValueError("Rankings require nonempty document IDs")
    if len(set(ranking)) != len(ranking):
        raise ValueError("Duplicate ranked documents would inflate evaluation")
    relevant = set(relevant)
    return [docno in relevant for docno in ranking[:k]], len(relevant)


def precision_at_k(ranking, relevant, k):
    flags, _ = _relevance(ranking, relevant, k)
    return sum(flags) / k


def recall_at_k(ranking, relevant, k):
    flags, total = _relevance(ranking, relevant, k)
    return sum(flags) / total if total else 0.0


def reciprocal_rank_at_k(ranking, relevant, k):
    flags, _ = _relevance(ranking, relevant, k)
    return next((1 / rank for rank, hit in enumerate(flags, 1) if hit), 0.0)


def success_at_k(ranking, relevant, k):
    flags, _ = _relevance(ranking, relevant, k)
    return float(any(flags))


def average_precision_at_k(ranking, relevant, k):
    """Sum precision at relevant ranks <= k, divided by min(total relevant,k)."""
    flags, total = _relevance(ranking, relevant, k)
    if not total:
        return 0.0
    hits, precisions = 0, []
    for rank, hit in enumerate(flags, 1):
        if hit:
            hits += 1
            precisions.append(hits / rank)
    return math.fsum(precisions) / min(total, k)


def evaluate_queries(query_ids, rankings, qrels, query_type):
    """Include every official query, with zeros for absent or unsuccessful runs."""
    query_ids = list(query_ids)
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("Evaluation needs nonempty, unique official query IDs")
    if query_type not in {"known_item", "topics"}:
        raise ValueError("Choose known_item or topics")
    rows = []
    for qid in query_ids:
        ranking = rankings.get(qid, [])
        relevant = {docno for docno, grade in qrels.get(qid, {}).items() if grade >= 1}
        if query_type == "known_item":
            metrics = {"RR@10": reciprocal_rank_at_k(ranking, relevant, 10),
                       "Success@1": success_at_k(ranking, relevant, 1),
                       "Success@10": success_at_k(ranking, relevant, 10),
                       "Recall@100": recall_at_k(ranking, relevant, 100)}
        else:
            metrics = {"P@10": precision_at_k(ranking, relevant, 10),
                       "P@100": precision_at_k(ranking, relevant, 100),
                       "AP@100": average_precision_at_k(ranking, relevant, 100)}
        rows.append({"qid": qid, "relevant_total": len(relevant), **metrics})
    names = KNOWN_METRICS if query_type == "known_item" else TOPIC_METRICS
    means = {MEAN_NAMES.get(name, name): math.fsum(row[name] for row in rows) / len(query_ids) for name in names}
    return {"queries": len(query_ids), "per_query": rows, "means": means}


def load_qrels(path, official_qids, official_docnos):
    """Validate TREC qid/iteration/docno/grade rows against the official IDs."""
    official_qids = list(official_qids)
    allowed_qids, allowed_docs = set(official_qids), set(official_docnos)
    result = {qid: {} for qid in official_qids}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"Expected four qrels fields: {path}:{number}")
            qid, _, docno, grade = fields
            if qid not in allowed_qids or docno not in allowed_docs:
                raise ValueError(f"Unofficial qid/docno in {path}:{number}")
            if docno in result[qid]:
                raise ValueError(f"Duplicate qrels query/document pair: {path}:{number}")
            result[qid][docno] = int(grade)
    return result


def load_run(path, official_qids, official_docnos):
    """Read explicit ranks, allowing missing queries and short runs to score zero.

    Unjudged documents are valid collection members and count as nonrelevant.
    Invalid IDs, duplicate documents, rank gaps and increasing scores are errors.
    """
    official_qids = list(official_qids)
    allowed_docs = set(official_docnos)
    rankings = {qid: [] for qid in official_qids}
    seen = {qid: set() for qid in official_qids}
    previous = dict.fromkeys(official_qids, math.inf)
    rows = 0
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = ["qid", "docno", "rank", "score"]
        if reader.fieldnames != columns:
            raise ValueError("Expected run columns qid,docno,rank,score")
        for row in reader:
            if None in row or any(row.get(key) in (None, "") for key in columns):
                raise ValueError("Missing run values or extra columns")
            qid, docno = row["qid"], row["docno"]
            if qid not in rankings or docno not in allowed_docs:
                raise ValueError("Run contains an unofficial query or document ID")
            rank, score = int(row["rank"]), float(row["score"])
            if rank != len(rankings[qid]) + 1 or docno in seen[qid]:
                raise ValueError("Run has a rank gap or duplicate document")
            if not math.isfinite(score) or score > previous[qid]:
                raise ValueError("Run scores must be finite and descending")
            rankings[qid].append(docno)
            seen[qid].add(docno)
            previous[qid] = score
            rows += 1
    details = {"rows": rows, "queries_with_results": sum(bool(v) for v in rankings.values()),
               "missing_queries": [qid for qid, docs in rankings.items() if not docs],
               "queries_with_fewer_than_100_results": sum(len(v) < 100 for v in rankings.values())}
    return rankings, details


def compare_with_pytrec_eval(query_ids, rankings, qrels, evaluation, query_type, tolerance=1e-12):
    """Compare every query, reporting native AND definition-adjusted differences.

    External scores encode saved rank monotonically, preventing trec_eval's
    score-tie policy from changing the input order. This does not change the run
    CSV. Truncate to 10 before independent recip_rank (which has no cutoff).
    Native map_cut@100 / Rq becomes assignment AP by multiplying Rq/min(Rq,100).
    Its numerator is calculated entirely by the external C evaluator.
    """
    import pytrec_eval

    query_ids = list(query_ids)
    judgments = {qid: qrels.get(qid, {}) for qid in query_ids}
    run100 = {qid: {docno: float(100 - i) for i, docno in enumerate(rankings.get(qid, [])[:100])}
              for qid in query_ids}
    if query_type == "known_item":
        run10 = {qid: dict(list(docs.items())[:10]) for qid, docs in run100.items()}
        short = pytrec_eval.RelevanceEvaluator(judgments, {"recip_rank", "success.1,10"}).evaluate(run10)
        full = pytrec_eval.RelevanceEvaluator(judgments, {"recall.100", "num_rel"}).evaluate(run100)
        external = {qid: {**short.get(qid, {}), **full.get(qid, {})} for qid in query_ids}
        names = KNOWN_METRICS
    elif query_type == "topics":
        external = pytrec_eval.RelevanceEvaluator(judgments, {"P.10,100", "map_cut.100", "num_rel"}).evaluate(run100)
        names = TOPIC_METRICS
    else:
        raise ValueError("Choose known_item or topics")
    compared = []
    for row in evaluation["per_query"]:
        qid, total = row["qid"], row["relevant_total"]
        external_total = int(external.get(qid, {}).get("num_rel", 0))
        if external_total != total:
            raise AssertionError(f"Independent relevant-document count differs for {qid}")
        for name in names:
            if total and EXTERNAL_NAMES[name] not in external.get(qid, {}):
                raise RuntimeError(f"Independent evaluator omitted {name} for {qid}")
            native = float(external.get(qid, {}).get(EXTERNAL_NAMES[name], 0.0))
            adjusted = native * external_total / min(external_total, 100) if name == "AP@100" and external_total else native
            compared.append({"qid": qid, "metric": name, "ours": row[name],
                             "external_relevant_total": external_total,
                             "external_metric": EXTERNAL_NAMES[name], "external_native": native,
                             "external_assignment_definition": adjusted,
                             "native_absolute_difference": abs(row[name] - native),
                             "assignment_absolute_difference": abs(row[name] - adjusted)})
    comparisons = {}
    for name in names:
        records = [row for row in compared if row["metric"] == name]
        raw = max(records, key=lambda row: row["native_absolute_difference"])
        adjusted = max(records, key=lambda row: row["assignment_absolute_difference"])
        comparisons[MEAN_NAMES.get(name, name)] = {
            "queries": len(records), "external_metric": EXTERNAL_NAMES[name],
            "native_definition_matches": name != "AP@100",
            "largest_native_per_query_absolute_difference": raw["native_absolute_difference"],
            "largest_native_difference_qid": raw["qid"],
            "largest_assignment_per_query_absolute_difference": adjusted["assignment_absolute_difference"],
            "largest_assignment_difference_qid": adjusted["qid"],
            "external_native_mean": math.fsum(row["external_native"] for row in records) / len(records),
            "external_assignment_mean": math.fsum(row["external_assignment_definition"] for row in records) / len(records),
            "passed": adjusted["assignment_absolute_difference"] <= tolerance}
    return {"passed": all(item["passed"] for item in comparisons.values()),
            "metrics": comparisons, "per_query": compared}


def _write_csv(path, columns, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_bm25_evaluation(official_dir, results_dir, output_dir):
    """Evaluate four separate official test sets; never run a retrieval method."""
    official_dir, results_dir = Path(official_dir).resolve(), Path(results_dir).resolve()
    output_dir = external_path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Use a new evaluation results directory: {output_dir}")
    started = perf_counter()
    results, per_query, mean_rows, comparison_rows, input_hashes = [], [], [], [], {}
    topics_path = official_dir / "topics.tsv"
    topics = read_table(topics_path, ["tid", "query"], "tid")
    if len(topics) != 32:
        raise ValueError("Expected 32 official topic queries")
    for collection, period, known_name in [("C1", "P1", "KI1"), ("C2", "P2", "KI2")]:
        collection_path = official_dir / f"collection_{period}.tsv"
        query_path = official_dir / f"queries_ki_{period}.tsv"
        run_path = results_dir / f"{collection.lower()}_bm25.csv"
        official = read_table(collection_path, ["docno", "web_url"], "docno")
        known = read_table(query_path, ["qid", "query", "docno"], "qid")
        if len(official) != EXPECTED_COLLECTION_COUNTS[collection] or len(known) != 1000:
            raise ValueError("Unexpected official collection or known-item query count")
        all_qids = known["qid"].tolist() + topics["tid"].tolist()
        if len(set(all_qids)) != 1032:
            raise ValueError("Known-item and topic IDs must be distinct")
        paths = [collection_path, query_path, topics_path, run_path,
                 official_dir / f"qrels_ki_{period}.txt", official_dir / f"qrels_topics_{period}.txt"]
        for path in paths:
            input_hashes[str(path)] = sha256_file(path)  # Check existence before evaluating.
        rankings, run_details = load_run(run_path, all_qids, official["docno"])
        for name, kind, qids, qrels_path in [
                (known_name, "known_item", known["qid"].tolist(), paths[-2]),
                (f"Topics_{collection}", "topics", topics["tid"].tolist(), paths[-1])]:
            qrels = load_qrels(qrels_path, qids, official["docno"])
            if any(not qrels[qid] for qid in qids):
                raise ValueError("An official query has no relevance judgments")
            if kind == "known_item":
                for qid, target in known[["qid", "docno"]].itertuples(index=False, name=None):
                    if {d for d, grade in qrels[qid].items() if grade >= 1} != {target}:
                        raise ValueError("Official known-item target and qrels disagree")
            evaluated = evaluate_queries(qids, rankings, qrels, kind)
            compared = compare_with_pytrec_eval(qids, rankings, qrels, evaluated, kind)
            names = KNOWN_METRICS if kind == "known_item" else TOPIC_METRICS
            for row in evaluated["per_query"]:
                for metric in names:
                    per_query.append({"test_set": name, "collection": collection, "qid": row["qid"],
                                      "relevant_total": row["relevant_total"], "metric": metric, "value": row[metric]})
            for metric, value in evaluated["means"].items():
                mean_rows.append({"test_set": name, "collection": collection, "queries": len(qids),
                                  "metric": metric, "value": value})
            comparison_rows.extend({"test_set": name, "collection": collection, **row} for row in compared["per_query"])
            totals = [row["relevant_total"] for row in evaluated["per_query"]]
            results.append({"test_set": name, "collection": collection, "query_type": kind,
                            "queries": len(qids), "means": evaluated["means"],
                            "relevant_documents_per_query": {"minimum": min(totals), "maximum": max(totals)},
                            "missing_ranking_queries": [qid for qid in qids if not rankings[qid]],
                            "source_run_validation": run_details, "validation": compared["metrics"],
                            "validation_passed": compared["passed"]})
    if any(sha256_file(path) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("An input changed during evaluation")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "bm25_per_query.csv", ["test_set", "collection", "qid", "relevant_total", "metric", "value"], per_query)
    _write_csv(output_dir / "bm25_metrics.csv", ["test_set", "collection", "queries", "metric", "value"], mean_rows)
    _write_csv(output_dir / "bm25_validation_per_query.csv",
               ["test_set", "collection", "qid", "metric", "ours", "external_relevant_total", "external_metric", "external_native",
                "external_assignment_definition", "native_absolute_difference", "assignment_absolute_difference"], comparison_rows)
    summary = {"part": "4.1", "method": "BM25", "independent_evaluator": "pytrec_eval (Terrier fork)",
               "independent_evaluator_version": importlib.metadata.version("pytrec-eval-terrier"),
               "all_passed": all(result["validation_passed"] for result in results), "tolerance": 1e-12,
               "test_sets": results, "input_sha256": input_hashes, "inputs_unchanged": True,
               "elapsed_seconds": perf_counter() - started,
               "definitions": {"relevance": "Official grade >= 1; unjudged documents are nonrelevant",
                   "MRR@10": "Mean reciprocal rank of first relevant result at ranks 1..10; otherwise zero",
                   "Success@k": "Mean indicator of any relevant result in ranks 1..k",
                   "Recall@100": "Retrieved relevant documents in top100 divided by total relevant documents",
                   "P@k": "Relevant documents in top k divided by k, including for short rankings",
                   "AP@100": "Sum of precision at relevant ranks <=100 divided by min(Rq,100)",
                   "MAP@100": "Mean assignment AP@100 over all official topic queries",
                   "missing_queries": "All official queries included; absent rankings and no hits receive zero",
                   "no_relevant_judgments": "AP and Recall are zero when Rq=0"},
               "external_adapters": {
                   "rank_order": "Strictly decreasing rank-based scores preserve saved CSV ranks, including score ties; original files untouched",
                   "MRR@10": "trec_eval recip_rank evaluated on runs truncated to ten documents",
                   "AP@100": "Native map_cut_100 divides by Rq, not min(Rq,100). Compare both native values and native * Rq/min(Rq,100)",
                   "empty_queries": "Submit empty runs; external omitted scores for queries without judgments are zero-completed",
                   "native_AP_definition_source": TREC_MAP_SOURCE},
               "files": ["bm25_per_query.csv", "bm25_metrics.csv", "bm25_validation_per_query.csv", "bm25_validation_summary.json"]}
    write_json(output_dir / "bm25_validation_summary.json", summary)
    if not summary["all_passed"]:
        raise AssertionError("Independent metric validation failed; saved per-query differences show the issue")
    return summary


TUNING_MODELS = ("M1", "M2", "GN")
TUNING_QUERY_COUNTS = {"known_item": 1000, "topics": 32}
TUNING_OBJECTIVES = {"known_item": ("RR@10", "MRR@10"), "topics": ("AP@100", "MAP@100")}
TUNING_TIE_RULE = "Maximize full-precision objective; exact ties: QE smaller m then smaller lambda; hybrid smaller alpha"


def parameter_key(family, parameters):
    """A canonical key for applying unchanged settings and resolving exact ties."""
    if family == "QE" and set(parameters) == {"m", "lambda"}:
        if (type(parameters["m"]) is int and parameters["m"] in (5, 10, 20)
                and not isinstance(parameters["lambda"], bool) and parameters["lambda"] in (.5, 1.)):
            return parameters["m"], parameters["lambda"]
    if family == "HYBRID" and set(parameters) == {"alpha"}:
        if not isinstance(parameters["alpha"], bool) and parameters["alpha"] in FUSION_ALPHAS:
            return (parameters["alpha"],)
    raise ValueError("Parameters do not match the assignment's QE or hybrid grid")


def select_best_setting(rows):
    """Select within one model/family/query-type/period cohort, without rounding."""
    rows = list(rows)
    if not rows:
        raise ValueError("No settings to select")
    fields = ("family", "model", "query_type", "period", "collection", "metric")
    cohort = tuple(rows[0][field] for field in fields)
    keys = set()
    for row in rows:
        if tuple(row[field] for field in fields) != cohort:
            raise ValueError("Parameter selection cannot mix periods, models, or query sets")
        if row["query_type"] not in TUNING_OBJECTIVES or row["metric"] != TUNING_OBJECTIVES[row["query_type"]][1]:
            raise ValueError("Wrong objective for the query type")
        if not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
            raise ValueError("Invalid effectiveness score")
        key = parameter_key(row["family"], row["parameters"])
        if key in keys:
            raise ValueError("Duplicate parameter setting")
        keys.add(key)
    best = min(rows, key=lambda row: (-row["score"], parameter_key(row["family"], row["parameters"])))
    return {**best, "parameters": dict(best["parameters"])}


def select_p1_settings(p1_rows):
    """Strict boundary: future rows are rejected, rather than used for selection."""
    groups = {}
    for row in p1_rows:
        if row["period"] != "P1" or row["collection"] != "C1":
            raise ValueError("P1 selection accepts only P1/C1 scores")
        key = row["family"], row["model"], row["query_type"]
        groups.setdefault(key, []).append(row)
    if not groups:
        raise ValueError("No P1 tuning scores")
    return [select_best_setting(rows) for rows in groups.values()]


def apply_p1_settings(p1_selection, p2_rows):
    """Apply frozen P1 keys exactly; the P2 oracle is a separate analysis only."""
    if any(row["period"] != "P2" or row["collection"] != "C2" for row in p2_rows):
        raise ValueError("Future testing requires P2/C2 scores")
    results = []
    for chosen in p1_selection:
        if chosen["period"] != "P1" or chosen["collection"] != "C1":
            raise ValueError("Transferred settings must come from P1/C1")
        group = [row for row in p2_rows if all(row[field] == chosen[field]
                 for field in ["family", "model", "query_type", "metric"])]
        matches = [row for row in group if parameter_key(row["family"], row["parameters"])
                   == parameter_key(chosen["family"], chosen["parameters"])]
        if len(matches) != 1:
            raise ValueError("The unchanged P1 setting has no unique P2 counterpart")
        transferred, oracle = matches[0], select_best_setting(group)
        gap = oracle["score"] - transferred["score"]
        if gap < 0:
            raise AssertionError("Oracle must be at least as effective as the transferred setting")
        results.append({"family": chosen["family"], "model": chosen["model"],
            "query_type": chosen["query_type"], "metric": chosen["metric"],
            "p1_selected_parameters": chosen["parameters"], "p1_score": chosen["score"],
            "p2_transferred_score": transferred["score"], "p2_oracle_parameters": oracle["parameters"],
            "p2_oracle_score": oracle["score"], "p2_oracle_gap": gap,
            "p1_filename": chosen["filename"], "p2_transferred_filename": transferred["filename"],
            "p2_oracle_filename": oracle["filename"], "p1_queries": chosen["queries"],
            "p2_queries": transferred["queries"], "oracle_role": "Analysis-only upper bound; not used for transfer"})
    return results


def discover_tuning_runs(results_dir):
    """Use checked manifests and checksums, not guessed run filenames."""
    results_dir = Path(results_dir).resolve()
    catalogue, hashes, seen = [], {}, set()
    for family, filename in [("QE", "qe_summary.json"), ("HYBRID", "hybrid_summary.json")]:
        path = results_dir / filename
        manifest = json.loads(path.read_text(encoding="utf-8"))
        hashes[str(path)] = sha256_file(path)
        if not manifest["all_complete"] or not manifest["all_completed_results_valid"]:
            raise ValueError("Tuning requires completed, validated retrieval runs")
        for pair in manifest["results"]:
            if pair["model"] not in TUNING_MODELS or pair["collection"] not in ("C1", "C2"):
                continue
            for config in pair["configurations"]:
                if family == "HYBRID" and config["method"] == "rrf":
                    continue
                params = ({"m": config["m"], "lambda": config["lambda"]} if family == "QE"
                          else {"alpha": config["alpha"]})
                key = family, pair["model"], pair["collection"], parameter_key(family, params)
                if key in seen:
                    raise ValueError("Duplicate tuning run in retrieval manifests")
                seen.add(key)
                run_path = (results_dir / config["filename"]).resolve()
                if not run_path.is_relative_to(results_dir) or not run_path.is_file():
                    raise ValueError("Tuning run must exist inside the saved-results directory")
                if sha256_file(run_path) != config["sha256"]:
                    raise ValueError(f"Saved retrieval checksum mismatch: {run_path}")
                hashes[str(run_path)] = config["sha256"]
                catalogue.append({"family": family, "model": pair["model"], "collection": pair["collection"],
                    "period": "P1" if pair["collection"] == "C1" else "P2", "parameters": params,
                    "filename": config["filename"], "sha256": config["sha256"]})
    for family in ("QE", "HYBRID"):
        grid = ({(m, weight) for m in (5, 10, 20) for weight in (.5, 1.)} if family == "QE"
                else {(alpha,) for alpha in FUSION_ALPHAS})
        for model in TUNING_MODELS:
            for collection in ("C1", "C2"):
                found = {parameter_key(family, row["parameters"]) for row in catalogue
                         if row["family"] == family and row["model"] == model and row["collection"] == collection}
                if found != grid:
                    raise ValueError(f"Incomplete parameter grid: {family}/{model}/{collection}")
    if len(catalogue) != 102:
        raise ValueError("Expected 36 QE and 66 linear-hybrid run files")
    return catalogue, hashes


def evaluate_tuning_period(catalogue, official_dir, results_dir, period):
    """Evaluate one period only; this function never loads the other period's qrels."""
    if period not in ("P1", "P2"):
        raise ValueError("Choose P1 or P2")
    collection = "C1" if period == "P1" else "C2"
    official_dir, results_dir = Path(official_dir), Path(results_dir)
    paths = [official_dir / f"collection_{period}.tsv", official_dir / f"queries_ki_{period}.tsv",
             official_dir / "topics.tsv", official_dir / f"qrels_ki_{period}.txt",
             official_dir / f"qrels_topics_{period}.txt"]
    hashes = {str(path): sha256_file(path) for path in paths}
    official = read_table(paths[0], ["docno", "web_url"], "docno")
    known = read_table(paths[1], ["qid", "query", "docno"], "qid")
    topics = read_table(paths[2], ["tid", "query"], "tid")
    if (len(official) != EXPECTED_COLLECTION_COUNTS[collection]
            or len(known) != TUNING_QUERY_COUNTS["known_item"] or len(topics) != TUNING_QUERY_COUNTS["topics"]):
        raise ValueError("Unexpected official query or collection count")
    qids = {"known_item": known["qid"].tolist(), "topics": topics["tid"].tolist()}
    all_qids = qids["known_item"] + qids["topics"]
    if len(set(all_qids)) != len(all_qids):
        raise ValueError("Known-item and topic IDs must be distinct")
    qrels = {"known_item": load_qrels(paths[3], qids["known_item"], official["docno"]),
             "topics": load_qrels(paths[4], qids["topics"], official["docno"])}
    for qid, target in known[["qid", "docno"]].itertuples(index=False, name=None):
        if {d for d, grade in qrels["known_item"][qid].items() if grade >= 1} != {target}:
            raise ValueError("Known-item qrels and official targets disagree")
    if any(not judgments for kind in qrels.values() for judgments in kind.values()):
        raise ValueError("An official query has no judgments")
    rows, per_query = [], []
    runs = [row for row in catalogue if row["period"] == period]
    for i, run in enumerate(runs, 1):
        if run["collection"] != collection:
            raise ValueError("Run period and collection disagree")
        rankings, details = load_run(results_dir / run["filename"], all_qids, official["docno"])
        if details["rows"] != len(all_qids) * 100 or any(len(docs) != 100 for docs in rankings.values()):
            raise ValueError("Tuning requires complete saved top-100 rankings for every official query")
        for kind in ("known_item", "topics"):
            objective, mean_name = TUNING_OBJECTIVES[kind]
            evaluation = evaluate_queries(qids[kind], rankings, qrels[kind], kind)
            rows.append({**run, "query_type": kind, "metric": mean_name,
                         "queries": len(qids[kind]), "score": evaluation["means"][mean_name]})
            per_query.extend({"period": period, "collection": collection, "family": run["family"],
                "model": run["model"], "query_type": kind, "parameters": run["parameters"],
                "qid": row["qid"], "metric": objective, "value": row[objective]}
                for row in evaluation["per_query"])
        if i % 10 == 0 or i == len(runs):
            print(f"{period}: evaluated {i}/{len(runs)} saved CSVs", flush=True)
    return rows, per_query, hashes


def alpha_plot_annotations(rows, p1_selection, transfer_results):
    """Locate frozen past choices on both curves; future oracle stays separate."""
    annotations = {}
    for chosen in p1_selection:
        if chosen["family"] != "HYBRID":
            continue
        if chosen["period"] != "P1" or chosen["collection"] != "C1":
            raise ValueError("Plot selection markers require frozen P1/C1 parameters")
        model, kind = chosen["model"], chosen["query_type"]
        if (model, kind) in annotations:
            raise ValueError("Duplicate frozen hybrid selection")
        matches = [row for row in transfer_results if row["family"] == "HYBRID"
                   and row["model"] == model and row["query_type"] == kind]
        if len(matches) != 1 or matches[0]["p1_selected_parameters"] != chosen["parameters"]:
            raise ValueError("Plot transfer record differs from frozen P1 parameters")
        transferred = matches[0]
        alpha, oracle_alpha = chosen["parameters"]["alpha"], transferred["p2_oracle_parameters"]["alpha"]
        def point(period, weight):
            points = [row for row in rows if row["family"] == "HYBRID" and row["model"] == model
                      and row["query_type"] == kind and row["period"] == period
                      and row["parameters"]["alpha"] == weight]
            if len(points) != 1:
                raise ValueError("A frozen choice or oracle point is absent from the alpha curve")
            return points[0]["score"]
        actual = (point("P1", alpha), point("P2", alpha), point("P2", oracle_alpha))
        expected = (chosen["score"], transferred["p2_transferred_score"], transferred["p2_oracle_score"])
        if actual != expected or transferred["p1_score"] != chosen["score"]:
            raise ValueError("Plot annotations disagree with saved effectiveness scores")
        annotations[(model, kind)] = {"p1_alpha": alpha, "p1_score": actual[0], "p2_score": actual[1],
                                      "p2_oracle_alpha": oracle_alpha, "p2_oracle_score": actual[2]}
    if set(annotations) != {(model, kind) for model in TUNING_MODELS for kind in TUNING_OBJECTIVES}:
        raise ValueError("Plot annotations need all six frozen hybrid choices")
    return annotations


PLOT_SELECTION_POLICY = {
    "star": "Frozen P1-selected alpha; P2 star shows its unchanged transferred effectiveness",
    "diamond": "P2 oracle, separately labeled as analysis only; never used for P1 selection",
    "table": "Displays exact selected alphas and rounded effectiveness; coincident markers may share a setting"}
PART42_PARAMETER_POLICY = "Part 4.2 QE and linear-hybrid rows must use the frozen P1 settings in part43_p1_selected.json, separately per model and query type; P2 oracle settings are analysis only"


def plot_alpha_curves(rows, output_dir, *, p1_selection, transfer_results, overwrite=False):
    """Four figures with frozen P1 markers, transferred P2 scores and separate oracles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    annotations = alpha_plot_annotations(rows, p1_selection, transfer_results)
    files = []
    colors, markers = ("#0072B2", "#D55E00", "#009E73"), ("o", "s", "^")
    for period in ("P1", "P2"):
        for kind in ("known_item", "topics"):
            fig, (ax, table_ax) = plt.subplots(2, 1, figsize=(8.5, 6.6), layout="constrained",
                                              gridspec_kw={"height_ratios": [3.7, 1.2]})
            try:
                table_rows = []
                for model, color, marker in zip(TUNING_MODELS, colors, markers):
                    curve = sorted([row for row in rows if row["family"] == "HYBRID" and row["period"] == period
                                    and row["query_type"] == kind and row["model"] == model],
                                   key=lambda row: row["parameters"]["alpha"])
                    if [row["parameters"]["alpha"] for row in curve] != list(FUSION_ALPHAS):
                        raise ValueError("Cannot plot an incomplete alpha curve")
                    ax.plot(FUSION_ALPHAS, [row["score"] for row in curve], color=color, marker=marker,
                            linewidth=1.8, markersize=4, label="GN (Google News)" if model == "GN" else model)
                    selected = annotations[(model, kind)]
                    score = selected["p1_score"] if period == "P1" else selected["p2_score"]
                    star = ax.scatter([selected["p1_alpha"]], [score], marker="*", s=150,
                                      c=color, edgecolors="black", linewidths=.7, zorder=5)
                    star.set_gid(f"P1-selected:{model}")
                    if period == "P2":
                        diamond = ax.scatter([selected["p2_oracle_alpha"]], [selected["p2_oracle_score"]],
                                             marker="D", s=220, facecolors="none", edgecolors=color,
                                             linewidths=1.6, zorder=4)
                        diamond.set_gid(f"P2-oracle-analysis-only:{model}")
                        table_rows.append([model, f"{selected['p1_alpha']:.1f}", f"{selected['p2_score']:.4f}",
                                           f"{selected['p2_oracle_alpha']:.1f}", f"{selected['p2_oracle_score']:.4f}"])
                    else:
                        table_rows.append([model, f"{selected['p1_alpha']:.1f}", f"{selected['p1_score']:.4f}",
                                           f"{selected['p2_score']:.4f}"])
                label = "Known-item queries" if kind == "known_item" else "Topic queries"
                collection = "C1" if period == "P1" else "C2"
                role = "tune on past" if period == "P1" else "fixed P1 settings on future"
                metric = TUNING_OBJECTIVES[kind][1]
                ax.set(title=f"{label}: {period} / {collection} ({role})", xlabel="BM25 weight alpha",
                       ylabel=metric, xlim=(-.025, 1.025), ylim=(0, 1))
                ax.set_xticks(FUSION_ALPHAS)
                ax.grid(alpha=.25)
                handles, labels = ax.get_legend_handles_labels()
                handles.append(Line2D([], [], marker="*", linestyle="none", markerfacecolor="white",
                                      markeredgecolor="black", markersize=12))
                labels.append("P1-selected alpha (fixed)")
                if period == "P2":
                    handles.append(Line2D([], [], marker="D", linestyle="none", markerfacecolor="none",
                                          markeredgecolor="black", markersize=9))
                    labels.append("P2 oracle (analysis only)")
                ax.legend(handles, labels, loc="upper left", ncols=2, fontsize=9)
                table_ax.axis("off")
                columns = (["Model", "Fixed P1 alpha", f"P2 {metric}", "P2 oracle alpha", f"Oracle {metric}"]
                           if period == "P2" else ["Model", "P1-selected alpha", f"P1 {metric}", f"P2 at fixed alpha"])
                table = table_ax.table(cellText=table_rows, colLabels=columns, cellLoc="center", loc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.45)
                for (row, _), cell in table.get_celld().items():
                    if row == 0:
                        cell.set_facecolor("#eeeeee")
                footer = ("Stars use frozen P1 choices. Hollow diamonds are P2 oracles; coincident markers share a setting."
                          if period == "P2" else "Selection uses P1 only. P2 values measure transfer using the same frozen alpha.")
                fig.supxlabel(footer, fontsize=9)
                filename = f"part43_alpha_{period.lower()}_{kind}.png"
                path = Path(output_dir) / filename
                if path.exists() and not overwrite:
                    raise FileExistsError(path)
                staged = path.with_name(f".{path.name}.in_progress")
                handle = staged.open("xb")
                try:
                    with handle:
                        fig.savefig(handle, format="png", dpi=180)
                    if overwrite:
                        staged.replace(path)
                    else:
                        path.hardlink_to(staged)
                finally:
                    staged.unlink(missing_ok=True)
                files.append(filename)
            finally:
                plt.close(fig)
    return files


def run_past_future_tuning(official_dir, results_dir, output_dir):
    """Freeze P1-only choices before evaluating P2; preserve all previous outputs."""
    official_dir, results_dir = Path(official_dir).resolve(), Path(results_dir).resolve()
    output_dir = external_path(output_dir)
    names = ["part43_p1_selected.json", "part43_tuning_scores.csv", "part43_per_query.csv",
             "part43_transfer_results.csv", "part43_summary.json"]
    names += [f"part43_alpha_{period}_{kind}.png" for period in ("p1", "p2") for kind in ("known_item", "topics")]
    if any((output_dir / name).exists() for name in names):
        raise FileExistsError("Part 4.3 output already exists; refusing to overwrite")
    started = perf_counter()
    catalogue, input_hashes = discover_tuning_runs(results_dir)
    past, past_queries, past_hashes = evaluate_tuning_period(catalogue, official_dir, results_dir, "P1")
    selected = select_p1_settings(past)
    if len(selected) != 12:
        raise ValueError("Expected separate P1 selections for two methods, three models, and two query types")
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = output_dir / "part43_p1_selected.json"
    write_json(freeze_path, {"selection_period": "P1", "selection_collection": "C1", "tie_rule": TUNING_TIE_RULE,
                            "selected": selected, "P1_input_sha256": past_hashes})
    frozen_hash = sha256_file(freeze_path)
    print("P1 parameters frozen to disk; only now starting P2 evaluation", flush=True)
    future, future_queries, future_hashes = evaluate_tuning_period(catalogue, official_dir, results_dir, "P2")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))["selected"]
    transfers = apply_p1_settings(frozen, future)
    if sha256_file(freeze_path) != frozen_hash:
        raise RuntimeError("Frozen P1 settings changed during future evaluation")
    rows = past + future
    per_query = past_queries + future_queries
    input_hashes.update(past_hashes)
    input_hashes.update(future_hashes)
    if any(sha256_file(path) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("An input changed during tuning")
    def serialize(record):
        return {key: json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, dict) else value
                for key, value in record.items()}
    _write_csv(output_dir / "part43_tuning_scores.csv",
               ["family", "model", "collection", "period", "parameters", "filename", "sha256", "query_type", "metric", "queries", "score"],
               [serialize(row) for row in rows])
    _write_csv(output_dir / "part43_per_query.csv",
               ["period", "collection", "family", "model", "query_type", "parameters", "qid", "metric", "value"],
               [serialize(row) for row in per_query])
    _write_csv(output_dir / "part43_transfer_results.csv", list(transfers[0]), [serialize(row) for row in transfers])
    plots = plot_alpha_curves(rows, output_dir, p1_selection=frozen, transfer_results=transfers)
    summary = {"part": "4.3", "all_complete": True, "retrieval_rerun": False, "indexes_rebuilt": False,
        "P2_influences_P1_selection": False, "P1_frozen_before_P2_evaluation": True,
        "selection_manifest_sha256": frozen_hash, "tie_rule": TUNING_TIE_RULE,
        "objectives": {kind: metric[1] for kind, metric in TUNING_OBJECTIVES.items()},
        "query_counts": TUNING_QUERY_COUNTS, "evaluated_run_files": len(catalogue), "tuning_score_rows": len(rows),
        "per_query_rows": len(per_query), "transfer_rows": len(transfers), "results": transfers,
        "oracle_role": "P2 oracle maximizes on future labels for analysis only; it never selects deployed parameters",
        "gap_definition": "P2 oracle effectiveness minus P2 effectiveness using the unchanged P1 setting",
        "MAP_definition": "Assignment AP@100 denominator min(Rq,100), consistent with validated Part 4.1",
        "plot_selection_policy": PLOT_SELECTION_POLICY, "part4_2_parameter_policy": PART42_PARAMETER_POLICY,
        "input_sha256": input_hashes, "inputs_unchanged": True, "plots": plots,
        "elapsed_seconds": perf_counter() - started,
        "files": names, "artifact_sha256": {name: sha256_file(output_dir / name) for name in names if name != "part43_summary.json"}}
    write_json(output_dir / "part43_summary.json", summary)
    return summary


def refresh_alpha_plots(output_dir):
    """Refresh only plots from saved effectiveness and immutable P1 selections."""
    output_dir = external_path(output_dir)
    summary_path = output_dir / "part43_summary.json"
    old = summary_path.read_bytes()
    summary = json.loads(old)
    if not summary["all_complete"]:
        raise ValueError("Plot refresh requires a completed Part 4.3 analysis")
    preserved = {name: digest for name, digest in summary["artifact_sha256"].items() if not name.endswith(".png")}
    if any(sha256_file(output_dir / name) != digest for name, digest in preserved.items()):
        raise ValueError("Saved Part 4.3 tables or selections changed")
    selection_path = output_dir / "part43_p1_selected.json"
    if sha256_file(selection_path) != summary["selection_manifest_sha256"]:
        raise ValueError("Frozen P1 selection checksum changed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
    with (output_dir / "part43_tuning_scores.csv").open(encoding="utf-8", newline="") as handle:
        rows = [{**row, "parameters": json.loads(row["parameters"]), "score": float(row["score"])}
                for row in csv.DictReader(handle)]
    plots = plot_alpha_curves(rows, output_dir, p1_selection=selection,
                              transfer_results=summary["results"], overwrite=True)
    if any(sha256_file(output_dir / name) != digest for name, digest in preserved.items()):
        raise RuntimeError("Plot refresh changed non-figure artifacts")
    summary["plots"] = plots
    summary["artifact_sha256"].update({name: sha256_file(output_dir / name) for name in plots})
    summary["plot_selection_policy"] = PLOT_SELECTION_POLICY
    summary["part4_2_parameter_policy"] = PART42_PARAMETER_POLICY
    publish_summary(summary_path, summary, old)
    return summary


# Appendix B uses a two-sided paired randomization test.  Its Monte Carlo
# p-value is the raw fraction of 10,000 sign-flip repetitions whose absolute
# mean is at least the observed absolute mean; there is no plus-one correction.
PART42_PERMUTATIONS = 10_000
PART42_RANDOM_SEED = 42
PART42_SIGNIFICANCE_LEVEL = .05
# Sign-flipped sums that are equal in exact arithmetic can differ by a few ULPs
# because summation order differs; such ties are "at least as extreme".  Distinct
# exact means are far apart (e.g. 1/(2520*1000) for MRR@10), so this cannot
# promote a genuinely smaller statistic.
PART42_TIE_TOLERANCE = 1e-12
PART42_METHOD_NAMES = (
    "TF-IDF", "BM25",
    "Dense M1", "Dense M2", "Dense M2s", "Dense GN",
    "QE M1", "QE M2", "QE GN",
    "Linear hybrid M1", "Linear hybrid M2", "Linear hybrid GN",
    "RRF M1", "RRF M2", "RRF GN",
)
PART42_OUTPUT_NAMES = (
    "part42_ki1_c1.csv", "part42_ki2_c2.csv",
    "part42_topics_c1.csv", "part42_topics_c2.csv",
    "part42_per_query.csv", "part42_randomization_tests.csv", "part42_summary.json",
)


def _part42_selected_filename(family, model, parameters, collection):
    """Map one frozen P1 setting to its already-saved C1 or C2 run."""
    key = parameter_key(family, parameters)
    prefix, model = collection.lower(), model.lower()
    if family == "QE":
        m, weight = key
        suffix = f"m{m}_l{weight:.1f}".replace(".", "p")
        return f"{prefix}_qe_{model}_{suffix}.csv"
    alpha, = key
    suffix = f"a{alpha:.1f}".replace(".", "p")
    return f"{prefix}_hybrid_{model}_{suffix}.csv"


def load_frozen_p1_selection(path):
    """Validate the immutable Part 4.3 manifest used by all Part 4.2 tables."""
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selection_period") != "P1" or payload.get("selection_collection") != "C1":
        raise ValueError("Part 4.2 requires a frozen P1/C1 selection manifest")
    rows = payload.get("selected")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("Frozen selection must contain 12 QE/hybrid choices")
    expected = {(family, model, kind) for family in ("QE", "HYBRID")
                for model in TUNING_MODELS for kind in TUNING_QUERY_COUNTS}
    actual = set()
    for row in rows:
        key = row.get("family"), row.get("model"), row.get("query_type")
        if key not in expected or key in actual:
            raise ValueError("Frozen selection has an unexpected or duplicate cohort")
        family, model, kind = key
        if (row.get("period") != "P1" or row.get("collection") != "C1"
                or row.get("metric") != TUNING_OBJECTIVES[kind][1]
                or row.get("queries") != TUNING_QUERY_COUNTS[kind]):
            raise ValueError("Frozen selection contains non-P1 data or the wrong objective/query count")
        parameters = row.get("parameters")
        parameter_key(family, parameters)
        expected_file = _part42_selected_filename(family, model, parameters, "C1")
        if row.get("filename") != expected_file:
            raise ValueError("Frozen selection filename does not encode its saved parameters")
        actual.add(key)
    if actual != expected:
        raise ValueError("Frozen selection is missing a required family/model/query-type cohort")
    return payload


def part42_method_plan(frozen_selection, collection, query_type):
    """Return the 15 saved runs, applying each query type's P1 choices unchanged."""
    if collection not in {"C1", "C2"} or query_type not in TUNING_QUERY_COUNTS:
        raise ValueError("Part 4.2 method plan needs C1/C2 and known_item/topics")
    chosen = {(row["family"], row["model"], row["query_type"]): row
              for row in frozen_selection["selected"]}
    prefix = collection.lower()
    plan = [
        {"method": "TF-IDF", "family": "LEXICAL", "model": "TF-IDF",
         "parameters": {"sublinear_tf": True, "norm": "l2", "smooth_idf": True},
         "filename": f"{prefix}_tfidf.csv"},
        {"method": "BM25", "family": "LEXICAL", "model": "BM25",
         "parameters": {"k1": 1.2, "b": .75}, "filename": f"{prefix}_bm25.csv"},
    ]
    for model in ("M1", "M2", "M2s", "GN"):
        plan.append({"method": f"Dense {model}", "family": "DENSE", "model": model,
                     "parameters": {}, "filename": f"{prefix}_dense_{model.lower()}.csv"})
    for family, label in (("QE", "QE"), ("HYBRID", "Linear hybrid")):
        for model in TUNING_MODELS:
            row = chosen[family, model, query_type]
            parameters = dict(row["parameters"])
            plan.append({"method": f"{label} {model}", "family": family, "model": model,
                         "parameters": parameters,
                         "filename": _part42_selected_filename(family, model, parameters, collection)})
    for model in TUNING_MODELS:
        plan.append({"method": f"RRF {model}", "family": "RRF", "model": model,
                     "parameters": {"k": 60}, "filename": f"{prefix}_rrf_{model.lower()}.csv"})
    if tuple(row["method"] for row in plan) != PART42_METHOD_NAMES:
        raise AssertionError("Part 4.2 method order is incomplete")
    return plan


def paired_randomization_against_bm25(score_vectors, *, permutations=PART42_PERMUTATIONS,
                                      seed=PART42_RANDOM_SEED, batch_size=1000,
                                      tie_tolerance=PART42_TIE_TOLERANCE, baseline="BM25"):
    """Appendix B two-sided paired sign-flip tests for several methods at once.

    Every column is paired to ``baseline`` (BM25 unless another method is named)
    in the supplied official-query order.  Sign draws do not depend on how many
    methods are compared, so each p-value equals its own single-pair test.  The
    random generator is restarted from the documented seed for each test set.
    The Appendix B p-value is ``extreme repetitions / permutations`` exactly.
    """
    if not math.isfinite(tie_tolerance) or not 0 <= tie_tolerance < 1e-9:
        raise ValueError("Tie tolerance must only absorb floating-point rounding")
    if (isinstance(permutations, bool) or not isinstance(permutations, int) or permutations <= 0
            or isinstance(seed, bool) or not isinstance(seed, int)
            or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0):
        raise ValueError("Randomization count, seed and batch size must be integers with positive counts")
    if baseline not in score_vectors or len(score_vectors) < 2:
        raise ValueError("Randomization requires the baseline and at least one comparison method")
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in score_vectors.items()}
    sizes = {values.shape for values in arrays.values()}
    if len(sizes) != 1 or next(iter(sizes))[0] == 0 or len(next(iter(sizes))) != 1:
        raise ValueError("Paired score vectors must be nonempty one-dimensional arrays of equal length")
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Paired scores must be finite")
    methods = [name for name in score_vectors if name != baseline]
    baseline_name, baseline = baseline, arrays[baseline]
    differences = np.column_stack([arrays[name] - baseline for name in methods])
    observed = differences.mean(axis=0)
    thresholds = np.abs(observed)
    extreme = np.zeros(len(methods), dtype=np.int64)
    rng = np.random.default_rng(seed)
    remaining = permutations
    while remaining:
        current = min(batch_size, remaining)
        signs = rng.integers(0, 2, size=(current, len(baseline)), dtype=np.int8) * 2 - 1
        randomized = np.abs((signs @ differences) / len(baseline))
        extreme += np.count_nonzero(randomized >= thresholds - tie_tolerance, axis=0)
        remaining -= current
    results = []
    for index, method in enumerate(methods):
        p_value = int(extreme[index]) / permutations
        significant = p_value < PART42_SIGNIFICANCE_LEVEL
        direction = ("significant_improvement" if significant and observed[index] > 0 else
                     "significant_decline" if significant and observed[index] < 0 else
                     "not_significant")
        results.append({"method": method, "baseline": baseline_name, "queries": len(baseline),
                        "method_mean": float(arrays[method].mean()),
                        "bm25_mean" if baseline_name == "BM25" else "baseline_mean": float(baseline.mean()),
                        "mean_difference": float(observed[index]),
                        "observed_absolute_mean_difference": float(thresholds[index]),
                        "permutations": permutations, "random_seed": seed,
                        "tie_tolerance": tie_tolerance,
                        "extreme_permutations": int(extreme[index]), "p_value": p_value,
                        "significance_level": PART42_SIGNIFICANCE_LEVEL,
                        "significant": significant, "direction": direction,
                        "alternative": "two-sided"})
    return results


def verify_frozen_reuse(frozen_selection, transfer_results, collection, query_type, tuned_rows):
    """Prove each QE/hybrid row is the unchanged P1 choice, never a P2 oracle.

    ``tuned_rows`` holds dicts with method/family/model/parameters (dict),
    source_file, sha256 and primary score.  C1 rows must reproduce the frozen
    Part 4.3 file, checksum and P1 score; C2 rows must reproduce the Part 4.3
    transferred (not oracle) file and score.
    """
    chosen = {(row["family"], row["model"], row["query_type"]): row for row in frozen_selection["selected"]}
    transfers = {(row["family"], row["model"], row["query_type"]): row for row in transfer_results}
    checks = []
    for row in tuned_rows:
        key = row["family"], row["model"], query_type
        frozen = chosen[key]
        if row["parameters"] != frozen["parameters"]:
            raise ValueError(f"{row['method']} does not use its frozen P1 parameters")
        if collection == "C1":
            expected_file, expected_score = frozen["filename"], frozen["score"]
            if row["sha256"] != frozen["sha256"]:
                raise ValueError(f"{row['method']} C1 run differs from the frozen P1 run")
        else:
            transfer = transfers.get(key)
            if transfer is None or transfer["p1_selected_parameters"] != frozen["parameters"]:
                raise ValueError(f"{row['method']} has no Part 4.3 transfer record for its frozen setting")
            expected_file, expected_score = transfer["p2_transferred_filename"], transfer["p2_transferred_score"]
        if row["source_file"] != expected_file or row["score"] != expected_score:
            raise ValueError(f"{row['method']} {collection} result differs from Part 4.3 frozen reuse")
        checks.append({"method": row["method"], "collection": collection, "query_type": query_type,
                       "parameters": frozen["parameters"], "source_file": row["source_file"],
                       "primary_score": row["score"], "matches_part43": True})
    return checks


def paired_randomization_test(method_scores, bm25_scores, **kwargs):
    """Convenience wrapper for one Appendix B comparison."""
    return paired_randomization_against_bm25(
        {"BM25": bm25_scores, "method": method_scores}, **kwargs)[0]


def run_main_results(official_dir, results_dir, output_dir, selection_path=None):
    """Evaluate the 15 saved methods for Part 4.2 without retrieval or retuning."""
    official_dir, results_dir = Path(official_dir).resolve(), Path(results_dir).resolve()
    output_dir = external_path(output_dir)
    selection_path = Path(selection_path or output_dir / "part43_p1_selected.json").resolve()
    targets = [output_dir / name for name in PART42_OUTPUT_NAMES]
    if any(path.exists() for path in targets):
        raise FileExistsError("Part 4.2 output already exists; refusing to overwrite")
    frozen = load_frozen_p1_selection(selection_path)
    selection_hash = sha256_file(selection_path)
    part43_path = selection_path.with_name("part43_summary.json")
    part43 = json.loads(part43_path.read_text(encoding="utf-8"))
    if part43.get("selection_manifest_sha256") != selection_hash or not part43.get("all_complete"):
        raise ValueError("Part 4.3 summary does not belong to the frozen P1 selection")
    transfer_results = part43["results"]
    protected = ({path: sha256_file(path) for path in output_dir.iterdir() if path.is_file()}
                 if output_dir.exists() else {})
    started = perf_counter()
    input_hashes = {selection_path: selection_hash, part43_path: sha256_file(part43_path)}
    per_query_rows, randomization_rows, table_rows = [], [], {}
    test_sets, run_validations, reuse_checks = [], [], []

    topics_path = official_dir / "topics.tsv"
    topics = read_table(topics_path, ["tid", "query"], "tid")
    if len(topics) != TUNING_QUERY_COUNTS["topics"]:
        raise ValueError("Unexpected official topic-query count")
    input_hashes[topics_path] = sha256_file(topics_path)
    table_filename = {"KI1/C1": "part42_ki1_c1.csv", "KI2/C2": "part42_ki2_c2.csv",
                      "Topics/C1": "part42_topics_c1.csv", "Topics/C2": "part42_topics_c2.csv"}

    for collection, period, known_label in (("C1", "P1", "KI1"), ("C2", "P2", "KI2")):
        collection_path = official_dir / f"collection_{period}.tsv"
        known_path = official_dir / f"queries_ki_{period}.tsv"
        official = read_table(collection_path, ["docno", "web_url"], "docno")
        known = read_table(known_path, ["qid", "query", "docno"], "qid")
        if (len(official) != EXPECTED_COLLECTION_COUNTS[collection]
                or len(known) != TUNING_QUERY_COUNTS["known_item"]):
            raise ValueError("Unexpected official collection or known-item query count")
        all_qids = known["qid"].tolist() + topics["tid"].tolist()
        if len(set(all_qids)) != sum(TUNING_QUERY_COUNTS.values()):
            raise ValueError("Official known-item and topic query IDs must be complete and disjoint")
        input_hashes[collection_path] = sha256_file(collection_path)
        input_hashes[known_path] = sha256_file(known_path)
        qrel_paths = {"known_item": official_dir / f"qrels_ki_{period}.txt",
                      "topics": official_dir / f"qrels_topics_{period}.txt"}
        qids_by_kind = {"known_item": known["qid"].tolist(), "topics": topics["tid"].tolist()}
        qrels_by_kind = {}
        for kind, path in qrel_paths.items():
            input_hashes[path] = sha256_file(path)
            qrels_by_kind[kind] = load_qrels(path, qids_by_kind[kind], official["docno"])
            if any(not qrels_by_kind[kind][qid] for qid in qids_by_kind[kind]):
                raise ValueError("An official Part 4.2 query lacks relevance judgments")
        for qid, target in known[["qid", "docno"]].itertuples(index=False, name=None):
            relevant = {docno for docno, grade in qrels_by_kind["known_item"][qid].items() if grade >= 1}
            if relevant != {target}:
                raise ValueError("Known-item target and official qrels disagree")

        for kind in ("known_item", "topics"):
            test_set = f"{known_label}/{collection}" if kind == "known_item" else f"Topics/{collection}"
            qids, qrels = qids_by_kind[kind], qrels_by_kind[kind]
            primary_internal, primary_metric = TUNING_OBJECTIVES[kind]
            plan = part42_method_plan(frozen, collection, kind)
            scores, rows, tuned = {}, [], []
            for spec in plan:
                run_path = results_dir / spec["filename"]
                if run_path not in input_hashes:
                    input_hashes[run_path] = sha256_file(run_path)
                rankings, details = load_run(run_path, all_qids, official["docno"])
                lengths = [len(rankings[qid]) for qid in all_qids]
                if (details["rows"] != 100 * len(all_qids) or details["missing_queries"]
                        or details["queries_with_fewer_than_100_results"] or set(lengths) != {100}):
                    raise ValueError(f"Incomplete official query set or non-top-100 run: {run_path.name}")
                evaluated = evaluate_queries(qids, rankings, qrels, kind)
                if evaluated["queries"] != TUNING_QUERY_COUNTS[kind]:
                    raise AssertionError("Evaluation did not preserve the complete official query set")
                score_vector = np.asarray([row[primary_internal] for row in evaluated["per_query"]], dtype=np.float64)
                scores[spec["method"]] = score_vector
                if spec["family"] in {"QE", "HYBRID"}:
                    tuned.append({**spec, "source_file": spec["filename"], "sha256": input_hashes[run_path],
                                  "score": evaluated["means"][primary_metric]})
                parameters = json.dumps(spec["parameters"], sort_keys=True, separators=(",", ":"))
                table = {"method": spec["method"], "family": spec["family"], "model": spec["model"],
                         "parameters": parameters, "source_file": spec["filename"], "queries": len(qids),
                         **evaluated["means"]}
                rows.append(table)
                metric_names = KNOWN_METRICS if kind == "known_item" else TOPIC_METRICS
                for query_row in evaluated["per_query"]:
                    for metric in metric_names:
                        per_query_rows.append({"test_set": test_set, "collection": collection, "period": period,
                                               "query_type": kind, "method": spec["method"],
                                               "parameters": parameters, "source_file": spec["filename"],
                                               "qid": query_row["qid"],
                                               "relevant_total": query_row["relevant_total"],
                                               "metric": metric, "value": query_row[metric]})
                validation = {"test_set": test_set, "method": spec["method"], "source_file": spec["filename"],
                              "official_queries_in_combined_run": len(all_qids),
                              "evaluated_queries": len(qids), "rows": details["rows"],
                              "results_per_official_query": 100, "missing_queries": 0,
                              "invalid_ids_ranks_scores_or_duplicates": 0, "passed": True}
                run_validations.append(validation)

            reuse_checks.extend(verify_frozen_reuse(frozen, transfer_results, collection, kind, tuned))
            comparisons = paired_randomization_against_bm25(scores)
            by_method = {row["method"]: row for row in comparisons}
            paired_hash = hashlib.sha256("\n".join(qids).encode("utf-8")).hexdigest()
            for comparison in comparisons:
                record = {"test_set": test_set, "collection": collection, "period": period,
                          "query_type": kind, "primary_metric": primary_metric,
                          "paired_qid_sha256": paired_hash, **comparison}
                randomization_rows.append(record)
            for row in rows:
                if row["method"] == "BM25":
                    row.update(primary_delta_vs_BM25=0.0, p_value_vs_BM25="",
                               significance_vs_BM25="reference", marker="BM25")
                else:
                    comparison = by_method[row["method"]]
                    marker = ("SIG+" if comparison["direction"] == "significant_improvement" else
                              "SIG-" if comparison["direction"] == "significant_decline" else "")
                    row.update(primary_delta_vs_BM25=comparison["mean_difference"],
                               p_value_vs_BM25=comparison["p_value"],
                               significance_vs_BM25=comparison["direction"], marker=marker)
            table_rows[test_set] = rows
            test_sets.append({"test_set": test_set, "collection": collection, "period": period,
                              "query_type": kind, "queries": len(qids), "primary_metric": primary_metric,
                              "table_file": table_filename[test_set], "methods": len(rows),
                              "paired_qid_sha256": paired_hash,
                              "frozen_tuned_runs": [{"method": row["method"], "parameters": row["parameters"],
                                                     "source_file": row["source_file"]}
                                                    for row in rows if row["family"] in {"QE", "HYBRID"}]})

    if sha256_file(selection_path) != selection_hash:
        raise RuntimeError("Frozen Part 4.3 selections changed during Part 4.2 evaluation")
    if any(sha256_file(path) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("A Part 4.2 input changed during evaluation")
    if any(sha256_file(path) != digest for path, digest in protected.items()):
        raise RuntimeError("A previous Part 4 output changed before publication")
    output_dir.mkdir(parents=True, exist_ok=True)
    for test_set in ("KI1/C1", "KI2/C2", "Topics/C1", "Topics/C2"):
        rows = table_rows[test_set]
        _write_csv(output_dir / table_filename[test_set], list(rows[0]), rows)
    _write_csv(output_dir / "part42_per_query.csv", list(per_query_rows[0]), per_query_rows)
    _write_csv(output_dir / "part42_randomization_tests.csv", list(randomization_rows[0]), randomization_rows)
    if any(sha256_file(path) != digest for path, digest in protected.items()):
        raise RuntimeError("Publishing Part 4.2 changed a previous Part 4 output")

    improvements = [{key: row[key] for key in ("test_set", "method", "primary_metric", "mean_difference", "p_value")}
                    for row in randomization_rows if row["direction"] == "significant_improvement"]
    declines = [{key: row[key] for key in ("test_set", "method", "primary_metric", "mean_difference", "p_value")}
                for row in randomization_rows if row["direction"] == "significant_decline"]
    artifact_names = list(PART42_OUTPUT_NAMES[:-1])
    summary = {
        "part": "4.2", "all_complete": True, "retrieval_rerun": False, "parameters_retuned": False,
        "frozen_selection_file": str(selection_path), "frozen_selection_sha256": selection_hash,
        "parameter_policy": PART42_PARAMETER_POLICY, "methods": list(PART42_METHOD_NAMES),
        "test_sets": test_sets, "tables": [table_filename[name] for name in
                                              ("KI1/C1", "KI2/C2", "Topics/C1", "Topics/C2")],
        "per_query_rows": len(per_query_rows), "randomization_comparisons": len(randomization_rows),
        "appendix_b_randomization": {
            "procedure": "For paired per-query differences d_i=method_i-BM25_i, compare |mean(d)| to 10,000 repetitions that independently flip each d_i sign with probability 1/2",
            "permutations": PART42_PERMUTATIONS, "random_seed": PART42_RANDOM_SEED,
            "random_generator": "numpy.random.default_rng; restarted with seed 42 for each test set",
            "floating_point_tie_tolerance": PART42_TIE_TOLERANCE,
            "alternative": "two-sided", "p_value": "count(|randomized mean| >= |observed mean|) / 10000; no plus-one correction",
            "significance_rule": "p < 0.05", "primary_metrics": {"known_item": "MRR@10", "topics": "MAP@100"},
            "pairing": "Exact official query-ID order; no aggregation or unpaired observations"},
        "significant_improvements_vs_BM25": improvements,
        "significant_declines_vs_BM25": declines,
        "frozen_parameter_reuse": {"passed": len(reuse_checks) == 24, "checks": reuse_checks,
                                   "rule": "C1 rows reproduce part43_p1_selected.json file/checksum/score; C2 rows reproduce the Part 4.3 transferred (not oracle) file/score"},
        "run_validation": {"passed": all(row["passed"] for row in run_validations),
                           "checks": run_validations,
                           "rule": "Every source run contains each of the 1,032 official collection query IDs exactly once with ranks 1..100, finite descending scores, official unique documents and no duplicate query/document pair"},
        "input_sha256": {str(path): digest for path, digest in input_hashes.items()},
        "inputs_unchanged": True, "previous_part4_files_unchanged": len(protected),
        "elapsed_seconds": perf_counter() - started,
        "files": list(PART42_OUTPUT_NAMES),
        "artifact_sha256": {name: sha256_file(output_dir / name) for name in artifact_names},
    }
    write_json(output_dir / "part42_summary.json", summary)
    return summary


PART44_DENSE = ("Dense M1", "Dense M2", "Dense M2s", "Dense GN")
PART44_QE = ("QE M1", "QE M2", "QE GN")
PART44_METHODS = ("BM25",) + PART44_DENSE + PART44_QE
PART44_YEAR_METHODS = ("BM25", "Dense M1", "Dense M2", "Dense GN")
PART44_TEST_SETS = {("C1", "known_item"): "KI1/C1", ("C2", "known_item"): "KI2/C2",
                    ("C1", "topics"): "Topics/C1", ("C2", "topics"): "Topics/C2"}
PART44_MODELS = {
    "M1": {"model_file": "w2v_P1.model", "corpus": "corpus_P1.txt", "source_model": None,
           "training_periods": "P1", "training": "Word2Vec from scratch on P1 (2005-2014)"},
    "M2": {"model_file": "w2v_P2_continued.model", "corpus": "corpus_P2.txt", "source_model": "w2v_P1.model",
           "training_periods": "P1+P2", "training": "M1 continued on P2 (2015-2025); P2 is its most recent data"},
    "M2s": {"model_file": "w2v_P2_fresh.model", "corpus": "corpus_P2.txt", "source_model": None,
            "training_periods": "P2", "training": "Word2Vec from scratch on P2 (2015-2025)"},
    "GN": {"model_file": "gn_subset.kv", "corpus": None, "source_model": None,
           "training_periods": "external", "training": "Pretrained Google News vectors; neither P1 nor P2"},
}
# (method, baseline): positive mean difference means ``method`` is better on the same queries.
PART44_PAIRS = (("Dense M2", "Dense M1"), ("Dense M2s", "Dense M1"), ("Dense M2", "Dense M2s"),
                ("Dense GN", "Dense M1"), ("Dense GN", "Dense M2"), ("Dense GN", "Dense M2s"),
                ("QE M2", "QE M1"), ("QE GN", "QE M1"), ("QE GN", "QE M2"))
PART44_OUTPUT_NAMES = ("part44_temporal_comparison.csv", "part44_paired_tests.csv", "part44_ki2_by_year.csv",
                       "part44_ki2_per_query_year.csv", "part44_ki2_mrr_by_year.png", "part44_summary.json")


def verify_model_provenance(results_dir, models_dir, collections=("C1", "C2")):
    """Check every dense/QE run against the embedding file and training corpus it claims.

    Local models: the training report must name the expected corpus and source
    model, and each run's recorded model checksums must equal the files on disk.
    GN: every run must record the same external GN vector checksum.
    """
    results_dir, models_dir = Path(results_dir), Path(models_dir)
    disk = {}
    def on_disk(name):
        if name not in disk:
            disk[name] = sha256_file(models_dir / name)
        return disk[name]
    training = {}
    for model, spec in PART44_MODELS.items():
        if spec["corpus"] is None:
            continue
        report = json.loads((models_dir / f"{spec['model_file']}.training.json").read_text(encoding="utf-8"))
        source = report.get("source_model_path")
        if (Path(report["model_path"]).name != spec["model_file"] or Path(report["corpus_path"]).name != spec["corpus"]
                or (Path(source).name if source else None) != spec["source_model"]):
            raise ValueError(f"{model} training report does not match its expected period/initialization")
        training[model] = {"model_file": spec["model_file"], "corpus": spec["corpus"],
                           "source_model": spec["source_model"], "training_periods": spec["training_periods"]}
    records, gn_hashes = [], set()
    for collection in collections:
        prefix = collection.lower()
        for family, model in [("Dense", m) for m in PART44_MODELS] + [("QE", m) for m in ("M1", "M2", "GN")]:
            if family == "Dense":
                path = results_dir / f"dense_{prefix}_{model.lower()}" / "dense_summary.json"
                summary = json.loads(path.read_text(encoding="utf-8"))
                provenance = summary["document_embeddings"]["provenance"]
                recorded = provenance["model_file_sha256"]
                run_name = f"{prefix}_dense_{model.lower()}.csv"
                output = summary["combined_output"]
                if Path(output["path"]).name != run_name or output["sha256"] != sha256_file(results_dir / run_name):
                    raise ValueError(f"Dense run file does not match its provenance summary: {run_name}")
            else:
                path = results_dir / f"qe_{prefix}_{model.lower()}" / "expansion_summary.json"
                summary = json.loads(path.read_text(encoding="utf-8"))
                provenance = summary["provenance"]
                recorded = provenance["model_files_sha256"]
            if (summary["model"] != model or provenance["model"] != model
                    or provenance["collection"] != collection):
                raise ValueError(f"Run summary model/collection mismatch: {path}")
            expected = PART44_MODELS[model]["model_file"]
            if expected not in recorded:
                raise ValueError(f"{family} {model}/{collection} did not use {expected}")
            if model == "GN":
                gn_hashes.add(recorded[expected])
                verified = "consistent external GN checksum across runs"
            else:
                report = f"{expected}.training.json"
                if recorded[expected] != on_disk(expected) or recorded.get(report) != on_disk(report):
                    raise ValueError(f"{family} {model}/{collection} used a different model file than {expected}")
                verified = "model and training-report checksums match files on disk"
            records.append({"method": f"{family} {model}", "collection": collection, "model_file": expected,
                            "model_sha256": recorded[expected], "training_periods": PART44_MODELS[model]["training_periods"],
                            "verification": verified})
    if len(gn_hashes) != 1:
        raise ValueError("Google News runs do not share one embedding file")
    return {"passed": True, "training_reports": training, "runs": records}


def load_target_years(known, official, period_years):
    """Official target publication year per known-item query, checked three ways.

    The year comes from the official query file's ``year`` column (the TARGET
    article's year); it must equal the collection's year for the target docno
    and lie inside the period.  Retrieved documents are never consulted.
    """
    if "year" not in known.columns:
        raise ValueError("Official known-item queries need the target publication year")
    collection_year = dict(zip(official["docno"], official["year"])) if "year" in official.columns else {}
    years = {}
    for qid, docno, year in known[["qid", "docno", "year"]].itertuples(index=False, name=None):
        year = int(year)
        if collection_year and int(collection_year[docno]) != year:
            raise ValueError(f"Target year in query file disagrees with the collection: {qid}")
        if year not in period_years:
            raise ValueError(f"Target year outside its period: {qid}")
        years[qid] = year
    return years


def group_by_publication_year(qids, target_years, values):
    """Mean per-query values by target year; every query is included exactly once."""
    qids = list(qids)
    if not qids or len(set(qids)) != len(qids):
        raise ValueError("Year grouping needs nonempty, unique query IDs")
    if any(qid not in target_years for qid in qids):
        raise ValueError("Every query needs an official target year")
    if any(len(column) != len(qids) for column in values.values()):
        raise ValueError("Per-query values must align with the query IDs")
    groups = {}
    for index, qid in enumerate(qids):
        groups.setdefault(target_years[qid], []).append(index)
    rows = []
    for year in sorted(groups):
        indices = groups[year]
        rows.append({"year": year, "queries": len(indices),
                     **{name: math.fsum(column[i] for i in indices) / len(indices) for name, column in values.items()}})
    if sum(row["queries"] for row in rows) != len(qids):
        raise AssertionError("Year groups must partition all queries")
    for name, column in values.items():
        weighted = math.fsum(row[name] * row["queries"] for row in rows) / len(qids)
        if abs(weighted - math.fsum(column) / len(qids)) > 1e-12:
            raise AssertionError("Year means do not recombine to the overall mean")
    return rows


def weighted_slope(xs, ys, weights):
    """Weighted least-squares slope, used only as a descriptive trend (None for one year)."""
    total = math.fsum(weights)
    mx = math.fsum(w * x for x, w in zip(xs, weights)) / total
    my = math.fsum(w * y for y, w in zip(ys, weights)) / total
    denominator = math.fsum(w * (x - mx) ** 2 for x, w in zip(xs, weights))
    if denominator == 0:
        return None
    return math.fsum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, weights)) / denominator


def plot_ki2_year_curves(year_rows, overall, path):
    """KI2/C2 MRR@10 by target year with the per-year query counts beneath."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#2b2b29", "#6b6a64", "#e4e3dd", "#fcfcfb"
    styles = {"BM25": ("#52514e", "s", (0, (5, 3))), "Dense M1": ("#2a78d6", "o", "-"),
              "Dense M2": ("#eb6834", "^", "-"), "Dense GN": ("#1baf7a", "D", "-")}
    years = [row["year"] for row in year_rows]
    counts = [row["queries"] for row in year_rows]
    fig, (ax, bars) = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True, layout="constrained",
                                   gridspec_kw={"height_ratios": [4, 1.15]}, facecolor=surface)
    try:
        ends = []
        for method in PART44_YEAR_METHODS:
            color, marker, style = styles[method]
            values = [row[method] for row in year_rows]
            ax.plot(years, values, color=color, marker=marker, linestyle=style, linewidth=2, markersize=6.5,
                    markeredgecolor=surface, markeredgewidth=1, label=f"{method}  (overall {overall[method]:.3f})",
                    zorder=3)
            ends.append([values[-1], values[-1], method])
        ends.sort()
        for i in range(1, len(ends)):  # Keep direct labels apart without moving the data.
            ends[i][1] = max(ends[i][1], ends[i - 1][1] + .05)
        for actual, placed, method in ends:
            ax.annotate(method, xy=(years[-1], actual), xytext=(years[-1] + .35, placed), color=ink, fontsize=9,
                        va="center", arrowprops={"arrowstyle": "-", "color": muted, "linewidth": .6})
        ax.set(xlim=(years[0] - .5, years[-1] + 1.7), ylim=(0, 1.05), ylabel="MRR@10 (mean RR@10 per year)",
               facecolor=surface)
        ax.set_title("KI2/C2 known-item queries: MRR@10 by publication year of the target article",
                     loc="left", color=ink, fontsize=12)
        ax.grid(axis="y", color=grid, linewidth=.8)
        ax.legend(loc="upper left", ncols=2, fontsize=9, frameon=False, labelcolor=ink)
        bars.bar(years, counts, width=.6, color="#a9a8a0", zorder=2)
        for year, count in zip(years, counts):
            bars.text(year, count + max(counts) * .06, f"n={count}", ha="center", va="bottom", fontsize=8.5, color=ink)
        bars.set(ylim=(0, max(counts) * 1.45), ylabel="Queries", facecolor=surface)
        bars.set_xticks(years)
        bars.set_xlabel("Target-article publication year", color=ink)
        for axis in (ax, bars):
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(colors=muted, labelcolor=ink)
        fig.supxlabel("All 1,000 KI2 queries; RR@10 = 0 when the target is outside the top 10.\n"
                      "Year = publication year of each query's official target article (queries_ki_P2.tsv), "
                      f"not of retrieved documents. Smallest group: {min(counts)} queries.",
                      fontsize=8, color=muted)
        with Path(path).open("xb") as handle:
            fig.savefig(handle, format="png", dpi=170, facecolor=surface)
    finally:
        plt.close(fig)


def _describe(test):
    word = "significant" if test["significant"] else "not significant"
    return (f"{test['test_set']}: {test['method']} - {test['baseline']} = {test['mean_difference']:+.4f} "
            f"{test['primary_metric']} (p={test['p_value']:.4f}, {word})")


def part44_findings(comparison, tests, year_analysis):
    """Turn measured values into observations, then label interpretations separately."""
    test = {(row["test_set"], row["method"], row["baseline"]): row for row in tests}
    score = {(row["query_type"], row["method"]): row for row in comparison}
    p1_sets, p2_sets = ("KI1/C1", "Topics/C1"), ("KI2/C2", "Topics/C2")
    findings = {}

    own = [test[s, "Dense M2s", "Dense M1"] for s in p1_sets + p2_sets]
    own += [test[s, "Dense M2", "Dense M1"] for s in p2_sets] + [test[s, "QE M2", "QE M1"] for s in p2_sets]
    relative = []
    for kind in ("known_item", "topics"):
        bm25 = score[kind, "BM25"]
        moves = {m: score[kind, f"Dense {m}"]["minus_BM25_P2"] - score[kind, f"Dense {m}"]["minus_BM25_P1"]
                 for m in ("M1", "M2", "M2s", "GN")}
        p2_trained_gain = min(moves["M2"], moves["M2s"]) > max(moves["M1"], moves["GN"])
        relative.append(
            f"{bm25['metric']}: BM25 itself moves {bm25['change_P2_minus_P1']:+.4f} from P1 to P2, so raw changes "
            "largely reflect the test sets. Change in (dense - BM25) from P1 to P2: "
            + ", ".join(f"{m} {v:+.4f}" for m, v in moves.items()) + ". "
            + ("P2-trained models (M2, M2s) improve their position relative to M1 and GN, consistent with an "
               "own-period effect." if p2_trained_gain else
               "P2-trained models do not uniformly improve their position relative to M1 and GN."))
    shifts = []
    for kind in ("known_item", "topics"):
        for method in PART44_DENSE + PART44_QE:
            row = score[kind, method]
            shifts.append(f"{row['metric']} {method}: P1 {row['P1_C1']:.4f} -> P2 {row['P2_C2']:.4f} "
                          f"({row['change_P2_minus_P1']:+.4f}); minus BM25 {row['minus_BM25_P1']:+.4f} -> "
                          f"{row['minus_BM25_P2']:+.4f}")
    own = sorted(own, key=lambda r: (r["test_set"], r["method"]))
    own_wins = {s: test[s, "Dense M2s", "Dense M1"]["mean_difference"] for s in p1_sets + p2_sets}
    p1_own = [s for s in p1_sets if own_wins[s] < 0]
    p2_own = [s for s in p2_sets if own_wins[s] > 0]
    findings["own_training_period"] = {
        "observations": [_describe(row) for row in own] + shifts,
        "own_period_model_better_on_same_queries": {s: (own_wins[s] < 0 if s in p1_sets else own_wins[s] > 0)
                                                     for s in own_wins},
        "interpretations": [
            "Same-query comparison (M1=P1-only vs M2s=P2-only) is the cleanest test; the raw P1->P2 change mixes "
            "model and test-set difficulty, which BM25's own change indicates.",
            *relative,
            (f"The own-period model wins on {len(p1_own) + len(p2_own)} of 4 test sets "
             f"({', '.join(p1_own + p2_own) or 'none'}); significant wins: "
             f"{', '.join(r['test_set'] for r in own if r['method'] == 'Dense M2s' and r['significant'] and r['test_set'] in p1_own + p2_own) or 'none'}. "
             + ("This supports an own-period advantage." if len(p1_own) + len(p2_own) == 4 else
                "The own-period advantage is therefore not uniform across query types."))]}

    forgetting = [test[s, "Dense M2", "Dense M1"] for s in p1_sets] + [test[s, "QE M2", "QE M1"] for s in p1_sets]
    lost = [r for r in forgetting if r["mean_difference"] < 0]
    findings["continued_training_on_P1"] = {
        "observations": [_describe(row) for row in forgetting],
        "interpretations": [
            ("M2 scores below M1 on P1 in " + ", ".join(f"{r['method'].split()[0]} {r['test_set']}" for r in lost)
             + ("; significant: " + ", ".join(f"{r['method'].split()[0]} {r['test_set']}" for r in lost if r["significant"])
                if any(r["significant"] for r in lost) else "; none of these losses is significant")
             if lost else "M2 does not score below M1 on any P1 test set; no measurable forgetting."),
            "Losses are evidence of forgetting only where significant; topic tests have 32 queries and low power."]}

    m2_vs_m2s = [test[s, "Dense M2", "Dense M2s"] for s in p1_sets + p2_sets]
    findings["M2_vs_M2s"] = {
        "observations": [_describe(row) for row in m2_vs_m2s],
        "interpretations": [
            f"M2 is ahead of M2s on {sum(r['mean_difference'] > 0 for r in m2_vs_m2s)} of 4 test sets "
            f"({sum(r['significant'] and r['mean_difference'] > 0 for r in m2_vs_m2s)} significant), behind on "
            f"{sum(r['mean_difference'] < 0 for r in m2_vs_m2s)} "
            f"({sum(r['significant'] and r['mean_difference'] < 0 for r in m2_vs_m2s)} significant).",
            "An M2 lead on P1 is consistent with retained P1 knowledge; an M2 lead on P2 would suggest P1 "
            "initialization also helps on future data (more training text overall)."]}

    gn = [test[s, "Dense GN", local] for s in p1_sets + p2_sets for local in ("Dense M1", "Dense M2", "Dense M2s")]
    gn += [test[s, "QE GN", local] for s in p1_sets + p2_sets for local in ("QE M1", "QE M2")]
    def tally(rows):
        return (f"better in {sum(r['mean_difference'] > 0 for r in rows)}/{len(rows)} "
                f"({sum(r['significant'] and r['mean_difference'] > 0 for r in rows)} significant), worse in "
                f"{sum(r['mean_difference'] < 0 for r in rows)} ({sum(r['significant'] and r['mean_difference'] < 0 for r in rows)} significant)")
    by_type = {kind: [r for r in gn if r["query_type"] == kind] for kind in ("known_item", "topics")}
    topic_losses = sum(r["significant"] and r["mean_difference"] < 0 for r in by_type["topics"])
    known_losses = sum(r["significant"] and r["mean_difference"] < 0 for r in by_type["known_item"])
    findings["GN_vs_local"] = {
        "observations": [_describe(row) for row in gn],
        "interpretations": [
            f"Topics: GN {tally(by_type['topics'])} against local models. "
            f"Known-item: GN {tally(by_type['known_item'])}.",
            ("GN's disadvantage is concentrated in topic queries; for short known-item queries a large general-news "
             "vocabulary is competitive, whereas topic retrieval appears to benefit from collection-specific embeddings."
             if topic_losses > known_losses else
             "GN's disadvantage is not concentrated in one query type."),
            "Being trained on neither period, GN has no own-period advantage; its relative position against BM25 "
            "is reported in own_training_period."]}

    findings["publication_year"] = {
        "observations": year_analysis["observations"],
        "interpretations": year_analysis["interpretations"]}
    return findings


def run_temporal_generalization(official_dir, results_dir, output_dir, models_dir=None, previous_dir=None):
    """Part 4.4 from saved runs only: period-by-model tables, paired tests and KI2 year curves."""
    official_dir, results_dir = Path(official_dir).resolve(), Path(results_dir).resolve()
    output_dir = external_path(output_dir)
    previous_dir = Path(previous_dir or output_dir).resolve()
    models_dir = Path(models_dir or results_dir.parent / "models").resolve()
    if any((output_dir / name).exists() for name in PART44_OUTPUT_NAMES):
        raise FileExistsError("Part 4.4 output already exists; refusing to overwrite")
    started = perf_counter()
    selection_path = previous_dir / "part43_p1_selected.json"
    frozen = load_frozen_p1_selection(selection_path)
    selection_hash = sha256_file(selection_path)
    part43 = json.loads((previous_dir / "part43_summary.json").read_text(encoding="utf-8"))
    part42 = json.loads((previous_dir / "part42_summary.json").read_text(encoding="utf-8"))
    if part43.get("selection_manifest_sha256") != selection_hash or part42.get("frozen_selection_sha256") != selection_hash:
        raise ValueError("Part 4.2/4.3 outputs do not belong to the frozen P1 selection")
    if any(sha256_file(previous_dir / name) != digest for name, digest in part42["artifact_sha256"].items()):
        raise ValueError("Part 4.2 artifacts changed after publication")
    protected = {path: sha256_file(path) for directory in {output_dir, previous_dir} if directory.exists()
                 for path in directory.iterdir() if path.is_file()}
    provenance = verify_model_provenance(results_dir, models_dir)

    topics_path = official_dir / "topics.tsv"
    topics = read_table(topics_path, ["tid", "query"], "tid")
    input_hashes = {topics_path: sha256_file(topics_path), selection_path: selection_hash}
    per_query, means, files, parameters, reuse, years = {}, {}, {}, {}, [], {}
    period_years = {"P1": range(2005, 2015), "P2": range(2015, 2026)}
    for collection, period in (("C1", "P1"), ("C2", "P2")):
        paths = {"collection": official_dir / f"collection_{period}.tsv", "known": official_dir / f"queries_ki_{period}.tsv",
                 "known_item": official_dir / f"qrels_ki_{period}.txt", "topics": official_dir / f"qrels_topics_{period}.txt"}
        input_hashes.update({path: sha256_file(path) for path in paths.values()})
        official = read_table(paths["collection"], ["docno", "web_url", "year"], "docno")
        known = read_table(paths["known"], ["qid", "query", "docno", "year"], "qid")
        if (len(official) != EXPECTED_COLLECTION_COUNTS[collection] or len(known) != TUNING_QUERY_COUNTS["known_item"]
                or len(topics) != TUNING_QUERY_COUNTS["topics"]):
            raise ValueError("Part 4.4 requires every official query and document")
        qids = {"known_item": known["qid"].tolist(), "topics": topics["tid"].tolist()}
        all_qids = qids["known_item"] + qids["topics"]
        if len(set(all_qids)) != len(all_qids):
            raise ValueError("Known-item and topic IDs must be disjoint")
        years[period] = load_target_years(known, official, period_years[period])
        qrels = {kind: load_qrels(paths[kind], qids[kind], official["docno"]) for kind in qids}
        for qid, target in known[["qid", "docno"]].itertuples(index=False, name=None):
            if {d for d, grade in qrels["known_item"][qid].items() if grade >= 1} != {target}:
                raise ValueError("Known-item target and official qrels disagree")
        loaded = {}
        for kind in ("known_item", "topics"):
            test_set = PART44_TEST_SETS[collection, kind]
            internal, primary = TUNING_OBJECTIVES[kind]
            tuned = []
            for spec in part42_method_plan(frozen, collection, kind):
                if spec["method"] not in PART44_METHODS:
                    continue
                run_path = results_dir / spec["filename"]
                if spec["filename"] not in loaded:
                    input_hashes[run_path] = sha256_file(run_path)
                    rankings, details = load_run(run_path, all_qids, official["docno"])
                    if details["rows"] != 100 * len(all_qids) or details["missing_queries"]:
                        raise ValueError(f"Incomplete saved run: {run_path.name}")
                    loaded[spec["filename"]] = rankings
                evaluated = evaluate_queries(qids[kind], loaded[spec["filename"]], qrels[kind], kind)
                key = test_set, spec["method"]
                per_query[key] = [(row["qid"], row[internal]) for row in evaluated["per_query"]]
                means[key], files[key], parameters[key] = evaluated["means"], spec["filename"], spec["parameters"]
                if spec["family"] == "QE":
                    tuned.append({**spec, "source_file": spec["filename"], "sha256": input_hashes[run_path],
                                  "score": evaluated["means"][primary]})
            reuse.extend(verify_frozen_reuse(frozen, part43["results"], collection, kind, tuned))

    # Cross-check every mean and per-query primary value against the validated Part 4.2 outputs.
    table_names = {"KI1/C1": "part42_ki1_c1.csv", "KI2/C2": "part42_ki2_c2.csv",
                   "Topics/C1": "part42_topics_c1.csv", "Topics/C2": "part42_topics_c2.csv"}
    checked_means = 0
    for test_set, name in table_names.items():
        with (previous_dir / name).open(encoding="utf-8", newline="") as handle:
            saved = {row["method"]: row for row in csv.DictReader(handle)}
        for method in PART44_METHODS:
            if saved[method]["source_file"] != files[test_set, method]:
                raise ValueError(f"{test_set} {method} uses a different run than Part 4.2")
            for metric, value in means[test_set, method].items():
                if float(saved[method][metric]) != value:
                    raise ValueError(f"{test_set} {method} {metric} differs from Part 4.2")
                checked_means += 1
    expected = {(test_set, method, qid): value for (test_set, method), rows in per_query.items() for qid, value in rows}
    checked_queries = 0
    with (previous_dir / "part42_per_query.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["test_set"], row["method"], row["qid"]
            if key in expected and row["metric"] == ("RR@10" if row["query_type"] == "known_item" else "AP@100"):
                if float(row["value"]) != expected[key]:
                    raise ValueError(f"Per-query value differs from Part 4.2: {key}")
                checked_queries += 1
    if checked_queries != len(expected):
        raise ValueError("Part 4.2 per-query file does not cover every Part 4.4 query")

    comparison = []
    for kind in ("known_item", "topics"):
        metric = TUNING_OBJECTIVES[kind][1]
        p1, p2 = PART44_TEST_SETS["C1", kind], PART44_TEST_SETS["C2", kind]
        b1, b2 = means[p1, "BM25"][metric], means[p2, "BM25"][metric]
        for method in PART44_METHODS:
            family, model = method.split(" ", 1) if method != "BM25" else ("Lexical", "BM25")
            s1, s2 = means[p1, method][metric], means[p2, method][metric]
            comparison.append({
                "query_type": kind, "metric": metric, "method": method, "family": family, "model": model,
                "training_periods": PART44_MODELS[model]["training_periods"] if model in PART44_MODELS else "none",
                "parameters": json.dumps(parameters[p1, method], sort_keys=True, separators=(",", ":")),
                "P1_C1": s1, "P2_C2": s2, "change_P2_minus_P1": s2 - s1,
                "minus_BM25_P1": s1 - b1, "minus_BM25_P2": s2 - b2,
                "ratio_to_BM25_P1": s1 / b1, "ratio_to_BM25_P2": s2 / b2,
                "queries_P1": len(per_query[p1, method]), "queries_P2": len(per_query[p2, method]),
                "source_P1": files[p1, method], "source_P2": files[p2, method]})
            if parameters[p1, method] != parameters[p2, method]:
                raise AssertionError("P2 must reuse the P1 parameters unchanged")

    tests = []
    for (collection, kind), test_set in PART44_TEST_SETS.items():
        metric = TUNING_OBJECTIVES[kind][1]
        qid_order = [qid for qid, _ in per_query[test_set, "BM25"]]
        for baseline in dict.fromkeys(base for _, base in PART44_PAIRS):
            methods = [m for m, base in PART44_PAIRS if base == baseline]
            vectors = {}
            for name in [baseline] + methods:
                if [qid for qid, _ in per_query[test_set, name]] != qid_order:
                    raise AssertionError("Paired tests require identical official query order")
                vectors[name] = [value for _, value in per_query[test_set, name]]
            for result in paired_randomization_against_bm25(vectors, baseline=baseline):
                tests.append({"test_set": test_set, "collection": collection, "query_type": kind,
                              "primary_metric": metric, **result})

    # KI2/C2 publication-year analysis (target article year only).
    ki2 = {method: per_query["KI2/C2", method] for method in PART44_YEAR_METHODS}
    ki2_qids = [qid for qid, _ in ki2["BM25"]]
    values = {method: [value for _, value in rows] for method, rows in ki2.items()}
    year_rows = group_by_publication_year(ki2_qids, years["P2"], values)
    overall = {method: means["KI2/C2", method]["MRR@10"] for method in PART44_YEAR_METHODS}
    xs, ws = [row["year"] for row in year_rows], [row["queries"] for row in year_rows]
    year_pairs = (("Dense M2", "Dense M1"), ("Dense GN", "Dense M1"), ("Dense GN", "Dense M2"),
                  ("BM25", "Dense M1"), ("BM25", "Dense M2"), ("BM25", "Dense GN"))
    agreement = []
    for method, baseline in year_pairs:
        diffs = [row[method] - row[baseline] for row in year_rows]
        overall_diff = overall[method] - overall[baseline]
        same = [row["year"] for row, d in zip(year_rows, diffs) if d * overall_diff > 0]
        agreement.append({"comparison": f"{method} - {baseline}", "overall_difference": overall_diff,
                          "years_same_sign": len(same), "years": len(year_rows),
                          "years_opposite_sign": [row["year"] for row, d in zip(year_rows, diffs) if d * overall_diff < 0],
                          "per_year_difference": {row["year"]: d for row, d in zip(year_rows, diffs)},
                          "weighted_slope_per_year": weighted_slope(xs, diffs, ws)})
    best_each_year = {row["year"]: max(PART44_YEAR_METHODS, key=lambda m: row[m]) for row in year_rows}
    overall_best = max(PART44_YEAR_METHODS, key=overall.get)
    def trend(value):
        return "undefined (one year)" if value is None else f"{value:+.4f}/year"
    year_observations = [f"{a['comparison']}: overall {a['overall_difference']:+.4f}; same sign in "
                         f"{a['years_same_sign']}/{a['years']} years; weighted trend {trend(a['weighted_slope_per_year'])}"
                         for a in agreement]
    year_observations.append(f"Best of the four each year: {best_each_year}; overall best: {overall_best}")
    year_observations.append(f"Queries per year range {min(ws)}-{max(ws)}; total {sum(ws)}")
    consistent = [a for a in agreement if a["years_same_sign"] >= a["years"] - 2]
    m2_m1 = next(a for a in agreement if a["comparison"] == "Dense M2 - Dense M1")
    year_interpretations = [
        "Only one target year; no trend can be assessed." if m2_m1["weighted_slope_per_year"] is None else ("Within P2, the M2 - M1 difference shows no clear trend with distance from P1 "
         f"({m2_m1['weighted_slope_per_year']:+.4f}/year), so M1's deficit is not measurably larger for later years."
         if abs(m2_m1["weighted_slope_per_year"]) < .005 else
         f"The M2 - M1 difference trends {m2_m1['weighted_slope_per_year']:+.4f}/year across P2 target years."),
        f"{len(consistent)} of {len(agreement)} overall orderings hold in all but at most two years: "
        + ", ".join(a["comparison"] for a in consistent) + ".",
        "Per-year groups have 31-122 queries, so single-year reversals of small overall differences are expected "
        "noise; only orderings that persist across most years support the overall findings."]

    comparison_path = output_dir / "part44_temporal_comparison.csv"
    year_csv = [{"year": row["year"], "queries": row["queries"], **{f"MRR@10 {m}": row[m] for m in PART44_YEAR_METHODS}}
                for row in year_rows]
    query_year_rows = [{"qid": qid, "target_year": years["P2"][qid],
                        **{f"RR@10 {m}": values[m][i] for m in PART44_YEAR_METHODS}} for i, qid in enumerate(ki2_qids)]
    findings = part44_findings(comparison, tests,
                               {"observations": year_observations, "interpretations": year_interpretations})
    if any(sha256_file(path) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("A Part 4.4 input changed during analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(comparison_path, list(comparison[0]), comparison)
    _write_csv(output_dir / "part44_paired_tests.csv", list(tests[0]), tests)
    _write_csv(output_dir / "part44_ki2_by_year.csv", list(year_csv[0]), year_csv)
    _write_csv(output_dir / "part44_ki2_per_query_year.csv", list(query_year_rows[0]), query_year_rows)
    plot_ki2_year_curves(year_rows, overall, output_dir / "part44_ki2_mrr_by_year.png")
    if any(sha256_file(path) != digest for path, digest in protected.items()):
        raise RuntimeError("Part 4.4 changed a previous output")
    summary = {
        "part": "4.4", "all_complete": True, "retrieval_rerun": False, "embeddings_retrained": False,
        "indexes_rebuilt": False, "parameters_retuned": False,
        "frozen_selection_sha256": selection_hash, "qe_parameter_policy": PART42_PARAMETER_POLICY,
        "frozen_qe_reuse": {"passed": len(reuse) == 12, "checks": reuse},
        "model_provenance": provenance,
        "query_counts": {test_set: len(per_query[test_set, "BM25"]) for test_set in PART44_TEST_SETS.values()},
        "metric_validation": {"passed": True, "part42_means_matched": checked_means,
                              "part42_per_query_values_matched": checked_queries,
                              "rule": "Exact equality with validated Part 4.2 tables and per-query values"},
        "publication_year": {"source": "queries_ki_P2.tsv year of the official target article; equals collection_P2.tsv year for every target",
                             "queries_total": sum(ws), "groups": {row["year"]: row["queries"] for row in year_rows},
                             "overall_MRR@10": overall, "agreement_with_overall": agreement,
                             "best_method_each_year": best_each_year},
        "comparability_note": "P1 and P2 scores come from different queries and collections, so cross-period changes are "
                              "descriptive; BM25-relative columns adjust for test-set difficulty. Paired Appendix B tests "
                              "(10,000 sign flips, seed 42, p < 0.05) compare models only on the same queries.",
        "findings": findings,
        "files": list(PART44_OUTPUT_NAMES),
        "input_sha256": {str(path): digest for path, digest in input_hashes.items()},
        "elapsed_seconds": perf_counter() - started,
        "artifact_sha256": {name: sha256_file(output_dir / name) for name in PART44_OUTPUT_NAMES[:-1]},
    }
    write_json(output_dir / "part44_summary.json", summary)
    return summary


PART45_OUTPUT_NAMES = ("part45a_new_vocab_queries.csv", "part45a_subset_tests.csv",
                       "part45b_drift_groups.csv", "part45b_drift_queries.csv",
                       "part45c_topic_ap.csv", "part45c_expansion_terms.csv",
                       "part45d_hybrid_examples.csv", "part45_summary.json")
# Groups below this size are reported, but no significance conclusion is drawn from them.
PART45_SMALL_GROUP = 30
# Illustrative hybrid analysis only: the P1-selected known-item alpha is 1.0 (identical to BM25).
PART45_HYBRID = {"collection": "C2", "period": "P2", "model": "M2", "preferred_alpha": .5}
PART45_ALPHA_FALLBACK_ORDER = (.5, .4, .6, .3, .7, .2, .8, .1, .9)


def find_term_queries(query_tokens, terms):
    """Queries (in official order) containing at least one listed preprocessed term."""
    terms = set(terms)
    return {qid: sorted(set(tokens) & terms) for qid, tokens in query_tokens.items() if set(tokens) & terms}


def target_rank(ranking, target):
    """1-based rank of the official target in a saved top-k ranking, else None."""
    return next((rank for rank, docno in enumerate(ranking, 1) if docno == target), None)


def paired_subset_comparison(qids, subset, baseline_scores, method_scores, *, baseline="Dense M1",
                             method="Dense M2", primary_metric="MRR@10"):
    """Compare two systems inside a subset and its complement, testing each separately.

    Both groups keep official query order and pairing. Empty groups are reported
    without a test; groups smaller than PART45_SMALL_GROUP are flagged.
    """
    qids, subset = list(qids), set(subset)
    if len(baseline_scores) != len(qids) or len(method_scores) != len(qids) or not subset <= set(qids):
        raise ValueError("Subset comparison needs aligned scores and official query IDs")
    rows = []
    for group, members in (("subset", [i for i, q in enumerate(qids) if q in subset]),
                           ("remainder", [i for i, q in enumerate(qids) if q not in subset])):
        record = {"group": group, "queries": len(members), "share": len(members) / len(qids),
                  "primary_metric": primary_metric, "baseline": baseline, "method": method,
                  "small_group": len(members) < PART45_SMALL_GROUP}
        if members:
            a = [baseline_scores[i] for i in members]
            b = [method_scores[i] for i in members]
            test = paired_randomization_against_bm25({baseline: a, method: b}, baseline=baseline)[0]
            record.update(baseline_mean=math.fsum(a) / len(a), method_mean=math.fsum(b) / len(b),
                          mean_difference=test["mean_difference"], p_value=test["p_value"],
                          significant=test["significant"] and not record["small_group"],
                          raw_p_below_005=test["significant"], improved=sum(y > x for x, y in zip(a, b)),
                          worse=sum(y < x for x, y in zip(a, b)), tied=sum(y == x for x, y in zip(a, b)),
                          permutations=test["permutations"], random_seed=test["random_seed"])
        else:
            record.update(baseline_mean=None, method_mean=None, mean_difference=None, p_value=None,
                          significant=False, raw_p_below_005=False, improved=0, worse=0, tied=0,
                          permutations=None, random_seed=None)
        rows.append(record)
    return rows


def extreme_examples(rows, key, k=3, tie="qid"):
    """Largest positive and most negative ``key`` values; exact ties by ``tie`` ascending."""
    positive = sorted((r for r in rows if r[key] > 0), key=lambda r: (-r[key], r[tie]))[:k]
    negative = sorted((r for r in rows if r[key] < 0), key=lambda r: (r[key], r[tie]))[:k]
    return positive, negative


def qe_expansion(record, m, expansion_lambda):
    """Exactly QueryExpander.expand: first m cached neighbours, weight lambda * cosine."""
    parameter_key("QE", {"m": m, "lambda": expansion_lambda})
    if record["fallback"]:
        return []
    return [{"rank": i, "term": entry["term"], "cosine": entry["cosine"],
             "weight": expansion_lambda * entry["cosine"]}
            for i, entry in enumerate(record["neighbors"][:m], 1)]


def term_relevance_stats(weights, vocabulary, terms, relevant_rows):
    """Collection df and relevant-document df of each term from saved BM25 postings."""
    relevant_rows = np.asarray(sorted(relevant_rows), dtype=np.int64)
    stats = {}
    for term in terms:
        column = vocabulary.get(term)
        if column is None:
            stats[term] = {"c2_df": 0, "relevant_df": 0, "relevant_fraction": None}
            continue
        start, stop = weights.indptr[column:column + 2]
        rows = weights.indices[start:stop]
        relevant = int(np.isin(rows, relevant_rows).sum())
        stats[term] = {"c2_df": int(stop - start), "relevant_df": relevant,
                       "relevant_fraction": relevant / (stop - start) if stop > start else None}
    return stats


def weighted_relevant_fraction(expansion, stats):
    """Expansion-weight average of term relevant fractions (terms present in C2 only)."""
    usable = [(e["weight"], stats[e["term"]]["relevant_fraction"]) for e in expansion
              if stats[e["term"]]["relevant_fraction"] is not None and e["weight"] > 0]
    total = math.fsum(w for w, _ in usable)
    return math.fsum(w * f for w, f in usable) / total if total else None


def fused_ranking(docnos, bm25_scores, dense_scores, alpha):
    """Re-derive the saved Part 3.4 linear-fusion order over the docno-sorted union."""
    from .retrieval import linear_fusion, minmax_scores
    docnos = np.asarray(docnos, dtype=str)
    if list(docnos) != sorted(docnos):
        raise ValueError("Candidate union must be in docno order, as saved by Part 3.4")
    fused = linear_fusion(bm25_scores, dense_scores, alpha)
    order = np.lexsort((docnos, -fused))
    return {"docnos": docnos[order], "fused": fused[order],
            "bm25_norm": minmax_scores(bm25_scores)[order], "dense_norm": minmax_scores(dense_scores)[order],
            "bm25_raw": np.asarray(bm25_scores, dtype=np.float64)[order],
            "dense_raw": np.asarray(dense_scores, dtype=np.float64)[order]}


def choose_illustrative_alpha(outcomes_by_alpha, needed=3):
    """Use alpha=0.5 when it yields enough wins and losses; otherwise the nearest that does."""
    for alpha in PART45_ALPHA_FALLBACK_ORDER:
        wins, losses = outcomes_by_alpha.get(alpha, (0, 0))
        if wins >= needed and losses >= needed:
            return alpha
    raise ValueError("No nondegenerate saved alpha yields enough wins and losses")


PART45_PRECISE_TERM = {"min_relevant_fraction": .8, "min_df": 10}
PART45_NOISY_TERM = {"max_relevant_fraction": .2, "min_df": 500}


def pooled_relevant_fraction(expansion, stats):
    """Weight x document-frequency pooled share of relevant documents (high-df terms count more)."""
    pairs = [(e["weight"], stats[e["term"]]) for e in expansion if e["weight"] > 0]
    total = math.fsum(w * st["c2_df"] for w, st in pairs)
    return math.fsum(w * st["relevant_df"] for w, st in pairs) / total if total else None


def classify_expansion_terms(terms, stats):
    """Split terms into precise (mostly relevant) and noisy (frequent, mostly nonrelevant)."""
    precise = [t for t in terms if stats[t]["relevant_fraction"] is not None
               and stats[t]["c2_df"] >= PART45_PRECISE_TERM["min_df"]
               and stats[t]["relevant_fraction"] >= PART45_PRECISE_TERM["min_relevant_fraction"]]
    noisy = [t for t in terms if stats[t]["relevant_fraction"] is not None
             and stats[t]["c2_df"] >= PART45_NOISY_TERM["min_df"]
             and stats[t]["relevant_fraction"] < PART45_NOISY_TERM["max_relevant_fraction"]]
    return precise, noisy


def _fmt3(value):
    return "n/a" if value is None else f"{value:.3f}"


def _load_csc_postings(index_dir):
    from .retrieval import load_index
    metadata = json.loads((Path(index_dir) / "index.json").read_text(encoding="utf-8"))
    return load_index(index_dir, metadata["provenance"], "bm25")


def fusion_reason(outcome, target, competitor, alpha, hybrid_rank):
    """State the linear-fusion arithmetic that decided the order of target and competitor."""
    if target["fused_score"] is None:
        return "The target is outside the fusion candidate union, so fusion cannot retrieve it."
    lexical = target["bm25_normalized"] - competitor["bm25_normalized"]
    semantic = target["dense_normalized"] - competitor["dense_normalized"]
    combined = alpha * lexical + (1 - alpha) * semantic
    arithmetic = f"{alpha:.1f} x {lexical:+.3f} + {1 - alpha:.1f} x {semantic:+.3f} = {combined:+.3f}"
    if outcome == "hybrid_beats_BM25":
        return (f"Fusion helped: against the BM25 top document, the target's dense advantage ({semantic:+.3f}) "
                f"outweighs its BM25 shortfall ({lexical:+.3f}) at alpha={alpha} ({arithmetic}).")
    return (f"Fusion hurt: the target's BM25 lead ({lexical:+.3f}) over the hybrid top document is outweighed by "
            f"its dense deficit ({semantic:+.3f}) at alpha={alpha} ({arithmetic}); "
            f"{hybrid_rank - 1} documents now rank above the target.")


def run_per_query_analysis(official_dir, results_dir, output_dir, *, index_dir=None, models_dir=None,
                           part2_dir=None, previous_dir=None):
    """Part 4.5 using saved runs, caches, indexes and models read-only."""
    from gensim.models import Word2Vec
    from .preprocessing import preprocess_text

    official_dir, results_dir = Path(official_dir).resolve(), Path(results_dir).resolve()
    output_dir = external_path(output_dir)
    previous_dir = Path(previous_dir or output_dir).resolve()
    index_dir = Path(index_dir or results_dir.parent / "part3_indexes").resolve()
    models_dir = Path(models_dir or results_dir.parent / "models").resolve()
    part2_dir = Path(part2_dir or results_dir.parent / "part2_results").resolve()
    if any((output_dir / name).exists() for name in PART45_OUTPUT_NAMES):
        raise FileExistsError("Part 4.5 output already exists; refusing to overwrite")
    started = perf_counter()
    protected = {path: sha256_file(path) for directory in {output_dir, previous_dir} if directory.exists()
                 for path in directory.iterdir() if path.is_file()}
    selection_path = previous_dir / "part43_p1_selected.json"
    frozen = load_frozen_p1_selection(selection_path)
    part42 = json.loads((previous_dir / "part42_summary.json").read_text(encoding="utf-8"))
    if part42["frozen_selection_sha256"] != sha256_file(selection_path):
        raise ValueError("Part 4.2 outputs do not belong to the frozen P1 selection")
    provenance = verify_model_provenance(results_dir, models_dir)
    input_hashes = {}

    def hashed(path):
        input_hashes[Path(path)] = sha256_file(path)
        return Path(path)

    # ---- Official data for both periods ------------------------------------------------------
    topics = read_table(hashed(official_dir / "topics.tsv"), ["tid", "query"], "tid")
    periods = {}
    for collection, period in (("C1", "P1"), ("C2", "P2")):
        official = read_table(hashed(official_dir / f"collection_{period}.tsv"), ["docno", "web_url", "year"], "docno")
        known = read_table(hashed(official_dir / f"queries_ki_{period}.tsv"), ["qid", "query", "docno", "year"], "qid")
        if len(official) != EXPECTED_COLLECTION_COUNTS[collection] or len(known) != TUNING_QUERY_COUNTS["known_item"]:
            raise ValueError("Part 4.5 requires every official document and known-item query")
        qrels_ki = load_qrels(hashed(official_dir / f"qrels_ki_{period}.txt"), known["qid"], official["docno"])
        targets = dict(zip(known["qid"], known["docno"]))
        for qid, target in targets.items():
            if {d for d, g in qrels_ki[qid].items() if g >= 1} != {target}:
                raise ValueError("Known-item target and official qrels disagree")
        qrels_topics = load_qrels(hashed(official_dir / f"qrels_topics_{period}.txt"), topics["tid"], official["docno"])
        periods[period] = {"collection": collection, "official": official, "known": known, "targets": targets,
                           "qrels_ki": qrels_ki, "qrels_topics": qrels_topics,
                           "all_qids": known["qid"].tolist() + topics["tid"].tolist(),
                           "tokens": {q: preprocess_text(t) for q, t in zip(known["qid"], known["query"])},
                           "urls": dict(zip(official["docno"], official["web_url"]))}

    def evaluate_run(period, filename, kind):
        info = periods[period]
        rankings, details = load_run(hashed(results_dir / filename), info["all_qids"], info["official"]["docno"])
        if details["rows"] != 100 * len(info["all_qids"]) or details["missing_queries"]:
            raise ValueError(f"Incomplete saved run: {filename}")
        qids = info["known"]["qid"].tolist() if kind == "known_item" else topics["tid"].tolist()
        qrels = info["qrels_ki"] if kind == "known_item" else info["qrels_topics"]
        evaluated = evaluate_queries(qids, rankings, qrels, kind)
        metric = TUNING_OBJECTIVES[kind][0]
        return rankings, {row["qid"]: row[metric] for row in evaluated["per_query"]}

    # Saved per-query evaluations (Part 4.2 main runs; Part 4.3 for alternative hybrid alphas).
    saved42 = {}
    with (previous_dir / "part42_per_query.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] in ("RR@10", "AP@100"):
                saved42[row["test_set"], row["method"], row["qid"]] = (float(row["value"]), row["source_file"])
    verified = Counter()

    def check42(test_set, method, filename, values):
        for qid, value in values.items():
            saved, source = saved42[test_set, method, qid]
            if saved != value or source != filename:
                raise ValueError(f"Per-query value differs from Part 4.2: {test_set} {method} {qid}")
        verified[f"{test_set} {method}"] = len(values)

    # ---- (a) M2-only vocabulary on KI2 ---------------------------------------------------------
    vocab = {}
    for model in ("M1", "M2"):
        vectors = Word2Vec.load(str(models_dir / PART44_MODELS[model]["model_file"])).wv
        vocab[model] = set(vectors.key_to_index)
        del vectors
    reports = {m: json.loads((models_dir / f"{PART44_MODELS[m]['model_file']}.training.json").read_text(encoding="utf-8"))
               for m in ("M1", "M2")}
    if (len(vocab["M1"]) != reports["M1"]["vocabulary_size"]
            or len(vocab["M2"]) != reports["M2"]["updated_vocabulary_size"]):
        raise ValueError("Loaded vocabularies differ from the training reports")
    new_terms = vocab["M2"] - vocab["M1"]
    if len(new_terms) != reports["M2"]["new_words_added"]:
        raise ValueError("M2-only vocabulary size differs from the continued-training report")

    dense_runs, dense_rr = {}, {}
    for period, test_set in (("P1", "KI1/C1"), ("P2", "KI2/C2")):
        prefix = periods[period]["collection"].lower()
        for model in ("M1", "M2"):
            filename = f"{prefix}_dense_{model.lower()}.csv"
            rankings, values = evaluate_run(period, filename, "known_item")
            check42(test_set, f"Dense {model}", filename, values)
            dense_runs[period, model], dense_rr[period, model] = rankings, values

    p2 = periods["P2"]
    ki2_qids = p2["known"]["qid"].tolist()
    new_hits = find_term_queries(p2["tokens"], new_terms)
    subset_tests = paired_subset_comparison(ki2_qids, new_hits, [dense_rr["P2", "M1"][q] for q in ki2_qids],
                                            [dense_rr["P2", "M2"][q] for q in ki2_qids])
    query_text = dict(zip(p2["known"]["qid"], p2["known"]["query"]))

    def ki_row(period, qid, words, label="words"):
        info = periods[period]
        target = info["targets"][qid]
        r1, r2 = (target_rank(dense_runs[period, m][qid], target) for m in ("M1", "M2"))
        tokens = info["tokens"][qid]
        return {"qid": qid, "query": dict(zip(info["known"]["qid"], info["known"]["query"]))[qid],
                label: " ".join(words), "target_docno": target, "target_year": int(dict(zip(info["known"]["qid"], info["known"]["year"]))[qid]),
                "m1_missing_tokens": " ".join(t for t in tokens if t not in vocab["M1"]),
                "m1_missing_token_fraction": sum(t not in vocab["M1"] for t in tokens) / len(tokens) if tokens else 0.0,
                "dense_m1_rank": r1, "dense_m2_rank": r2,
                "rr10_m1": dense_rr[period, "M1"][qid], "rr10_m2": dense_rr[period, "M2"][qid],
                "rr10_change_m2_minus_m1": dense_rr[period, "M2"][qid] - dense_rr[period, "M1"][qid]}

    new_rows = [ki_row("P2", qid, words, "new_words") for qid, words in new_hits.items()]
    new_gain, new_loss = extreme_examples(new_rows, "rr10_change_m2_minus_m1")
    word_counts = Counter(w for words in new_hits.values() for w in words)
    fully_missing = sum(row["m1_missing_token_fraction"] == 1 for row in new_rows)

    # ---- (b) Part 2 most-drifted words ---------------------------------------------------------
    drift = json.loads(hashed(part2_dir / "semantic_drift.json").read_text(encoding="utf-8"))
    most_drifted, most_stable = drift["rankings"]["most_drifted"], drift["rankings"]["most_stable"]
    drift_groups, drift_query_rows, drift_examples = [], [], {}
    for period, test_set in (("P1", "KI1/C1"), ("P2", "KI2/C2")):
        info = periods[period]
        qids = info["known"]["qid"].tolist()
        m1 = [dense_rr[period, "M1"][q] for q in qids]
        m2 = [dense_rr[period, "M2"][q] for q in qids]
        hits = find_term_queries(info["tokens"], most_drifted)
        stable = find_term_queries(info["tokens"], most_stable)
        new_here = find_term_queries(info["tokens"], new_terms)
        groups = [("drifted_any", hits), ("stable_control_any", stable),
                  ("drifted_excluding_M2_only_vocab", {q: w for q, w in hits.items() if q not in new_here})]
        groups += [(f"drifted_word:{word}", {q: w for q, w in hits.items() if word in w}) for word in most_drifted]
        for name, members in groups:
            for row in paired_subset_comparison(qids, members, m1, m2):
                drift_groups.append({"test_set": test_set, "group_definition": name, **row})
        for qid, words in hits.items():
            drift_query_rows.append({"test_set": test_set, **ki_row(period, qid, words, "drifted_words"),
                                     "contains_M2_only_vocab": qid in new_here})
        rows_here = [r for r in drift_query_rows if r["test_set"] == test_set]
        gain, loss = extreme_examples(rows_here, "rr10_change_m2_minus_m1")
        drift_examples[test_set] = {"largest_M2_gains": gain, "largest_M2_losses": loss,
                                    "word_counts": dict(Counter(w for ws in hits.values() for w in ws)),
                                    "overlap_with_M2_only_vocab": sum(q in new_here for q in hits)}

    # ---- (c) Topics on C2: BM25, QE M1, QE M2 with frozen P1 topic settings -------------------
    plan = {spec["method"]: spec for spec in part42_method_plan(frozen, "C2", "topics")}
    topic_ap = {}
    for method in ("BM25", "QE M1", "QE M2"):
        _, values = evaluate_run("P2", plan[method]["filename"], "topics")
        check42("Topics/C2", method, plan[method]["filename"], values)
        topic_ap[method] = values
    topic_rows = [{"tid": tid, "query": q, **{f"AP@100 {m}": topic_ap[m][tid] for m in topic_ap},
                   "relevant_documents": sum(g >= 1 for g in p2["qrels_topics"][tid].values()),
                   "delta_QE_M2_minus_QE_M1": topic_ap["QE M2"][tid] - topic_ap["QE M1"][tid]}
                  for tid, q in zip(topics["tid"], topics["query"])]
    helped, hurt = extreme_examples(topic_rows, "delta_QE_M2_minus_QE_M1", tie="tid")
    roles = {r["tid"]: "helped" for r in helped} | {r["tid"]: "hurt" for r in hurt}
    for row in topic_rows:
        row["role"] = roles.get(row["tid"], "")

    caches = {}
    for model in ("M1", "M2"):
        pair = json.loads(hashed(results_dir / f"qe_c2_{model.lower()}" / "expansion_summary.json").read_text(encoding="utf-8"))
        cache_path = Path(pair["candidate_cache"])
        cache = json.loads(hashed(cache_path).read_text(encoding="utf-8"))
        if cache["provenance"] != pair["provenance"] or cache["max_terms"] != 20:
            raise ValueError(f"QE cache provenance differs from the saved C2 {model} run")
        caches[model] = {tuple(entry["tokens"]): entry for entry in cache["entries"]}
    reference = json.loads(hashed(results_dir / "qe_topic_expansions.json").read_text(encoding="utf-8"))
    cross_checked = 0
    for model in ("M1", "M2"):
        for item in reference["collections"]["C2"][model]["topics"]:
            if caches[model][tuple(item["tokens"])]["neighbors"] != item["neighbors"]:
                raise ValueError("QE cache disagrees with saved Part 3.3 topic expansions")
            cross_checked += 1
    bm25_c2 = _load_csc_postings(index_dir / "C2_bm25")
    if set(bm25_c2.docnos) != set(p2["official"]["docno"]):
        raise ValueError("C2 BM25 index docnos differ from the official collection")
    row_of = dict(zip(bm25_c2.docnos, range(len(bm25_c2.docnos))))
    expansion_rows, topic_explanations = [], []
    for row in helped + hurt:
        tid = row["tid"]
        tokens = tuple(preprocess_text(row["query"]))
        relevant = [row_of[d] for d, g in p2["qrels_topics"][tid].items() if g >= 1]
        per_model = {}
        for model in ("M1", "M2"):
            params = plan[f"QE {model}"]["parameters"]
            expansion = qe_expansion(caches[model][tokens], params["m"], params["lambda"])
            stats = term_relevance_stats(bm25_c2.weights, bm25_c2.vectorizer.vocabulary_,
                                         [e["term"] for e in expansion], relevant)
            per_model[model] = {"parameters": params, "expansion": expansion, "stats": stats,
                                "weighted_relevant_fraction": weighted_relevant_fraction(expansion, stats),
                                "pooled_relevant_fraction": pooled_relevant_fraction(expansion, stats)}
        terms = {m: [e["term"] for e in per_model[m]["expansion"]] for m in per_model}
        shared = set(terms["M1"]) & set(terms["M2"])
        for model, data in per_model.items():
            for e in data["expansion"]:
                expansion_rows.append({"tid": tid, "query": row["query"], "role": row["role"], "model": f"QE {model}",
                                       "m": data["parameters"]["m"], "lambda": data["parameters"]["lambda"], **e,
                                       "shared_with_other_model": e["term"] in shared, **data["stats"][e["term"]]})
        base_rate = len(relevant) / len(p2["official"])
        wf = {m: per_model[m]["weighted_relevant_fraction"] for m in per_model}
        pf = {m: per_model[m]["pooled_relevant_fraction"] for m in per_model}
        unique = {m: [t for t in terms[m] if t not in shared] for m in terms}
        kinds = {m: classify_expansion_terms(unique[m], per_model[m]["stats"]) for m in unique}

        def listed(model, words):
            st = per_model[model]["stats"]
            return ", ".join(f"{t} (df {st[t]['c2_df']}, {st[t]['relevant_fraction']:.0%} relevant)" for t in words) or "none"

        better, worse = ("M2", "M1") if row["role"] == "helped" else ("M1", "M2")
        evidence = []
        if kinds[better][0]:
            evidence.append(f"{better}-only precise terms: {listed(better, kinds[better][0])}")
        if kinds[worse][1]:
            evidence.append(f"{worse}-only noisy terms: {listed(worse, kinds[worse][1])}")
        against = []
        if kinds[worse][0]:
            against.append(f"{worse}-only precise terms: {listed(worse, kinds[worse][0])}")
        if kinds[better][1]:
            against.append(f"{better}-only noisy terms: {listed(better, kinds[better][1])}")
        interpretation = (("Consistent with the AP change: " + "; ".join(evidence) + "."
                           + (" Counter-evidence: " + "; ".join(against) + "." if against else ""))
                          if evidence else
                          "Model-specific expansion terms show no clear precise/noisy contrast, so the term statistics "
                          "alone do not explain this AP change." + (" Observed: " + "; ".join(against) + "." if against else ""))
        topic_explanations.append({
            "tid": tid, "query": row["query"], "role": row["role"],
            "AP@100": {m: row[f"AP@100 {m}"] for m in topic_ap}, "delta_QE_M2_minus_QE_M1": row["delta_QE_M2_minus_QE_M1"],
            "relevant_documents": len(relevant), "relevant_base_rate": base_rate,
            "shared_terms": sorted(shared), "M1_only_terms": [t for t in terms["M1"] if t not in shared],
            "M2_only_terms": [t for t in terms["M2"] if t not in shared],
            "per_term_mean_relevant_fraction": wf, "pooled_relevant_fraction": pf,
            "M1_only_precise_terms": kinds["M1"][0], "M1_only_noisy_terms": kinds["M1"][1],
            "M2_only_precise_terms": kinds["M2"][0], "M2_only_noisy_terms": kinds["M2"][1],
            "observation": (f"{tid} '{row['query']}': AP@100 BM25 {row['AP@100 BM25']:.4f}, QE M1 {row['AP@100 QE M1']:.4f}, "
                            f"QE M2 {row['AP@100 QE M2']:.4f} ({row['delta_QE_M2_minus_QE_M1']:+.4f}). "
                            f"{len(shared)}/20 expansion terms shared. Relevant share of documents containing each "
                            f"expansion term (weight-averaged per term): M1 {_fmt3(wf['M1'])}, M2 {_fmt3(wf['M2'])}; "
                            f"pooled over all matched documents: M1 {_fmt3(pf['M1'])}, M2 {_fmt3(pf['M2'])} "
                            f"(topic base rate {base_rate:.3f})."),
            "interpretation": interpretation})

    # ---- (d) Illustrative alternative hybrid (not P1-selected) --------------------------------
    hyb_model, period = PART45_HYBRID["model"], PART45_HYBRID["period"]
    prefix = PART45_HYBRID["collection"].lower()
    bm25_rankings, bm25_rr = evaluate_run(period, f"{prefix}_bm25.csv", "known_item")
    check42("KI2/C2", "BM25", f"{prefix}_bm25.csv", bm25_rr)
    saved43 = {}
    with (previous_dir / "part43_per_query.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["family"] == "HYBRID" and row["model"] == hyb_model and row["collection"] == PART45_HYBRID["collection"]
                    and row["query_type"] == "known_item"):
                saved43[json.loads(row["parameters"])["alpha"], row["qid"]] = float(row["value"])
    outcomes, hybrid_rr, hybrid_rankings = {}, {}, {}
    for alpha in PART45_ALPHA_FALLBACK_ORDER:
        filename = f"{prefix}_hybrid_{hyb_model.lower()}_a{alpha:.1f}".replace(".", "p") + ".csv"
        rankings, values = evaluate_run(period, filename, "known_item")
        if any(saved43[alpha, q] != v for q, v in values.items()):
            raise ValueError(f"Hybrid per-query RR@10 differs from Part 4.3 at alpha={alpha}")
        verified[f"KI2/C2 hybrid {hyb_model} alpha={alpha}"] = len(values)
        diffs = [values[q] - bm25_rr[q] for q in ki2_qids]
        outcomes[alpha] = (sum(d > 0 for d in diffs), sum(d < 0 for d in diffs))
        hybrid_rr[alpha], hybrid_rankings[alpha] = values, rankings
        if alpha == PART45_HYBRID["preferred_alpha"] and min(outcomes[alpha]) >= 3:
            break
    alpha = choose_illustrative_alpha(outcomes)
    hybrid_summary = json.loads(hashed(results_dir / f"hybrid_{prefix}_{hyb_model.lower()}" / "hybrid_summary.json")
                                .read_text(encoding="utf-8"))
    npz_path = results_dir / f"hybrid_{prefix}_{hyb_model.lower()}" / "candidate_scores.npz"
    if sha256_file(hashed(npz_path)) != hybrid_summary["candidate_cache"]["sha256"]:
        raise ValueError("Saved hybrid candidate scores changed")
    cached = np.load(npz_path)
    offsets = {q: (int(cached["offsets"][i]), int(cached["offsets"][i + 1])) for i, q in enumerate(cached["qids"].tolist())}
    rows = [{"qid": q, "rr10_bm25": bm25_rr[q], "rr10_hybrid": hybrid_rr[alpha][q],
             "rr10_change_hybrid_minus_bm25": hybrid_rr[alpha][q] - bm25_rr[q]} for q in ki2_qids]
    wins, losses = extreme_examples(rows, "rr10_change_hybrid_minus_bm25")
    vocabulary = bm25_c2.vectorizer.vocabulary_
    hybrid_examples = []
    for outcome, chosen in (("hybrid_beats_BM25", wins), ("hybrid_loses_to_BM25", losses)):
        for row in chosen:
            qid, target = row["qid"], p2["targets"][row["qid"]]
            start, stop = offsets[qid]
            fused = fused_ranking(cached["docnos"][start:stop], cached["bm25"][start:stop], cached["dense"][start:stop], alpha)
            saved_top = hybrid_rankings[alpha][qid]
            if list(fused["docnos"][:len(saved_top)]) != saved_top:
                raise ValueError(f"Re-derived fusion order differs from the saved hybrid run for {qid}")
            position = {d: i for i, d in enumerate(fused["docnos"])}
            bm25_r = target_rank(bm25_rankings[qid], target)
            dense_r = target_rank(dense_runs[period, hyb_model][qid], target)
            hybrid_r = position[target] + 1 if target in position else None
            competitor = bm25_rankings[qid][0] if outcome == "hybrid_beats_BM25" else str(fused["docnos"][0])
            tokens = sorted(set(preprocess_text(query_text[qid])))

            def describe(docno):
                i = position.get(docno)
                contributions = {t: float(bm25_c2.weights[row_of[docno], vocabulary[t]]) for t in tokens if t in vocabulary}
                return {"docno": docno, "web_url": p2["urls"][docno],
                        "bm25_rank": target_rank(bm25_rankings[qid], docno),
                        "dense_rank": target_rank(dense_runs[period, hyb_model][qid], docno),
                        "hybrid_rank": None if i is None else i + 1,
                        "bm25_score": None if i is None else float(fused["bm25_raw"][i]),
                        "dense_cosine": None if i is None else float(fused["dense_raw"][i]),
                        "bm25_normalized": None if i is None else float(fused["bm25_norm"][i]),
                        "dense_normalized": None if i is None else float(fused["dense_norm"][i]),
                        "fused_score": None if i is None else float(fused["fused"][i]),
                        "matched_query_terms": sorted(t for t, v in contributions.items() if v > 0),
                        "bm25_term_contributions": {t: v for t, v in contributions.items() if v > 0}}

            tgt, comp = describe(target), describe(competitor)
            missing = [t for t in tokens if t not in vocab[hyb_model]]
            role = "BM25 top document" if outcome == "hybrid_beats_BM25" else "hybrid top document"
            def fmt(v):
                return "n/a" if v is None else f"{v:.3f}"
            explanation = (
                f"Target {target} (BM25 rank {bm25_r or '>100'}, dense {hyb_model} rank {dense_r or '>100'}) has normalized "
                f"BM25 {fmt(tgt['bm25_normalized'])} and dense {fmt(tgt['dense_normalized'])}, fused {fmt(tgt['fused_score'])} "
                f"-> hybrid rank {hybrid_r or 'outside union'}. The {role} {competitor} has normalized BM25 "
                f"{fmt(comp['bm25_normalized'])}, dense {fmt(comp['dense_normalized'])}, fused {fmt(comp['fused_score'])}. "
                f"Query terms matched: target {len(tgt['matched_query_terms'])}/{len(tokens)}, {role} "
                f"{len(comp['matched_query_terms'])}/{len(tokens)}. " + fusion_reason(outcome, tgt, comp, alpha, hybrid_r)
                + (f" Query tokens missing from {hyb_model}: {', '.join(missing)}." if missing else ""))
            hybrid_examples.append({
                "outcome": outcome, "qid": qid, "query": query_text[qid], "target_docno": target,
                "target_web_url": p2["urls"][target], "model": hyb_model, "period": period,
                "collection": PART45_HYBRID["collection"], "alpha": alpha, "setting": "illustrative alternative (not P1-selected)",
                "bm25_rank": bm25_r, "hybrid_rank": hybrid_r, "dense_rank": dense_r,
                "rr10_bm25": row["rr10_bm25"], "rr10_hybrid": row["rr10_hybrid"],
                "rr10_change_hybrid_minus_bm25": row["rr10_change_hybrid_minus_bm25"],
                "query_tokens_missing_from_model": missing, "target": tgt, "competitor_role": role,
                "competitor": comp, "explanation": explanation})

    # ---- Interpretations derived from measured values ----------------------------------------
    sub, rest = subset_tests
    drift_main = next(g for g in drift_groups if g["test_set"] == "KI2/C2" and g["group_definition"] == "drifted_any"
                      and g["group"] == "subset")
    drift_rest = next(g for g in drift_groups if g["test_set"] == "KI2/C2" and g["group_definition"] == "drifted_any"
                      and g["group"] == "remainder")
    stable_main = next(g for g in drift_groups if g["test_set"] == "KI2/C2"
                       and g["group_definition"] == "stable_control_any" and g["group"] == "subset")
    drift_ex = next(g for g in drift_groups if g["test_set"] == "KI2/C2"
                    and g["group_definition"] == "drifted_excluding_M2_only_vocab" and g["group"] == "subset")
    ki1_drift = next(g for g in drift_groups if g["test_set"] == "KI1/C1" and g["group_definition"] == "drifted_any"
                     and g["group"] == "subset")
    dominant, dominant_n = Counter(drift_examples["KI2/C2"]["word_counts"]).most_common(1)[0]
    small_words = [g["group_definition"].split(":")[1] for g in drift_groups if g["test_set"] == "KI2/C2"
                   and g["group_definition"].startswith("drifted_word:") and g["group"] == "subset" and g["small_group"]]

    def signed(value):
        return "n/a" if value is None else f"{value:+.4f}"

    def plain(value):
        return "n/a" if value is None else f"{value:.4f}"

    findings = {
        "a_new_vocabulary": {
            "observations": [
                f"{len(new_hits)} of {len(ki2_qids)} KI2 queries ({len(new_hits) / len(ki2_qids):.1%}) contain at least one "
                f"of the {len(new_terms)} M2-only words; {fully_missing} of them have no token in M1's vocabulary.",
                f"Subset: Dense M1 MRR@10 {plain(sub['baseline_mean'])}, Dense M2 {plain(sub['method_mean'])}, "
                f"difference {signed(sub['mean_difference'])}, p={plain(sub['p_value'])}.",
                f"Remainder ({rest['queries']}): Dense M1 {plain(rest['baseline_mean'])}, Dense M2 {plain(rest['method_mean'])}, "
                f"difference {signed(rest['mean_difference'])}, p={plain(rest['p_value'])}."],
            "interpretations": [
                "Mechanism (from the retrieval code): dense query embeddings skip tokens without a vector, so M2-only "
                "words are ignored by M1 but used by M2.",
                (("M2's advantage over M1 is concentrated in this subset "
                  f"({signed(sub['mean_difference'])} vs {signed(rest['mean_difference'])} elsewhere)"
                  + ("; it is significant in the subset but not in the remaining queries." if sub["significant"]
                     and not rest["significant"] else "."))
                 if sub["mean_difference"] > rest["mean_difference"] else
                 "M2's advantage is not larger on queries with M2-only words, so new vocabulary does not explain it.")]},
        "b_semantic_drift": {
            "most_drifted_words_from_part2": most_drifted, "stable_control_words_from_part2": most_stable,
            "selection_rule_from_part2": drift["rankings"]["criterion"],
            "observations": [
                f"KI1/C1: {ki1_drift['queries']} queries contain a most-drifted word (empty group; no test).",
                f"KI2/C2: {drift_main['queries']} queries contain a most-drifted word; {dominant_n} contain '{dominant}'. "
                f"Word counts: {drift_examples['KI2/C2']['word_counts']}.",
                f"KI2 drifted group: M2 - M1 = {signed(drift_main['mean_difference'])} MRR@10 (p={plain(drift_main['p_value'])}); "
                f"other KI2 queries: {signed(drift_rest['mean_difference'])} (p={plain(drift_rest['p_value'])}).",
                f"Stable-word control group ({stable_main['queries']} queries): M2 - M1 = {signed(stable_main['mean_difference'])} "
                f"(p={plain(stable_main['p_value'])}; {'small group' if stable_main['small_group'] else 'adequate size'}).",
                f"Drifted group without M2-only vocabulary ({drift_ex['queries']} queries): M2 - M1 = "
                f"{signed(drift_ex['mean_difference'])}.",
                f"Single-word groups too small for conclusions (< {PART45_SMALL_GROUP} queries): {', '.join(small_words)}."],
            "interpretations": [
                (f"The drifted-word group is essentially a '{dominant}' group ({dominant_n}/{drift_main['queries']}), so any "
                 "effect cannot be attributed to drift in general."),
                ("M2's continued training, which updated these words' vectors, does not help on these queries: M2 - M1 is "
                 f"{signed(drift_main['mean_difference'])} here versus {signed(drift_rest['mean_difference'])} elsewhere"
                 + (" and the within-group difference is not significant." if not drift_main["raw_p_below_005"] else ".")
                 if drift_main["mean_difference"] <= drift_rest["mean_difference"] else
                 "M2 gains more on drifted-word queries than elsewhere, consistent with updated word meanings helping."),
                "Query behaviour therefore provides at most weak evidence of a semantic-drift effect; the group "
                "differences are descriptive, and comparing the two groups' gains is not itself a paired test."]},
        "c_topics": {"term_classes": {"precise": PART45_PRECISE_TERM, "noisy": PART45_NOISY_TERM,
                                      "statistics": "From saved C2 BM25 postings and official P2 topic qrels"},
                     "criterion": "delta = AP@100(QE M2) - AP@100(QE M1) per topic on C2; three largest positive "
                                  "(helped) and three most negative (hurt); exact ties by topic ID",
                     "qe_parameters": {m: plan[f"QE {m}"]["parameters"] for m in ("M1", "M2")},
                     "helped": [r["tid"] for r in helped], "hurt": [r["tid"] for r in hurt],
                     "topics": topic_explanations},
        "d_hybrid": {"label": "Illustrative alternative hybrid setting; NOT the P1-selected main-table setting",
                     "p1_selected_known_item_alpha": next(r["parameters"]["alpha"] for r in frozen["selected"]
                                                          if r["family"] == "HYBRID" and r["model"] == hyb_model
                                                          and r["query_type"] == "known_item"),
                     "why_not_p1_selected": "P1-selected known-item alpha=1.0 reproduces BM25 exactly, so it has no wins or losses",
                     "model": hyb_model, "collection": PART45_HYBRID["collection"], "period": period, "alpha": alpha,
                     "alpha_choice": ("alpha=0.5 used as specified" if alpha == .5 else
                                      f"alpha=0.5 lacked three wins and losses; nearest alpha with enough: {alpha}"),
                     "outcome_counts_by_alpha": {str(a): {"wins": w, "losses": l} for a, (w, l) in outcomes.items()},
                     "selection": "Three largest RR@10 gains and three largest losses versus BM25; exact ties by qid",
                     "examples": hybrid_examples}}

    # ---- Write outputs ------------------------------------------------------------------------
    if any(sha256_file(path) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("A Part 4.5 input changed during analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "part45a_new_vocab_queries.csv", list(new_rows[0]), new_rows)
    _write_csv(output_dir / "part45a_subset_tests.csv", list(subset_tests[0]), subset_tests)
    _write_csv(output_dir / "part45b_drift_groups.csv", list(drift_groups[0]), drift_groups)
    _write_csv(output_dir / "part45b_drift_queries.csv", list(drift_query_rows[0]), drift_query_rows)
    _write_csv(output_dir / "part45c_topic_ap.csv", list(topic_rows[0]), topic_rows)
    _write_csv(output_dir / "part45c_expansion_terms.csv", list(expansion_rows[0]), expansion_rows)
    flat = [{key: (json.dumps(value) if isinstance(value, (dict, list)) else value) for key, value in example.items()}
            for example in hybrid_examples]
    _write_csv(output_dir / "part45d_hybrid_examples.csv", list(flat[0]), flat)
    if any(sha256_file(path) != digest for path, digest in protected.items()):
        raise RuntimeError("Part 4.5 changed a previous output")
    summary = {
        "part": "4.5", "all_complete": True, "retrieval_rerun": False, "indexes_rebuilt": False,
        "embeddings_retrained": False, "main_results_modified": False,
        "query_counts": {"KI1/C1": len(periods["P1"]["known"]), "KI2/C2": len(ki2_qids), "Topics/C2": len(topic_rows)},
        "model_provenance_passed": provenance["passed"],
        "vocabulary": {"M1": len(vocab["M1"]), "M2": len(vocab["M2"]), "M2_only": len(new_terms),
                       "M1_only": len(vocab["M1"] - vocab["M2"]), "source": "Saved Word2Vec models; sizes match training reports"},
        "a_new_vocabulary": {"matching_queries": len(new_hits), "percentage": 100 * len(new_hits) / len(ki2_qids),
                             "queries_with_no_M1_token": fully_missing, "tests": subset_tests,
                             "most_common_new_words": word_counts.most_common(15),
                             "examples": {"largest_M2_gains": new_gain, "largest_M2_losses": new_loss},
                             "example_rule": "Three largest RR@10 changes in each direction within the subset; ties by qid"},
        "b_semantic_drift": {"groups": drift_groups, "examples": drift_examples, "small_group_threshold": PART45_SMALL_GROUP},
        "findings": findings,
        "verification": {"per_query_values_matched": dict(verified),
                         "rule": "Every per-query RR@10/AP@100 recomputed from saved runs equals Part 4.2 (main runs) or "
                                 "Part 4.3 (alternative hybrid alphas) exactly",
                         "qe_cache_provenance_matches_runs": True, "qe_cache_cross_checked_topics": cross_checked,
                         "hybrid_fusion_rederived_matches_saved_top100": True,
                         "known_item_targets_match_qrels": True},
        "appendix_b": {"permutations": PART42_PERMUTATIONS, "random_seed": PART42_RANDOM_SEED,
                       "p_value": "raw fraction, no plus-one correction", "significance": "p < 0.05",
                       "small_groups": f"Groups with fewer than {PART45_SMALL_GROUP} queries are reported with raw p but "
                                       "never marked significant"},
        "input_sha256": {str(path): digest for path, digest in input_hashes.items()},
        "elapsed_seconds": perf_counter() - started, "files": list(PART45_OUTPUT_NAMES),
        "artifact_sha256": {name: sha256_file(output_dir / name) for name in PART45_OUTPUT_NAMES[:-1]},
    }
    write_json(output_dir / "part45_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", choices=["bm25", "past_future", "refresh_plots", "main_results", "temporal", "per_query"], default="bm25")
    args = parser.parse_args()
    if args.experiment == "refresh_plots":
        summary = refresh_alpha_plots(args.output_dir)
        print(f"Updated {len(summary['plots'])} annotated plots using unchanged saved P1 choices and scores")
        return
    if args.official_dir is None or args.results_dir is None:
        parser.error("Evaluation requires --official-dir and --results-dir")
    if args.experiment == "past_future":
        summary = run_past_future_tuning(args.official_dir, args.results_dir, args.output_dir)
        print(f"Part 4.3 complete: {summary['evaluated_run_files']} existing CSVs, {summary['transfer_rows']} transfer comparisons")
        return
    if args.experiment == "per_query":
        summary = run_per_query_analysis(args.official_dir, args.results_dir, args.output_dir)
        print(f"Part 4.5 complete: {summary['a_new_vocabulary']['matching_queries']} KI2 queries with M2-only words; "
              f"helped {summary['findings']['c_topics']['helped']}, hurt {summary['findings']['c_topics']['hurt']}")
        return
    if args.experiment == "temporal":
        summary = run_temporal_generalization(args.official_dir, args.results_dir, args.output_dir)
        print(f"Part 4.4 complete: {summary['query_counts']}; KI2 year groups {summary['publication_year']['groups']}")
        return
    if args.experiment == "main_results":
        summary = run_main_results(args.official_dir, args.results_dir, args.output_dir)
        for test_set in summary["test_sets"]:
            print(f"{test_set['test_set']}: {test_set['queries']} queries, {test_set['methods']} methods -> {test_set['table_file']}")
        print(f"Significant vs BM25: {len(summary['significant_improvements_vs_BM25'])} improvements, "
              f"{len(summary['significant_declines_vs_BM25'])} declines")
        return
    summary = run_bm25_evaluation(args.official_dir, args.results_dir, args.output_dir)
    for result in summary["test_sets"]:
        print(f"{result['test_set']}: {result['queries']} queries; {result['means']}; independent validation passed")


if __name__ == "__main__":
    main()
