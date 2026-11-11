"""Part 4.1 metric examples and Part 4.3 transfer tests; assignment data stay unchanged."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation import (
    average_precision_at_k, compare_with_pytrec_eval, evaluate_queries,
    load_qrels, load_run, precision_at_k, recall_at_k, reciprocal_rank_at_k,
    run_bm25_evaluation, success_at_k,
    apply_p1_settings, discover_tuning_runs, evaluate_tuning_period, parameter_key,
    plot_alpha_curves, run_past_future_tuning, select_best_setting, select_p1_settings,
    alpha_plot_annotations, refresh_alpha_plots,
    PART42_METHOD_NAMES, load_frozen_p1_selection, paired_randomization_against_bm25,
    paired_randomization_test, part42_method_plan, run_main_results, verify_frozen_reuse,
    PART44_MODELS, group_by_publication_year, load_target_years, run_temporal_generalization,
    verify_model_provenance, weighted_slope,
    choose_illustrative_alpha, classify_expansion_terms, extreme_examples, find_term_queries, fused_ranking,
    fusion_reason, paired_subset_comparison, pooled_relevant_fraction, qe_expansion, run_per_query_analysis,
    target_rank, term_relevance_stats, weighted_relevant_fraction,
)
from src.retrieval import FUSION_ALPHAS, sha256_file


class EvaluationMetricTests(unittest.TestCase):
    def test_known_item_manual_rank_three(self):
        ranking, relevant = ["x", "y", "target"], {"target"}
        self.assertAlmostEqual(reciprocal_rank_at_k(ranking, relevant, 10), 1 / 3)
        self.assertEqual(success_at_k(ranking, relevant, 1), 0)
        self.assertEqual(success_at_k(ranking, relevant, 10), 1)
        self.assertEqual(recall_at_k(ranking, relevant, 100), 1)

    def test_known_item_cutoff_boundaries(self):
        for rank in [1, 10, 11, 100, 101]:
            ranking = [f"n{i}" for i in range(rank - 1)] + ["target"]
            with self.subTest(rank=rank):
                self.assertEqual(reciprocal_rank_at_k(ranking, {"target"}, 10), 1 / rank if rank <= 10 else 0)
                self.assertEqual(success_at_k(ranking, {"target"}, 1), float(rank == 1))
                self.assertEqual(success_at_k(ranking, {"target"}, 10), float(rank <= 10))
                self.assertEqual(recall_at_k(ranking, {"target"}, 100), float(rank <= 100))

    def test_topic_manual_ap_uses_total_relevant_not_retrieved_count(self):
        ranking, relevant = ["x", "a", "y", "b", "c"], {"a", "b", "c", "unretrieved"}
        self.assertEqual(precision_at_k(ranking, relevant, 10), .3)
        self.assertEqual(precision_at_k(ranking, relevant, 100), .03)
        self.assertAlmostEqual(average_precision_at_k(ranking, relevant, 100), (.5 + .5 + .6) / 4)
        self.assertEqual(recall_at_k(ranking, relevant, 100), .75)

    def test_ap_denominator_is_capped_at_cutoff(self):
        relevant = {f"d{i}" for i in range(200)}
        ranking = [f"d{i}" for i in range(100)]
        self.assertEqual(average_precision_at_k(ranking, relevant, 100), 1)
        self.assertEqual(recall_at_k(ranking, relevant, 100), .5)

    def test_precision_uses_fixed_cutoff_for_short_runs(self):
        self.assertEqual(precision_at_k(["a", "x"], {"a"}, 10), .1)
        self.assertEqual(precision_at_k(["a"], {"a"}, 100), .01)
        self.assertEqual(average_precision_at_k(["a"], {"a", "b", "c"}, 100), 1 / 3)

    def test_empty_and_no_relevant_queries_are_zero(self):
        functions = [precision_at_k, recall_at_k, reciprocal_rank_at_k, success_at_k, average_precision_at_k]
        for function in functions:
            self.assertEqual(function([], {"a"}, 100), 0)
            self.assertEqual(function(["a"], set(), 100), 0)

    def test_ap_does_not_count_relevance_after_rank_100(self):
        ranking = [f"x{i}" for i in range(100)] + ["a"]
        self.assertEqual(average_precision_at_k(ranking, {"a"}, 100), 0)
        self.assertEqual(precision_at_k(ranking, {"a"}, 100), 0)

    def test_recall_divides_by_all_relevant_documents(self):
        self.assertAlmostEqual(recall_at_k(["a", "x", "b"], {"a", "b", "unretrieved"}, 100), 2 / 3)

    def test_all_official_queries_including_missing_runs_contribute_to_means(self):
        qids = ["q1", "q2", "q3"]
        qrels = {qid: {"a": 1} for qid in qids}
        result = evaluate_queries(qids, {"q1": ["a"], "q2": ["x"]}, qrels, "known_item")
        self.assertEqual(result["queries"], 3)
        self.assertEqual([row["qid"] for row in result["per_query"]], qids)
        self.assertEqual(result["means"], {"MRR@10": 1 / 3, "Success@1": 1 / 3,
                                           "Success@10": 1 / 3, "Recall@100": 1 / 3})
        self.assertEqual(result["per_query"][2]["RR@10"], 0)

    def test_binary_relevance_and_topic_means(self):
        result = evaluate_queries(["t1", "t2"], {"t1": ["nonrel", "a"]},
                                  {"t1": {"nonrel": 0, "a": 2, "unjudged": -1}, "t2": {}}, "topics")
        self.assertEqual(result["per_query"][0]["relevant_total"], 1)
        self.assertEqual(result["means"], {"P@10": .05, "P@100": .005, "MAP@100": .25})

    def test_invalid_cutoffs_and_duplicate_documents_are_rejected(self):
        for k in [0, -1, 1.5, True]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                precision_at_k(["a"], {"a"}, k)
        with self.assertRaises(ValueError):
            average_precision_at_k(["a", "a"], {"a"}, 100)
        for qids, kind in [([], "topics"), (["q", "q"], "topics"), (["q"], "unknown")]:
            with self.assertRaises(ValueError):
                evaluate_queries(qids, {}, {}, kind)


class EvaluationInputTests(unittest.TestCase):
    def write_run(self, path, rows, columns=("qid", "docno", "rank", "score")):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)

    def test_run_loader_preserves_explicit_tied_ranks_and_missing_queries(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.csv"
            self.write_run(path, [["q1", "a", 1, 0], ["q1", "b", 2, 0]])
            before = path.read_bytes()
            rankings, details = load_run(path, ["q1", "q2"], ["a", "b"])
            self.assertEqual(rankings, {"q1": ["a", "b"], "q2": []})
            self.assertEqual(details["missing_queries"], ["q2"])
            self.assertEqual(details["rows"], 2)
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_run_ids_ranks_duplicates_and_scores_are_rejected(self):
        good = [["q", "a", 1, 2], ["q", "b", 2, 1]]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.csv"
            for i, column, value in [(0, 0, "foreign"), (0, 1, "foreign"), (1, 1, "a"),
                                     (0, 2, 2), (1, 2, 1), (1, 3, 3), (0, 3, "nan"),
                                     (0, 3, "inf"), (0, 3, "")]:
                bad = [row.copy() for row in good]
                bad[i][column] = value
                self.write_run(path, bad)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_run(path, ["q"], ["a", "b"])
            self.write_run(path, good, columns=["wrong", "docno", "rank", "score"])
            with self.assertRaises(ValueError):
                load_run(path, ["q"], ["a", "b"])

    def test_qrels_loader_checks_ids_duplicates_and_field_count(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "qrels.txt"
            path.write_text("q 0 a 1\nq 0 b 0\n", encoding="utf-8")
            self.assertEqual(load_qrels(path, iter(["q", "missing"]), ["a", "b"]),
                             {"q": {"a": 1, "b": 0}, "missing": {}})
            for text in ["q 0 a 1\nq 0 a 1\n", "foreign 0 a 1\n", "q 0 outside 1\n", "q a 1\n", "q 0 a invalid\n"]:
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_qrels(path, ["q"], ["a", "b"])


class IndependentEvaluatorTests(unittest.TestCase):
    def test_pytrec_known_item_cutoff_missing_query_and_tie_order(self):
        qids = ["tie", "rank11", "missing", "noqrels"]
        runs = {"tie": ["a", "b"], "rank11": [f"x{i}" for i in range(10)] + ["a"]}
        qrels = {"tie": {"a": 1}, "rank11": {"a": 1}, "missing": {"a": 1}, "noqrels": {}}
        ours = evaluate_queries(qids, runs, qrels, "known_item")
        compared = compare_with_pytrec_eval(qids, runs, qrels, ours, "known_item")
        self.assertTrue(compared["passed"])
        self.assertEqual(len(compared["per_query"]), 16)
        self.assertTrue(all(item["largest_native_per_query_absolute_difference"] == 0 for item in compared["metrics"].values()))
        rr11 = next(row for row in compared["per_query"] if row["qid"] == "rank11" and row["metric"] == "RR@10")
        self.assertEqual(rr11["external_native"], 0)
        recall11 = next(row for row in compared["per_query"] if row["qid"] == "rank11" and row["metric"] == "Recall@100")
        self.assertEqual(recall11["external_native"], 1)

    def test_pytrec_native_ap_difference_and_independent_denominator_conversion(self):
        qrels = {"large": {f"d{i}": 1 for i in range(200)},
                 "manual": {"a": 1, "b": 1, "c": 1, "unretrieved": 1}, "empty": {}}
        runs = {"large": [f"d{i}" for i in range(100)], "manual": ["x", "a", "y", "b", "c"]}
        ours = evaluate_queries(list(qrels), runs, qrels, "topics")
        compared = compare_with_pytrec_eval(list(qrels), runs, qrels, ours, "topics")
        self.assertTrue(compared["passed"])
        ap = compared["metrics"]["MAP@100"]
        self.assertFalse(ap["native_definition_matches"])
        self.assertEqual(ap["largest_native_per_query_absolute_difference"], .5)
        self.assertEqual(ap["largest_native_difference_qid"], "large")
        self.assertLessEqual(ap["largest_assignment_per_query_absolute_difference"], 1e-12)
        row = next(row for row in compared["per_query"] if row["qid"] == "large" and row["metric"] == "AP@100")
        self.assertEqual(row["ours"], 1)
        self.assertEqual(row["external_native"], .5)
        self.assertEqual(row["external_assignment_definition"], 1)
        self.assertEqual(row["external_relevant_total"], 200)


class EvaluationPipelineTests(unittest.TestCase):
    def test_four_separate_sets_complete_query_universes_and_preserved_inputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            official, results = root / "official", root / "part3_results"
            official.mkdir()
            results.mkdir()
            pd.DataFrame({"tid": [f"T{i:02d}" for i in range(32)], "query": ["topic"] * 32}).to_csv(
                official / "topics.tsv", sep="\t", index=False)
            for collection, period in [("C1", "P1"), ("C2", "P2")]:
                docs = [f"{collection}-{i:06d}" for i in range(105)]
                qids = [f"{period}-{i:04d}" for i in range(1000)]
                pd.DataFrame({"docno": docs, "web_url": [f"url/{d}" for d in docs]}).to_csv(
                    official / f"collection_{period}.tsv", sep="\t", index=False)
                pd.DataFrame({"qid": qids, "query": ["known"] * 1000, "docno": [docs[0]] * 1000}).to_csv(
                    official / f"queries_ki_{period}.tsv", sep="\t", index=False)
                (official / f"qrels_ki_{period}.txt").write_text(
                    "".join(f"{qid} 0 {docs[0]} 1\n" for qid in qids), encoding="utf-8")
                (official / f"qrels_topics_{period}.txt").write_text(
                    "".join(f"T{i:02d} 0 {docs[0]} 1\n" for i in range(32)), encoding="utf-8")
                with (results / f"{collection.lower()}_bm25.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["qid", "docno", "rank", "score"])
                    writer.writerow([qids[0], docs[1], 1, 2])
                    if collection == "C1":
                        writer.writerow([qids[0], docs[0], 2, 1])
                        writer.writerow(["T00", docs[0], 1, 1])
            before = {p: sha256_file(p) for directory in [official, results] for p in directory.iterdir()}
            with patch.dict("src.evaluation.EXPECTED_COLLECTION_COUNTS", {"C1": 105, "C2": 105}), \
                 patch("src.retrieval.build_index", side_effect=AssertionError("Index rebuilding prohibited")), \
                 patch("src.retrieval.run_pipeline", side_effect=AssertionError("Retrieval prohibited")):
                summary = run_bm25_evaluation(official, results, root / "part4_results")
                with self.assertRaises(FileExistsError):
                    run_bm25_evaluation(official, results, root / "part4_results")
            self.assertTrue(summary["all_passed"])
            self.assertEqual([(s["test_set"], s["queries"]) for s in summary["test_sets"]],
                             [("KI1", 1000), ("Topics_C1", 32), ("KI2", 1000), ("Topics_C2", 32)])
            scores = {s["test_set"]: s["means"] for s in summary["test_sets"]}
            self.assertEqual(scores["KI1"]["MRR@10"], .0005)
            self.assertEqual(scores["KI1"]["Success@10"], .001)
            self.assertEqual(scores["Topics_C1"]["MAP@100"], 1 / 32)
            self.assertTrue(all(value == 0 for value in scores["KI2"].values()))
            self.assertTrue(all(value == 0 for value in scores["Topics_C2"].values()))
            frame = pd.read_csv(root / "part4_results" / "bm25_per_query.csv")
            self.assertEqual(len(frame), 8192)
            self.assertEqual(frame[["test_set", "qid"]].drop_duplicates().shape[0], 2064)
            self.assertEqual(len(pd.read_csv(root / "part4_results" / "bm25_metrics.csv")), 14)
            for p, digest in before.items():
                self.assertEqual(sha256_file(p), digest)
            self.assertTrue(json.loads((root / "part4_results" / "bm25_validation_summary.json").read_text())["all_passed"])


def tuning_row(parameters, score, *, family="QE", model="M1", kind="known_item", period="P1"):
    return {"family": family, "model": model, "query_type": kind, "period": period,
            "collection": "C1" if period == "P1" else "C2", "parameters": parameters,
            "metric": "MRR@10" if kind == "known_item" else "MAP@100", "score": score,
            "filename": f"{period}_{family}_{model}_{parameter_key(family, parameters)}.csv",
            "queries": 1000 if kind == "known_item" else 32}


def annotation_fixture():
    rows = [tuning_row({"alpha": alpha}, .9 - .3 * (alpha - (.2 if period == "P1" else .8)) ** 2,
                       family="HYBRID", model=model, kind=kind, period=period)
            for period in ("P1", "P2") for kind in ("known_item", "topics")
            for model in ("M1", "M2", "GN") for alpha in FUSION_ALPHAS]
    selected = select_p1_settings([row for row in rows if row["period"] == "P1"])
    transferred = apply_p1_settings(selected, [row for row in rows if row["period"] == "P2"])
    return rows, selected, transferred


class PastFutureSelectionTests(unittest.TestCase):
    def test_best_full_precision_score_wins_without_rounding(self):
        rows = [tuning_row({"m": 5, "lambda": .5}, .8),
                tuning_row({"m": 20, "lambda": 1.}, .800000000001)]
        self.assertEqual(select_best_setting(rows)["parameters"], {"m": 20, "lambda": 1.})
        self.assertEqual(select_best_setting(list(reversed(rows))), select_best_setting(rows))

    def test_qe_exact_ties_prefer_smallest_m_then_lambda(self):
        rows = [tuning_row({"m": m, "lambda": weight}, .7)
                for m, weight in [(20, .5), (5, 1.), (10, .5), (5, .5)]]
        self.assertEqual(select_best_setting(rows)["parameters"], {"m": 5, "lambda": .5})
        self.assertEqual(select_best_setting(rows[1:]+rows[:1]), select_best_setting(rows))

    def test_hybrid_exact_ties_prefer_smaller_alpha(self):
        rows = [tuning_row({"alpha": alpha}, .9, family="HYBRID") for alpha in [1., .5, 0.]]
        self.assertEqual(select_best_setting(rows)["parameters"], {"alpha": 0.})

    def test_query_types_are_tuned_separately(self):
        rows = [tuning_row({"m": 5, "lambda": .5}, .9), tuning_row({"m": 10, "lambda": .5}, .8),
                tuning_row({"m": 5, "lambda": .5}, .4, kind="topics"),
                tuning_row({"m": 10, "lambda": .5}, .7, kind="topics")]
        chosen = {row["query_type"]: row for row in select_p1_settings(rows)}
        self.assertEqual(chosen["known_item"]["parameters"]["m"], 5)
        self.assertEqual(chosen["topics"]["parameters"]["m"], 10)
        with self.assertRaisesRegex(ValueError, "mix"):
            select_best_setting(rows)

    def test_future_scores_cannot_enter_past_selection(self):
        past = tuning_row({"m": 5, "lambda": .5}, .8)
        future = tuning_row({"m": 10, "lambda": 1.}, 1., period="P2")
        with self.assertRaisesRegex(ValueError, "only P1/C1"):
            select_p1_settings([past, future])
        with self.assertRaisesRegex(ValueError, "mix"):
            select_best_setting([past, future])
        with self.assertRaisesRegex(ValueError, "P2/C2"):
            apply_p1_settings([past], [past])

    def test_frozen_parameters_applied_unchanged_despite_better_future_setting(self):
        past = [tuning_row({"m": 5, "lambda": .5}, .8), tuning_row({"m": 10, "lambda": 1.}, .7)]
        selected = select_p1_settings(past)
        frozen_before = json.dumps(selected, sort_keys=True)
        future = [tuning_row({"m": 5, "lambda": .5}, .25, period="P2"),
                  tuning_row({"m": 10, "lambda": 1.}, .9, period="P2")]
        result = apply_p1_settings(selected, future)[0]
        self.assertEqual(result["p1_selected_parameters"], {"m": 5, "lambda": .5})
        self.assertEqual(result["p2_transferred_score"], .25)
        self.assertEqual(result["p2_oracle_parameters"], {"m": 10, "lambda": 1.})
        self.assertAlmostEqual(result["p2_oracle_gap"], .65)
        self.assertEqual(json.dumps(selected, sort_keys=True), frozen_before)
        future[1]["score"] = .1
        self.assertEqual(apply_p1_settings(selected, future)[0]["p1_selected_parameters"], selected[0]["parameters"])
        self.assertEqual(select_p1_settings(past), selected)

    def test_missing_future_counterpart_does_not_silently_apply_oracle(self):
        selected = select_p1_settings([tuning_row({"alpha": .2}, .8, family="HYBRID")])
        future = [tuning_row({"alpha": .9}, .99, family="HYBRID", period="P2")]
        with self.assertRaisesRegex(ValueError, "unchanged P1 setting"):
            apply_p1_settings(selected, future)

    def test_future_oracle_ties_and_zero_gap(self):
        selected = select_p1_settings([tuning_row({"alpha": .7}, .8, family="HYBRID")])
        future = [tuning_row({"alpha": .7}, .6, family="HYBRID", period="P2"),
                  tuning_row({"alpha": .3}, .6, family="HYBRID", period="P2")]
        result = apply_p1_settings(selected, future)[0]
        self.assertEqual(result["p2_oracle_parameters"], {"alpha": .3})
        self.assertEqual(result["p1_selected_parameters"], {"alpha": .7})
        self.assertEqual(result["p2_oracle_gap"], 0)

    def test_invalid_parameter_grids_and_objectives_rejected(self):
        for family, parameters in [("QE", {"m": 3, "lambda": .5}), ("QE", {"m": 5, "lambda": 0}),
                                   ("HYBRID", {"alpha": .05}), ("HYBRID", {"alpha": False})]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                parameter_key(family, parameters)
        row = tuning_row({"m": 5, "lambda": .5}, .8)
        row["metric"] = "MAP@100"
        with self.assertRaisesRegex(ValueError, "Wrong objective"):
            select_best_setting([row])

    def test_plot_annotations_transfer_frozen_alpha_and_keep_oracle_separate(self):
        rows, selected, transfers = annotation_fixture()
        before = json.dumps(selected, sort_keys=True)
        annotations = alpha_plot_annotations(rows, selected, transfers)
        self.assertEqual(len(annotations), 6)
        for marker in annotations.values():
            self.assertEqual(marker["p1_alpha"], .2)
            self.assertEqual(marker["p2_oracle_alpha"], .8)
            self.assertEqual(marker["p1_score"], .9)
            self.assertAlmostEqual(marker["p2_score"], .792)
            self.assertEqual(marker["p2_oracle_score"], .9)
        self.assertEqual(json.dumps(selected, sort_keys=True), before)

    def test_plot_annotations_reject_future_choices_and_inconsistent_transfers(self):
        rows, selected, transfers = annotation_fixture()
        bad = [dict(row) for row in selected]
        bad[0]["period"] = "P2"
        with self.assertRaisesRegex(ValueError, "frozen P1/C1"):
            alpha_plot_annotations(rows, bad, transfers)
        bad_transfers = [dict(row) for row in transfers]
        bad_transfers[0]["p1_selected_parameters"] = {"alpha": .8}
        with self.assertRaisesRegex(ValueError, "differs from frozen"):
            alpha_plot_annotations(rows, selected, bad_transfers)


class PastFuturePipelineTests(unittest.TestCase):
    def build_inputs(self, root):
        official, runs, output = root / "official", root / "runs", root / "part4"
        official.mkdir()
        runs.mkdir()
        output.mkdir()
        (output / "bm25_existing.csv").write_text("previous Part 4.1 artifact\n", encoding="utf-8")
        topics = ["T01", "T02"]
        pd.DataFrame({"tid": topics, "query": ["topic a", "topic b"]}).to_csv(official / "topics.tsv", sep="\t", index=False)
        manifests = {family: {"all_complete": True, "all_completed_results_valid": True, "results": []}
                     for family in ("QE", "HYBRID")}
        for collection, period in [("C1", "P1"), ("C2", "P2")]:
            docs = [f"{collection}-{i:06d}" for i in range(105)]
            known = [f"{period}-a", f"{period}-b"]
            pd.DataFrame({"docno": docs, "web_url": [f"url/{d}" for d in docs]}).to_csv(
                official / f"collection_{period}.tsv", sep="\t", index=False)
            pd.DataFrame({"qid": known, "query": ["known a", "known b"], "docno": [docs[0]] * 2}).to_csv(
                official / f"queries_ki_{period}.tsv", sep="\t", index=False)
            for qids, label in [(known, "ki"), (topics, "topics")]:
                (official / f"qrels_{label}_{period}.txt").write_text(
                    "".join(f"{qid} 0 {docs[0]} 1\n" for qid in qids), encoding="utf-8")
            for family in ("QE", "HYBRID"):
                grid = ([{"m": m, "lambda": weight} for m in [5, 10, 20] for weight in [.5, 1.]] if family == "QE"
                        else [{"alpha": alpha} for alpha in FUSION_ALPHAS])
                for model in ["M1", "M2", "GN"]:
                    pair = {"model": model, "collection": collection, "configurations": []}
                    for index, params in enumerate(grid):
                        filename = f"{collection}_{family}_{model}_{index}.csv"
                        with (runs / filename).open("w", newline="", encoding="utf-8") as handle:
                            writer = csv.writer(handle)
                            writer.writerow(["qid", "docno", "rank", "score"])
                            for kind, qids in [("known_item", known), ("topics", topics)]:
                                if family == "QE":
                                    past_best = {"m": 5 if kind == "known_item" else 10, "lambda": .5}
                                    future_best = {"m": 10 if kind == "known_item" else 20, "lambda": 1.}
                                else:
                                    past_best = {"alpha": .2 if kind == "known_item" else .8}
                                    future_best = {"alpha": .6 if kind == "known_item" else .4}
                                target_rank = ((1 if params == past_best else 2) if period == "P1" else
                                               (1 if params == future_best else (3 if params == past_best else 2)))
                                order = docs[1:100].copy()
                                order.insert(target_rank - 1, docs[0])
                                for qid in qids:
                                    writer.writerows([qid, docno, rank, 101 - rank] for rank, docno in enumerate(order, 1))
                        pair["configurations"].append({**params, "filename": filename, "sha256": sha256_file(runs / filename),
                                                       **({"method": "linear"} if family == "HYBRID" else {})})
                    manifests[family]["results"].append(pair)
        for family, filename in [("QE", "qe_summary.json"), ("HYBRID", "hybrid_summary.json")]:
            (runs / filename).write_text(json.dumps(manifests[family]), encoding="utf-8")
        return official, runs, output

    def test_complete_pipeline_freezes_past_before_future_and_preserves_inputs(self):
        with TemporaryDirectory() as temporary:
            official, runs, output = self.build_inputs(Path(temporary))
            original = {p: sha256_file(p) for directory in [official, runs, output] for p in directory.iterdir()}
            seen = []
            def period_spy(catalogue, official_dir, results_dir, period):
                frozen = output / "part43_p1_selected.json"
                if period == "P1":
                    self.assertFalse(frozen.exists())
                else:
                    self.assertTrue(frozen.exists())
                    rows = json.loads(frozen.read_text())["selected"]
                    self.assertTrue(all(row["period"] == "P1" and row["collection"] == "C1" for row in rows))
                seen.append(period)
                return evaluate_tuning_period(catalogue, official_dir, results_dir, period)
            def fake_plots(rows, outdir, *, p1_selection, transfer_results, overwrite=False):
                self.assertTrue(all(row["period"] == "P1" for row in p1_selection))
                self.assertEqual(len(transfer_results), 12)
                names = [f"part43_alpha_{period}_{kind}.png" for period in ("p1", "p2") for kind in ("known_item", "topics")]
                for name in names:
                    (Path(outdir) / name).write_bytes(b"test figure placeholder")
                return names
            with patch.dict("src.evaluation.EXPECTED_COLLECTION_COUNTS", {"C1": 105, "C2": 105}), \
                 patch.dict("src.evaluation.TUNING_QUERY_COUNTS", {"known_item": 2, "topics": 2}), \
                 patch("src.evaluation.evaluate_tuning_period", side_effect=period_spy), \
                 patch("src.evaluation.plot_alpha_curves", side_effect=fake_plots), \
                 patch("src.retrieval.run_pipeline", side_effect=AssertionError("Retrieval prohibited")):
                summary = run_past_future_tuning(official, runs, output)
                with self.assertRaises(FileExistsError):
                    run_past_future_tuning(official, runs, output)
            self.assertEqual(seen, ["P1", "P2"])
            self.assertTrue(summary["P1_frozen_before_P2_evaluation"])
            self.assertFalse(summary["P2_influences_P1_selection"])
            self.assertEqual(summary["evaluated_run_files"], 102)
            self.assertEqual(summary["tuning_score_rows"], 204)
            self.assertEqual(summary["per_query_rows"], 408)
            self.assertEqual(summary["transfer_rows"], 12)
            for row in summary["results"]:
                self.assertEqual(row["p1_score"], 1)
                self.assertAlmostEqual(row["p2_transferred_score"], 1 / 3)
                self.assertEqual(row["p2_oracle_score"], 1)
                self.assertAlmostEqual(row["p2_oracle_gap"], 2 / 3)
                expected = ({"m": 5 if row["query_type"] == "known_item" else 10, "lambda": .5} if row["family"] == "QE"
                            else {"alpha": .2 if row["query_type"] == "known_item" else .8})
                self.assertEqual(row["p1_selected_parameters"], expected)
            for p, digest in original.items():
                self.assertEqual(sha256_file(p), digest)

    def test_discovery_rejects_incomplete_parameter_grid(self):
        with TemporaryDirectory() as temporary:
            _, runs, _ = self.build_inputs(Path(temporary))
            path = runs / "qe_summary.json"
            manifest = json.loads(path.read_text())
            manifest["results"][0]["configurations"].pop()
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incomplete parameter grid"):
                discover_tuning_runs(runs)

    def test_four_plots_include_complete_curves(self):
        rows, selected, transfers = annotation_fixture()
        from matplotlib.figure import Figure
        original = Figure.savefig
        captured = []
        def capture(figure, *args, **kwargs):
            axis = figure.axes[0]
            captured.append((axis.get_title(), [(artist.get_gid(), artist.get_offsets().tolist())
                                                for artist in axis.collections]))
            return original(figure, *args, **kwargs)
        with TemporaryDirectory() as temporary:
            with patch.object(Figure, "savefig", new=capture):
                files = plot_alpha_curves(rows, temporary, p1_selection=selected, transfer_results=transfers)
            self.assertEqual(len(files), 4)
            for name in files:
                payload = (Path(temporary) / name).read_bytes()
                self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertGreater(len(payload), 1000)
        self.assertEqual(len(captured), 4)
        for title, markers in captured:
            stars = [point[0] for gid, point in markers if gid.startswith("P1-selected:")]
            oracles = [point[0] for gid, point in markers if gid.startswith("P2-oracle-analysis-only:")]
            self.assertEqual(len(stars), 3)
            self.assertTrue(all(point[0] == .2 for point in stars))
            if "P2" in title:
                self.assertEqual(len(oracles), 3)
                self.assertTrue(all(point[0] == .8 for point in oracles))
                self.assertTrue(all(abs(point[1] - .792) < 1e-12 for point in stars))
            else:
                self.assertEqual(oracles, [])

    def test_plot_refresh_only_reads_saved_scores_and_preserves_nonfigure_artifacts(self):
        rows, selected, transfers = annotation_fixture()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / "part43_p1_selected.json"
            selection.write_text(json.dumps({"selected": selected}), encoding="utf-8")
            with (root / "part43_tuning_scores.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows({**row, "parameters": json.dumps(row["parameters"])} for row in rows)
            (root / "part43_transfer_results.csv").write_text("saved transfer table", encoding="utf-8")
            (root / "part43_per_query.csv").write_text("saved per-query scores", encoding="utf-8")
            hashes = {p.name: sha256_file(p) for p in root.iterdir()}
            summary = {"all_complete": True, "selection_manifest_sha256": sha256_file(selection),
                       "results": transfers, "artifact_sha256": hashes}
            (root / "part43_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with patch("src.evaluation.select_p1_settings", side_effect=AssertionError("Retuning prohibited")), \
                 patch("src.evaluation.evaluate_tuning_period", side_effect=AssertionError("Reevaluation prohibited")):
                updated = refresh_alpha_plots(root)
            self.assertEqual(len(updated["plots"]), 4)
            self.assertEqual(updated["selection_manifest_sha256"], hashes[selection.name])
            for name, digest in hashes.items():
                self.assertEqual(sha256_file(root / name), digest)
            self.assertIn("analysis only", updated["plot_selection_policy"]["diamond"])
            self.assertIn("Part 4.2", updated["part4_2_parameter_policy"])


def reference_randomization_p(method, bm25, permutations=10_000, seed=42):
    """Literal Appendix B loop: flip each d_i with probability 1/2, compare |mean|."""
    import numpy as np
    differences = [a - b for a, b in zip(method, bm25)]
    n = len(differences)
    observed = abs(sum(differences) / n)
    bits = np.random.default_rng(seed).integers(0, 2, size=(permutations, n), dtype=np.int8)
    extreme = 0
    for row in bits:
        statistic = abs(sum(d if bit else -d for bit, d in zip(row, differences)) / n)
        extreme += statistic >= observed - 1e-12
    return extreme / permutations


class FixedSigns:
    """Stand-in generator returning predetermined 0/1 sign bits in draw order."""
    def __init__(self, bits):
        self.bits, self.cursor = bits, 0

    def integers(self, low, high, size, dtype):
        import numpy as np
        rows, _ = size
        block = self.bits[self.cursor:self.cursor + rows]
        self.cursor += rows
        return np.asarray(block, dtype=dtype)


class RandomizationTestFormulaTests(unittest.TestCase):
    def test_matches_literal_appendix_b_loop_with_seed_42(self):
        import numpy as np
        rng = np.random.default_rng(7)
        bm25 = rng.random(32)
        for shift in (0., .05, .2):
            method = np.clip(bm25 + shift + rng.normal(0, .2, 32), 0, 1)
            with self.subTest(shift=shift):
                result = paired_randomization_test(method, bm25)
                self.assertEqual(result["p_value"], reference_randomization_p(method, bm25))
                self.assertEqual(result["permutations"], 10_000)
                self.assertEqual(result["random_seed"], 42)
                self.assertEqual(result["p_value"], result["extreme_permutations"] / 10_000)

    def test_statistic_is_absolute_mean_and_two_sided(self):
        bm25 = [.5, .2, .9, .1, .4]
        better = [.9, .6, 1., .5, .7]
        worse = [2 * b - m for m, b in zip(better, bm25)]  # Mirrored differences.
        up, down = paired_randomization_test(better, bm25), paired_randomization_test(worse, bm25)
        self.assertAlmostEqual(up["mean_difference"], -down["mean_difference"])
        self.assertAlmostEqual(up["observed_absolute_mean_difference"], abs(up["mean_difference"]))
        self.assertEqual(up["p_value"], down["p_value"])
        self.assertEqual(up["alternative"], "two-sided")

    def test_no_plus_one_correction(self):
        bm25 = [0.] * 40
        result = paired_randomization_test([1.] * 40, bm25)
        self.assertEqual(result["extreme_permutations"], 0)  # P(all 40 signs equal) ~ 2e-12.
        self.assertEqual(result["p_value"], 0.)
        self.assertTrue(result["significant"])
        self.assertEqual(result["direction"], "significant_improvement")

    def test_identical_methods_have_p_one(self):
        result = paired_randomization_test([.3, .7, 1., 0.], [.3, .7, 1., 0.])
        self.assertEqual(result["p_value"], 1.)
        self.assertEqual(result["direction"], "not_significant")

    def test_threshold_is_strictly_less_than_005(self):
        # d = (+1, +1): only the (+,+) and (-,-) sign rows are as extreme as observed.
        for extreme, significant in [(500, False), (499, True)]:
            bits = [[1, 1]] * extreme + [[1, 0]] * (10_000 - extreme)
            with self.subTest(extreme=extreme), \
                 patch("src.evaluation.np.random.default_rng", return_value=FixedSigns(bits)):
                result = paired_randomization_test([0., 0.], [1., 1.])
            self.assertEqual(result["p_value"], extreme / 10_000)
            self.assertEqual(result["significant"], significant)
            self.assertEqual(result["direction"], "significant_decline" if significant else "not_significant")

    def test_rounding_ties_count_as_at_least_as_extreme(self):
        import numpy as np
        values = [0, 1, .5, 1 / 3, .25, .2, -1, -.5, -1 / 3, 1 / 7, -1 / 9]
        for seed in range(40):
            d = np.random.default_rng(seed).choice(values, size=1000)
            bits = [[1] * 1000, [0] * 1000] * 5_000  # Identity and global flip are exact ties.
            with self.subTest(seed=seed), \
                 patch("src.evaluation.np.random.default_rng", return_value=FixedSigns(bits)):
                result = paired_randomization_test(d, np.zeros(1000))
            self.assertEqual(result["p_value"], 1.)

    def test_pairing_and_query_order(self):
        import numpy as np
        rng = np.random.default_rng(3)
        bm25, method = rng.random(50), rng.random(50)
        order = rng.permutation(50)
        same = paired_randomization_test(method, bm25)
        reordered = paired_randomization_test(method[order], bm25[order])
        self.assertEqual(same["mean_difference"], paired_randomization_test(method, bm25)["mean_difference"])
        self.assertAlmostEqual(same["mean_difference"], reordered["mean_difference"])
        self.assertEqual(same["queries"], 50)
        with self.assertRaises(ValueError):
            paired_randomization_test(method[:49], bm25)
        with self.assertRaises(ValueError):
            paired_randomization_test([np.nan] * 50, bm25)

    def test_batch_comparisons_equal_individual_tests(self):
        import numpy as np
        rng = np.random.default_rng(11)
        vectors = {"BM25": rng.random(32), "A": rng.random(32), "B": rng.random(32)}
        batch = {row["method"]: row for row in paired_randomization_against_bm25(vectors)}
        for name in ("A", "B"):
            single = paired_randomization_test(vectors[name], vectors["BM25"])
            self.assertEqual(batch[name]["p_value"], single["p_value"])


def frozen_part42_selection(c1_hashes=None, scores=None):
    chosen = {("QE", "known_item"): {"m": 5, "lambda": .5}, ("QE", "topics"): {"m": 10, "lambda": 1.},
              ("HYBRID", "known_item"): {"alpha": .2}, ("HYBRID", "topics"): {"alpha": .8}}
    rows = []
    for (family, kind), parameters in chosen.items():
        for model in ("M1", "M2", "GN"):
            label = "qe" if family == "QE" else "hybrid"
            suffix = (f"m{parameters['m']}_l{parameters['lambda']:.1f}" if family == "QE"
                      else f"a{parameters['alpha']:.1f}").replace(".", "p")
            filename = f"c1_{label}_{model.lower()}_{suffix}.csv"
            rows.append({"family": family, "model": model, "collection": "C1", "period": "P1",
                         "parameters": dict(parameters), "filename": filename,
                         "sha256": (c1_hashes or {}).get(filename, "0" * 64), "query_type": kind,
                         "metric": "MRR@10" if kind == "known_item" else "MAP@100",
                         "queries": 1000 if kind == "known_item" else 32,
                         "score": (scores or {}).get(kind, .5)})
    return {"selection_period": "P1", "selection_collection": "C1", "selected": rows}


class Part42PlanTests(unittest.TestCase):
    def test_plan_has_15_methods_and_query_type_specific_p1_settings(self):
        frozen = frozen_part42_selection()
        for collection in ("C1", "C2"):
            known = part42_method_plan(frozen, collection, "known_item")
            topics = part42_method_plan(frozen, collection, "topics")
            self.assertEqual(tuple(row["method"] for row in known), PART42_METHOD_NAMES)
            files = {row["method"]: row["filename"] for row in known}
            topic_files = {row["method"]: row["filename"] for row in topics}
            prefix = collection.lower()
            self.assertEqual(files["QE M1"], f"{prefix}_qe_m1_m5_l0p5.csv")
            self.assertEqual(topic_files["QE M1"], f"{prefix}_qe_m1_m10_l1p0.csv")
            self.assertEqual(files["Linear hybrid GN"], f"{prefix}_hybrid_gn_a0p2.csv")
            self.assertEqual(topic_files["Linear hybrid GN"], f"{prefix}_hybrid_gn_a0p8.csv")
            self.assertEqual(files["Dense M2s"], f"{prefix}_dense_m2s.csv")
            self.assertEqual(files["RRF GN"], f"{prefix}_rrf_gn.csv")

    def test_p2_plan_reuses_p1_parameters_unchanged(self):
        frozen = frozen_part42_selection()
        before = json.dumps(frozen, sort_keys=True)
        for kind in ("known_item", "topics"):
            p1 = {row["method"]: row["parameters"] for row in part42_method_plan(frozen, "C1", kind)}
            p2 = {row["method"]: row["parameters"] for row in part42_method_plan(frozen, "C2", kind)}
            self.assertEqual(p1, p2)
        self.assertEqual(json.dumps(frozen, sort_keys=True), before)

    def test_frozen_manifest_rejects_future_rows_and_mismatched_files(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "selected.json"
            path.write_text(json.dumps(frozen_part42_selection()), encoding="utf-8")
            self.assertEqual(len(load_frozen_p1_selection(path)["selected"]), 12)
            for field, value in [("period", "P2"), ("collection", "C2"), ("filename", "c1_qe_m1_m20_l0p5.csv"),
                                 ("parameters", {"m": 7, "lambda": .5}), ("metric", "MAP@100")]:
                payload = frozen_part42_selection()
                payload["selected"][0][field] = value
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_frozen_p1_selection(path)
            payload = frozen_part42_selection()
            payload["selected"].pop()
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_frozen_p1_selection(path)

    def test_reuse_check_rejects_p2_oracle_runs(self):
        frozen = frozen_part42_selection()
        transfer = [{"family": "QE", "model": "M1", "query_type": "known_item",
                     "p1_selected_parameters": {"m": 5, "lambda": .5},
                     "p2_transferred_filename": "c2_qe_m1_m5_l0p5.csv", "p2_transferred_score": .3,
                     "p2_oracle_filename": "c2_qe_m1_m20_l1p0.csv", "p2_oracle_score": .4}]
        good = {"method": "QE M1", "family": "QE", "model": "M1", "parameters": {"m": 5, "lambda": .5},
                "source_file": "c2_qe_m1_m5_l0p5.csv", "sha256": "x", "score": .3}
        self.assertTrue(verify_frozen_reuse(frozen, transfer, "C2", "known_item", [good])[0]["matches_part43"])
        oracle = {**good, "parameters": {"m": 20, "lambda": 1.}, "source_file": "c2_qe_m1_m20_l1p0.csv", "score": .4}
        for bad in (oracle, {**good, "source_file": "c2_qe_m1_m20_l1p0.csv"}, {**good, "score": .4}):
            with self.assertRaises(ValueError):
                verify_frozen_reuse(frozen, transfer, "C2", "known_item", [bad])
        c1 = {**good, "source_file": "c1_qe_m1_m5_l0p5.csv", "sha256": "wrong", "score": .5}
        with self.assertRaisesRegex(ValueError, "frozen P1 run"):
            verify_frozen_reuse(frozen, transfer, "C1", "known_item", [c1])


class Part42PipelineTests(unittest.TestCase):
    RANKS = {"tfidf": 1, "dense": 4}

    def write_run(self, path, qids, docs, rank):
        order = docs[1:100].copy()
        order.insert(rank - 1, docs[0])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["qid", "docno", "rank", "score"])
            for qid in qids:
                writer.writerows([qid, docno, i, 101 - i] for i, docno in enumerate(order, 1))

    def build(self, root):
        official, runs, output = root / "official", root / "part3_results", root / "part4_results"
        for directory in (official, runs, output):
            directory.mkdir()
        topics = ["T01", "T02"]
        pd.DataFrame({"tid": topics, "query": ["a", "b"]}).to_csv(official / "topics.tsv", sep="\t", index=False)
        frozen = frozen_part42_selection(scores={"known_item": 1., "topics": 1.})
        for collection, period in (("C1", "P1"), ("C2", "P2")):
            docs = [f"{collection}-{i:06d}" for i in range(105)]
            known = [f"{period}-a", f"{period}-b"]
            pd.DataFrame({"docno": docs, "web_url": [f"u/{d}" for d in docs]}).to_csv(
                official / f"collection_{period}.tsv", sep="\t", index=False)
            pd.DataFrame({"qid": known, "query": ["x", "y"], "docno": [docs[0]] * 2}).to_csv(
                official / f"queries_ki_{period}.tsv", sep="\t", index=False)
            for qids, label in ((known, "ki"), (topics, "topics")):
                (official / f"qrels_{label}_{period}.txt").write_text(
                    "".join(f"{qid} 0 {docs[0]} 1\n" for qid in qids), encoding="utf-8")
            prefix, qids = collection.lower(), known + topics
            names = ["tfidf", "bm25"] + [f"{kind}_{m}" for kind in ("dense",) for m in ("m1", "m2", "m2s", "gn")]
            names += [f"rrf_{m}" for m in ("m1", "m2", "gn")]
            for name in names:
                self.write_run(runs / f"{prefix}_{name}.csv", qids, docs, self.RANKS.get(name.split("_")[0], 2))
            for row in frozen["selected"]:
                selected = row["filename"].replace("c1_", f"{prefix}_", 1)
                self.write_run(runs / selected, qids, docs, 1 if collection == "C1" else 3)
            if collection == "C2":  # Better future-only settings that must never be used.
                for model in ("m1", "m2", "gn"):
                    self.write_run(runs / f"c2_qe_{model}_m20_l0p5.csv", qids, docs, 1)
                    self.write_run(runs / f"c2_hybrid_{model}_a0p6.csv", qids, docs, 1)
        for row in frozen["selected"]:
            row["sha256"], row["queries"] = sha256_file(runs / row["filename"]), 2
        selection = output / "part43_p1_selected.json"
        selection.write_text(json.dumps(frozen), encoding="utf-8")
        transfers = [{"family": row["family"], "model": row["model"], "query_type": row["query_type"],
                      "p1_selected_parameters": row["parameters"],
                      "p2_transferred_filename": row["filename"].replace("c1_", "c2_", 1),
                      "p2_transferred_score": 1 / 3, "p2_oracle_score": 1.,
                      "p2_oracle_filename": (f"c2_qe_{row['model'].lower()}_m20_l0p5.csv" if row["family"] == "QE"
                                             else f"c2_hybrid_{row['model'].lower()}_a0p6.csv")}
                     for row in frozen["selected"]]
        (output / "part43_summary.json").write_text(json.dumps(
            {"all_complete": True, "selection_manifest_sha256": sha256_file(selection), "results": transfers}),
            encoding="utf-8")
        (output / "bm25_metrics.csv").write_text("previous Part 4.1 artifact\n", encoding="utf-8")
        return official, runs, output

    def test_four_tables_use_frozen_p1_settings_and_preserve_inputs(self):
        with TemporaryDirectory() as temporary:
            official, runs, output = self.build(Path(temporary))
            before = {p: sha256_file(p) for directory in (official, runs, output) for p in directory.iterdir()}
            with patch.dict("src.evaluation.EXPECTED_COLLECTION_COUNTS", {"C1": 105, "C2": 105}), \
                 patch.dict("src.evaluation.TUNING_QUERY_COUNTS", {"known_item": 2, "topics": 2}), \
                 patch("src.evaluation.select_p1_settings", side_effect=AssertionError("Retuning prohibited")), \
                 patch("src.retrieval.build_index", side_effect=AssertionError("Index rebuilding prohibited")), \
                 patch("src.retrieval.run_pipeline", side_effect=AssertionError("Retrieval prohibited")):
                summary = run_main_results(official, runs, output)
                with self.assertRaises(FileExistsError):
                    run_main_results(official, runs, output)
            for path, digest in before.items():
                self.assertEqual(sha256_file(path), digest)
            self.assertTrue(summary["frozen_parameter_reuse"]["passed"])
            self.assertEqual(len(summary["frozen_parameter_reuse"]["checks"]), 24)
            tables = {name: pd.read_csv(output / name, keep_default_na=False) for name in summary["tables"]}
            self.assertEqual(sorted(tables), ["part42_ki1_c1.csv", "part42_ki2_c2.csv",
                                              "part42_topics_c1.csv", "part42_topics_c2.csv"])
            for name, table in tables.items():
                self.assertEqual(tuple(table["method"]), PART42_METHOD_NAMES)
                self.assertFalse(table["source_file"].str.contains("m20_l0p5|a0p6").any(), name)
                bm25 = table.set_index("method").loc["BM25"]
                self.assertEqual(bm25["marker"], "BM25")
                self.assertEqual(bm25["p_value_vs_BM25"], "")
            ki2 = tables["part42_ki2_c2.csv"].set_index("method")
            self.assertEqual(ki2.loc["QE M1", "source_file"], "c2_qe_m1_m5_l0p5.csv")
            self.assertEqual(json.loads(ki2.loc["QE M1", "parameters"]), {"m": 5, "lambda": .5})
            self.assertAlmostEqual(ki2.loc["QE M1", "MRR@10"], 1 / 3)
            self.assertEqual(list(ki2.columns[:6]), ["family", "model", "parameters", "source_file", "queries", "MRR@10"])
            self.assertEqual(set(ki2["queries"]), {2})
            topics2 = tables["part42_topics_c2.csv"].set_index("method")
            self.assertEqual(topics2.loc["Linear hybrid GN", "source_file"], "c2_hybrid_gn_a0p8.csv")
            self.assertAlmostEqual(topics2.loc["Linear hybrid GN", "MAP@100"], 1 / 3)
            self.assertEqual({"P@10", "P@100", "MAP@100"} - set(topics2.columns), set())
            ki1 = tables["part42_ki1_c1.csv"].set_index("method")
            self.assertEqual(ki1.loc["QE GN", "MRR@10"], 1.)
            self.assertAlmostEqual(ki1.loc["TF-IDF", "primary_delta_vs_BM25"], .5)
            tests = pd.read_csv(output / "part42_randomization_tests.csv")
            self.assertEqual(len(tests), 4 * 14)
            self.assertEqual(set(tests["permutations"]), {10_000})
            self.assertEqual(set(tests["random_seed"]), {42})
            self.assertEqual(set(tests["primary_metric"]), {"MRR@10", "MAP@100"})
            self.assertTrue(((tests["p_value"] * 10_000).round() == tests["extreme_permutations"]).all())
            self.assertEqual(len(pd.read_csv(output / "part42_per_query.csv")), 2 * (2 * 4 + 2 * 3) * 15)
            saved = json.loads((output / "part42_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["retrieval_rerun"])
            self.assertFalse(saved["parameters_retuned"])
            for name, digest in saved["artifact_sha256"].items():
                self.assertEqual(sha256_file(output / name), digest)


class PublicationYearTests(unittest.TestCase):
    def test_groups_partition_all_queries_and_keep_zero_reciprocal_ranks(self):
        qids = ["a", "b", "c", "d", "e"]
        years = {"a": 2016, "b": 2015, "c": 2016, "d": 2025, "e": 2016}
        rr = [1., .5, 0., 1 / 3, 0.]  # c and e missed the top 10 and still count.
        rows = group_by_publication_year(qids, years, {"Dense M1": rr, "BM25": [1.] * 5})
        self.assertEqual([row["year"] for row in rows], [2015, 2016, 2025])
        self.assertEqual([row["queries"] for row in rows], [1, 3, 1])
        self.assertEqual(sum(row["queries"] for row in rows), len(qids))
        self.assertEqual(rows[1]["Dense M1"], 1 / 3)
        self.assertEqual(rows[0]["Dense M1"], .5)
        self.assertEqual(rows[1]["BM25"], 1.)

    def test_year_means_recombine_to_the_overall_mrr(self):
        import numpy as np
        rng = np.random.default_rng(5)
        qids = [f"q{i}" for i in range(1000)]
        years = {qid: int(rng.integers(2015, 2026)) for qid in qids}
        rr = [float(x) for x in rng.choice([0, 1, .5, 1 / 3, .25, .1], size=1000)]
        rows = group_by_publication_year(qids, years, {"m": rr})
        self.assertEqual(sum(row["queries"] for row in rows), 1000)
        self.assertAlmostEqual(sum(row["m"] * row["queries"] for row in rows) / 1000, sum(rr) / 1000, places=12)

    def test_missing_year_duplicate_query_or_misaligned_values_rejected(self):
        with self.assertRaises(ValueError):
            group_by_publication_year(["a", "b"], {"a": 2015}, {"m": [1, 0]})
        with self.assertRaises(ValueError):
            group_by_publication_year(["a", "a"], {"a": 2015}, {"m": [1, 0]})
        with self.assertRaises(ValueError):
            group_by_publication_year(["a", "b"], {"a": 2015, "b": 2016}, {"m": [1]})

    def test_target_year_comes_from_official_target_not_retrieved_documents(self):
        official = pd.DataFrame({"docno": ["t1", "t2", "other"], "web_url": ["u1", "u2", "u3"],
                                 "year": ["2016", "2024", "2019"]})
        known = pd.DataFrame({"qid": ["q1", "q2"], "query": ["x", "y"], "docno": ["t1", "t2"], "year": ["2016", "2024"]})
        self.assertEqual(load_target_years(known, official, range(2015, 2026)), {"q1": 2016, "q2": 2024})
        wrong = known.assign(year=["2019", "2024"])  # The year of a non-target document.
        with self.assertRaisesRegex(ValueError, "disagrees"):
            load_target_years(wrong, official, range(2015, 2026))
        with self.assertRaisesRegex(ValueError, "outside"):
            load_target_years(known, official, range(2005, 2015))
        with self.assertRaisesRegex(ValueError, "target publication year"):
            load_target_years(known.drop(columns="year"), official, range(2015, 2026))

    def test_weighted_slope(self):
        self.assertAlmostEqual(weighted_slope([2015, 2016, 2017], [0., .1, .2], [5, 1, 9]), .1)
        self.assertAlmostEqual(weighted_slope([2015, 2016, 2017], [.3, .3, .3], [1, 1, 1]), 0.)
        self.assertIsNone(weighted_slope([2016], [.4], [2]))


class GeneralBaselineRandomizationTests(unittest.TestCase):
    def test_model_vs_model_baseline_matches_single_pair_test(self):
        import numpy as np
        rng = np.random.default_rng(9)
        m1, m2, m2s = rng.random(40), rng.random(40), rng.random(40)
        batch = paired_randomization_against_bm25({"Dense M1": m1, "Dense M2": m2, "Dense M2s": m2s},
                                                  baseline="Dense M1")
        self.assertEqual([row["method"] for row in batch], ["Dense M2", "Dense M2s"])
        self.assertTrue(all(row["baseline"] == "Dense M1" and "baseline_mean" in row for row in batch))
        single = paired_randomization_test(m2, m1)
        self.assertEqual(batch[0]["p_value"], single["p_value"])
        self.assertAlmostEqual(batch[0]["mean_difference"], float(np.mean(m2 - m1)))
        self.assertEqual(batch[0]["p_value"], reference_randomization_p(m2, m1))
        with self.assertRaises(ValueError):
            paired_randomization_against_bm25({"Dense M2": m2}, baseline="Dense M1")


def write_provenance(runs, models, collections=("C1", "C2")):
    """Training reports, model files and per-run summaries matching PART44_MODELS."""
    models.mkdir(exist_ok=True)
    for model, spec in PART44_MODELS.items():
        if spec["corpus"] is None:
            continue
        (models / spec["model_file"]).write_bytes(f"{model} vectors".encode())
        report = {"model_path": str(models / spec["model_file"]), "corpus_path": f"C:/corpora/{spec['corpus']}"}
        if spec["source_model"]:
            report["source_model_path"] = str(models / spec["source_model"])
        (models / f"{spec['model_file']}.training.json").write_text(json.dumps(report), encoding="utf-8")

    def recorded(model):
        name = PART44_MODELS[model]["model_file"]
        if model == "GN":
            return {name: "f" * 64}
        return {name: sha256_file(models / name), f"{name}.training.json": sha256_file(models / f"{name}.training.json")}

    for collection in collections:
        prefix = collection.lower()
        for model in PART44_MODELS:
            folder = runs / f"dense_{prefix}_{model.lower()}"
            folder.mkdir(exist_ok=True)
            run = runs / f"{prefix}_dense_{model.lower()}.csv"
            (folder / "dense_summary.json").write_text(json.dumps({
                "model": model, "combined_output": {"path": str(run), "sha256": sha256_file(run)},
                "document_embeddings": {"provenance": {"collection": collection, "model": model,
                                                       "model_file_sha256": recorded(model)}}}), encoding="utf-8")
        for model in ("M1", "M2", "GN"):
            folder = runs / f"qe_{prefix}_{model.lower()}"
            folder.mkdir(exist_ok=True)
            (folder / "expansion_summary.json").write_text(json.dumps({
                "model": model, "collection": collection,
                "provenance": {"collection": collection, "model": model, "model_files_sha256": recorded(model)}}),
                encoding="utf-8")


class ModelProvenanceTests(unittest.TestCase):
    def build(self, root):
        runs = root / "runs"
        runs.mkdir()
        for prefix in ("c1", "c2"):
            for model in PART44_MODELS:
                (runs / f"{prefix}_dense_{model.lower()}.csv").write_text(f"{prefix}{model}", encoding="utf-8")
        write_provenance(runs, root / "models")
        return runs, root / "models"

    def test_model_to_period_mapping_verified(self):
        with TemporaryDirectory() as temporary:
            runs, models = self.build(Path(temporary))
            result = verify_model_provenance(runs, models)
            self.assertTrue(result["passed"])
            self.assertEqual(len(result["runs"]), 14)
            self.assertEqual(result["training_reports"]["M1"]["corpus"], "corpus_P1.txt")
            self.assertEqual(result["training_reports"]["M2"]["source_model"], "w2v_P1.model")
            self.assertIsNone(result["training_reports"]["M2s"]["source_model"])
            periods = {row["method"]: row["training_periods"] for row in result["runs"]}
            self.assertEqual(periods, {"Dense M1": "P1", "Dense M2": "P1+P2", "Dense M2s": "P2", "Dense GN": "external",
                                       "QE M1": "P1", "QE M2": "P1+P2", "QE GN": "external"})

    def test_swapped_model_or_wrong_training_corpus_rejected(self):
        with TemporaryDirectory() as temporary:
            runs, models = self.build(Path(temporary))
            path = runs / "dense_c2_m2" / "dense_summary.json"
            payload = json.loads(path.read_text())
            payload["document_embeddings"]["provenance"]["model_file_sha256"] = {"w2v_P2_fresh.model": "x"}
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "did not use"):
                verify_model_provenance(runs, models)
        with TemporaryDirectory() as temporary:
            runs, models = self.build(Path(temporary))
            report = models / "w2v_P2_fresh.model.training.json"
            payload = json.loads(report.read_text())
            payload["source_model_path"] = "w2v_P1.model"  # M2s must be trained from scratch.
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "M2s"):
                verify_model_provenance(runs, models)


class TemporalPipelineTests(unittest.TestCase):
    def test_temporal_analysis_reuses_frozen_qe_and_validated_metrics(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            official, runs, output = Part42PipelineTests().build(root)
            for period, year in (("P1", "2010"), ("P2", "2016")):
                for name in (f"collection_{period}.tsv", f"queries_ki_{period}.tsv"):
                    table = pd.read_csv(official / name, sep="\t", dtype=str)
                    table["year"] = year
                    table.to_csv(official / name, sep="\t", index=False)
            write_provenance(runs, root / "models")
            with patch.dict("src.evaluation.EXPECTED_COLLECTION_COUNTS", {"C1": 105, "C2": 105}), \
                 patch.dict("src.evaluation.TUNING_QUERY_COUNTS", {"known_item": 2, "topics": 2}):
                run_main_results(official, runs, output)
                before = {p: sha256_file(p) for directory in (official, runs, output) for p in directory.iterdir()
                          if p.is_file()}
                with patch("src.evaluation.select_p1_settings", side_effect=AssertionError("Retuning prohibited")), \
                     patch("src.retrieval.run_pipeline", side_effect=AssertionError("Retrieval prohibited")):
                    summary = run_temporal_generalization(official, runs, output, models_dir=root / "models")
                    with self.assertRaises(FileExistsError):
                        run_temporal_generalization(official, runs, output, models_dir=root / "models")
            for path, digest in before.items():
                self.assertEqual(sha256_file(path), digest)
            self.assertTrue(summary["frozen_qe_reuse"]["passed"])
            self.assertTrue(summary["metric_validation"]["passed"])
            self.assertEqual(summary["publication_year"]["groups"], {2016: 2})
            comparison = pd.read_csv(output / "part44_temporal_comparison.csv").set_index(["query_type", "method"])
            self.assertEqual(len(comparison), 16)
            qe = comparison.loc[("known_item", "QE M1")]
            self.assertEqual(qe["source_P2"], "c2_qe_m1_m5_l0p5.csv")  # Frozen P1 setting, not the better P2 oracle.
            self.assertEqual(json.loads(qe["parameters"]), {"m": 5, "lambda": .5})
            self.assertAlmostEqual(qe["P2_C2"], 1 / 3)
            self.assertEqual(qe["P1_C1"], 1.)
            self.assertEqual(comparison.loc[("topics", "QE GN"), "source_P2"], "c2_qe_gn_m10_l1p0.csv")
            dense = comparison.loc[("known_item", "Dense M1")]
            self.assertAlmostEqual(dense["P1_C1"], .25)  # Target at rank 4 in the fixture.
            self.assertAlmostEqual(dense["minus_BM25_P1"], -.25)
            tests = pd.read_csv(output / "part44_paired_tests.csv")
            self.assertEqual(len(tests), 4 * 9)
            self.assertEqual(set(tests["permutations"]), {10_000})
            years = pd.read_csv(output / "part44_ki2_by_year.csv")
            self.assertEqual(years["queries"].sum(), 2)
            self.assertEqual(list(years.columns), ["year", "queries", "MRR@10 BM25", "MRR@10 Dense M1",
                                                   "MRR@10 Dense M2", "MRR@10 Dense GN"])
            self.assertTrue((output / "part44_ki2_mrr_by_year.png").read_bytes().startswith(b"\x89PNG"))
            self.assertEqual(set(summary["findings"]), {"own_training_period", "continued_training_on_P1",
                                                        "M2_vs_M2s", "GN_vs_local", "publication_year"})
            for name, digest in summary["artifact_sha256"].items():
                self.assertEqual(sha256_file(output / name), digest)



class PerQueryAnalysisTests(unittest.TestCase):
    def test_new_vocabulary_queries_use_m2_minus_m1_terms_in_official_order(self):
        m1, m2 = {"trump", "vote", "film"}, {"trump", "vote", "film", "covid", "tiktok"}
        tokens = {"KI2-0003": ["covid", "vote"], "KI2-0001": ["film"], "KI2-0002": ["tiktok", "covid", "tiktok"]}
        hits = find_term_queries(tokens, m2 - m1)
        self.assertEqual(list(hits), ["KI2-0003", "KI2-0002"])
        self.assertEqual(hits["KI2-0002"], ["covid", "tiktok"])
        self.assertEqual(find_term_queries(tokens, set()), {})

    def test_target_rank(self):
        self.assertEqual(target_rank(["a", "t", "b"], "t"), 2)
        self.assertIsNone(target_rank(["a", "b"], "t"))

    def test_subset_and_remainder_are_tested_separately(self):
        import numpy as np
        rng = np.random.default_rng(4)
        qids = [f"q{i}" for i in range(60)]
        m1, m2 = rng.random(60), rng.random(60)
        subset = {f"q{i}" for i in range(0, 60, 2)}
        rows = paired_subset_comparison(qids, subset, list(m1), list(m2))
        self.assertEqual([r["group"] for r in rows], ["subset", "remainder"])
        self.assertEqual(rows[0]["queries"] + rows[1]["queries"], 60)
        inside = [i for i in range(60) if i % 2 == 0]
        expected = paired_randomization_test(m2[inside], m1[inside])
        self.assertEqual(rows[0]["p_value"], expected["p_value"])
        self.assertAlmostEqual(rows[0]["mean_difference"], float(np.mean(m2[inside] - m1[inside])))
        self.assertAlmostEqual(rows[1]["method_mean"], float(np.mean(m2[1::2])))
        self.assertEqual(rows[0]["improved"] + rows[0]["worse"] + rows[0]["tied"], 30)
        self.assertEqual(rows[0]["permutations"], 10_000)
        self.assertEqual(rows[0]["random_seed"], 42)

    def test_empty_and_small_groups_are_never_called_significant(self):
        qids = [f"q{i}" for i in range(40)]
        m1, m2 = [0.] * 40, [1.] * 40
        empty, rest = paired_subset_comparison(qids, set(), m1, m2)
        self.assertEqual(empty["queries"], 0)
        self.assertIsNone(empty["p_value"])
        self.assertFalse(empty["significant"])
        self.assertTrue(rest["significant"])
        small, _ = paired_subset_comparison(qids, {f"q{i}" for i in range(12)}, m1, m2)
        self.assertTrue(small["small_group"])
        self.assertTrue(small["raw_p_below_005"])  # Raw p is reported for 12 queries...
        self.assertFalse(small["significant"])      # ...but no significance conclusion is drawn.
        with self.assertRaises(ValueError):
            paired_subset_comparison(qids, {"foreign"}, m1, m2)

    def test_extreme_examples_order_ties_and_exclude_zero(self):
        rows = [{"qid": q, "d": d} for q, d in [("b", 1.), ("a", 1.), ("c", .5), ("d", 0.), ("e", -1.), ("f", -.2)]]
        up, down = extreme_examples(rows, "d", k=2)
        self.assertEqual([r["qid"] for r in up], ["a", "b"])
        self.assertEqual([r["qid"] for r in down], ["e", "f"])
        self.assertEqual(extreme_examples(rows, "d", k=5)[0][-1]["qid"], "c")

    def test_qe_expansion_matches_query_expander_and_frozen_topic_settings(self):
        from src.retrieval import QueryExpander
        record = {"tokens": ["football"], "fallback": False,
                  "neighbors": [{"term": f"t{i:02d}", "cosine": .9 - i / 100} for i in range(20)]}
        expander = QueryExpander(model=None, collection_vocabulary=[])
        expander.cache[("football",)] = record
        frozen = frozen_part42_selection()
        params = {row["method"]: row["parameters"] for row in part42_method_plan(frozen, "C2", "topics")}
        self.assertEqual(params["QE M2"], {"m": 10, "lambda": 1.})  # P1 topic choice, reused on C2.
        for model in ("QE M1", "QE M2"):
            m, weight = params[model]["m"], params[model]["lambda"]
            ours = qe_expansion(record, m, weight)
            weights, details = expander.expand("football", m, weight)
            self.assertEqual({e["term"]: e["weight"] for e in ours},
                             {k: v for k, v in weights.items() if k != "football"})
            self.assertEqual([e["term"] for e in ours], [e["term"] for e in details["expansion"]])
        self.assertEqual(qe_expansion({**record, "fallback": True, "neighbors": []}, 5, .5), [])
        with self.assertRaises(ValueError):
            qe_expansion(record, 7, .5)

    def test_term_statistics_from_postings(self):
        from scipy import sparse
        # Rows are documents, columns are terms (CSC postings, as in the saved BM25 index).
        dense = np.array([[1, 0, 2], [0, 1, 1], [3, 0, 0], [0, 0, 1]], dtype=float)
        weights = sparse.csc_matrix(dense)
        vocabulary = {"a": 0, "b": 1, "c": 2}
        stats = term_relevance_stats(weights, vocabulary, ["a", "b", "c", "zzz"], relevant_rows=[0, 2])
        self.assertEqual(stats["a"], {"c2_df": 2, "relevant_df": 2, "relevant_fraction": 1.})
        self.assertEqual(stats["c"]["c2_df"], 3)
        self.assertAlmostEqual(stats["c"]["relevant_fraction"], 1 / 3)
        self.assertIsNone(stats["zzz"]["relevant_fraction"])
        expansion = [{"term": "a", "weight": .5}, {"term": "c", "weight": 1.}, {"term": "zzz", "weight": .4}]
        self.assertAlmostEqual(weighted_relevant_fraction(expansion, stats), (.5 * 1 + 1 * (1 / 3)) / 1.5)
        self.assertAlmostEqual(pooled_relevant_fraction(expansion, stats), (.5 * 2 + 1 * 1) / (.5 * 2 + 1 * 3))
        precise, noisy = classify_expansion_terms(["a", "c"], {
            "a": {"c2_df": 50, "relevant_df": 45, "relevant_fraction": .9},
            "c": {"c2_df": 900, "relevant_df": 90, "relevant_fraction": .1}})
        self.assertEqual((precise, noisy), (["a"], ["c"]))

    def test_rederived_fusion_order_matches_part34_writer(self):
        from src.retrieval import linear_fusion, rank_scores
        rng = np.random.default_rng(8)
        docnos = np.asarray(sorted(f"C2-{i:06d}" for i in rng.choice(9999, 150, replace=False)))
        bm25, dense = rng.random(150).round(2), rng.random(150).round(2)  # Rounding creates score ties.
        for alpha in (.3, .5, .7):
            saved, _ = rank_scores(linear_fusion(bm25, dense, alpha), docnos, np.arange(150))
            ours = fused_ranking(docnos, bm25, dense, alpha)
            self.assertEqual(list(ours["docnos"][:100]), list(saved))
            self.assertTrue(np.all(np.diff(ours["fused"]) <= 0))
        with self.assertRaises(ValueError):
            fused_ranking(docnos[::-1], bm25, dense, .5)

    def test_illustrative_alpha_prefers_half_then_nearest(self):
        self.assertEqual(choose_illustrative_alpha({.5: (3, 9), .4: (10, 10)}), .5)
        self.assertEqual(choose_illustrative_alpha({.5: (2, 9), .4: (1, 9), .6: (5, 3)}), .6)
        with self.assertRaises(ValueError):
            choose_illustrative_alpha({.5: (2, 2), 1.: (0, 0)})

    def test_fusion_reason_reports_the_deciding_arithmetic(self):
        target = {"fused_score": .97, "bm25_normalized": .94, "dense_normalized": 1.}
        top = {"fused_score": .886, "bm25_normalized": 1., "dense_normalized": .772}
        text = fusion_reason("hybrid_beats_BM25", target, top, .5, 1)
        self.assertIn("-0.060", text)
        self.assertIn("+0.228", text)
        self.assertIn("= +0.084", text)
        loss = fusion_reason("hybrid_loses_to_BM25", top, target, .5, 15)
        self.assertIn("14 documents now rank above", loss)
        self.assertIn("outside the fusion candidate union",
                      fusion_reason("hybrid_loses_to_BM25", {"fused_score": None}, top, .5, None))

    def test_refuses_to_overwrite_part45_outputs(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary) / "part4"
            out.mkdir()
            (out / "part45_summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_per_query_analysis(Path(temporary), Path(temporary), out)


if __name__ == "__main__":
    unittest.main()
