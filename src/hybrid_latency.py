"""Section 3.5: measured END-TO-END hybrid query latency from saved artifacts.

One timed call per query runs every online step once: query embedding, full BM25
first stage, full dense first stage, candidate union, candidate-score lookup,
fusion, and top-100 ranking. Saved indexes, document vectors and models are read
only; no rankings are written. Untimed: hashing/loading, warm-up, comparison with
saved runs, and JSON writing.

Run from the repository root, writing to a NEW file outside the repository::

    python -m src.hybrid_latency --collection C1 --model M1 \
        --embedding-model ../data/models/w2v_P1.model \
        --output ../data/part3_latency_e2e/c1_m1.json
"""

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from .preprocessing import preprocess_text
from .retrieval import (
    EXPECTED_COLLECTION_COUNTS, FUSION_ALPHAS, REPOSITORY, embed_query_batch, ensure_memory,
    external_path, fusion_candidates, linear_fusion, load_dense_index, load_index, load_rankings,
    memory_snapshot, model_parameters, rank_scores, read_table, rrf_scores, sha256_file, write_json,
)

STAGES = ("total", "query_embedding", "bm25_first_stage", "dense_first_stage", "union_fusion_ranking")


def load_artifacts(official_dir, index_dir, collection, model_label, model_path):
    """Mirror run_hybrid_pipeline's identity checks; build nothing."""
    period = {"C1": "P1", "C2": "P2"}[collection]
    collection_path = official_dir / f"collection_{period}.tsv"
    official = read_table(collection_path, ["docno", "web_url"], "docno")
    known = read_table(official_dir / f"queries_ki_{period}.tsv", ["qid", "query"], "qid")
    topics = read_table(official_dir / "topics.tsv", ["tid", "query"], "tid").rename(columns={"tid": "qid"})
    if len(official) != EXPECTED_COLLECTION_COUNTS[collection] or len(known) != 1000 or len(topics) != 32:
        raise ValueError("Unexpected official collection or query counts")
    queries = pd.concat([known[["qid", "query"]], topics[["qid", "query"]]], ignore_index=True)
    bm25_dir, tfidf_dir = index_dir / f"{collection}_bm25", index_dir / f"{collection}_tfidf"
    dense_dir = index_dir / f"{collection}_{model_label}"
    bm25_meta = json.loads((bm25_dir / "index.json").read_text(encoding="utf-8"))
    dense_meta = json.loads((dense_dir / "dense_index.json").read_text(encoding="utf-8"))
    lexical, provenance = bm25_meta["provenance"], dense_meta["provenance"]
    preprocessing_hash = sha256_file(REPOSITORY / "src/preprocessing.py")
    if (lexical["smoke"] or provenance["smoke"] or lexical["collection"] != collection
            or lexical["collection_sha256"] != sha256_file(collection_path)
            or lexical["preprocessing_sha256"] != preprocessing_hash
            or provenance["preprocessing_sha256"] != preprocessing_hash
            or lexical["model_parameters"] != model_parameters("bm25")
            or provenance["collection"] != collection or provenance["model"] != model_label
            or provenance["idf_vectorizer_sha256"] != sha256_file(tfidf_dir / "vectorizer.joblib")
            or not dense_meta["normalized"]):
        raise ValueError("Saved BM25/dense indexes do not match current inputs or preprocessing")
    model_files = [model_path, *sorted(model_path.parent.glob(model_path.name + ".*"))]
    if {p.name: sha256_file(p) for p in model_files if p.is_file()} != provenance["model_file_sha256"]:
        raise ValueError("Embedding model differs from the one used for the saved dense index")
    if model_label == "GN" and (dense_meta["google_news"] or not provenance.get("gn_preprocessed")):
        raise ValueError("GN requires the directly keyed local subset")
    ensure_memory(2 * model_path.stat().st_size + dense_meta["document_matrix_bytes"])
    bm25 = load_index(bm25_dir, lexical, "bm25")
    vectorizer = joblib.load(tfidf_dir / "vectorizer.joblib")
    if bm25.vectorizer.analyzer is not preprocess_text or vectorizer.analyzer is not preprocess_text:
        raise ValueError("Indexes must use the shared preprocess_text analyzer")
    if model_label == "GN":
        from .google_news import load_gn_vectors
        model = load_gn_vectors(model_path, model_format="keyedvectors")
    else:
        from gensim.models import Word2Vec
        model = Word2Vec.load(str(model_path)).wv
    dense = load_dense_index(dense_dir, provenance, model, vectorizer)  # also verifies matrix checksum
    if not np.array_equal(bm25.docnos, dense.docnos) or set(bm25.docnos) != set(official["docno"]):
        raise ValueError("BM25 and dense rows must describe the same documents in the same order")
    return official, queries, bm25, dense


def hybrid_search(query, bm25, dense, row_of, method, alpha=None):
    """One complete hybrid query; every operation runs exactly once."""
    t0 = perf_counter()
    vectors, _ = embed_query_batch([query], dense.model, dense.tfidf_vectorizer,
                                   dense.idf_document_count, google_news=dense.google_news)
    t1 = perf_counter()
    # Same arithmetic as RetrievalIndex.search: distinct terms, weight 1, all postings.
    bm25_all = np.zeros(len(bm25.docnos), dtype=np.float64)
    for term in bm25.vectorizer.transform([query]).tocsr().indices:
        start, stop = bm25.weights.indptr[term:term + 2]
        bm25_all[bm25.weights.indices[start:stop]] += bm25.weights.data[start:stop] * 1.0
    bm25_top = rank_scores(bm25_all, bm25.docnos, bm25.tie_order)
    t2 = perf_counter()
    # Same arithmetic as DenseIndex.search_batch, for a batch of one query.
    dense_all = np.asarray(vectors[:1] @ dense.documents.T)[0]
    np.clip(dense_all, -1, 1, out=dense_all)
    dense_top = rank_scores(dense_all, dense.docnos, dense.tie_order)
    t3 = perf_counter()
    candidates = fusion_candidates(bm25_top[0], dense_top[0])  # sorted, so tie order is arange
    if method == "rrf":
        fused = rrf_scores(candidates, bm25_top[0], dense_top[0])
    else:
        rows = np.fromiter((row_of[d] for d in candidates), dtype=np.int64, count=len(candidates))
        fused = linear_fusion(bm25_all[rows], dense_all[rows], alpha)
    final = rank_scores(fused, candidates, np.arange(len(candidates)))
    t4 = perf_counter()
    return final, bm25_top, dense_top, (t4 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3)


def run_pass(texts, bm25, dense, row_of, method, alpha):
    times, outputs = np.empty((len(texts), len(STAGES))), []
    for i, text in enumerate(texts):
        final, bm25_top, dense_top, timing = hybrid_search(text, bm25, dense, row_of, method, alpha)
        times[i] = timing  # bookkeeping happens after the timed interval
        outputs.append((final, bm25_top, dense_top))
    return times, outputs


def describe(ms):
    return {"mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)), "min_ms": float(ms.min()), "max_ms": float(ms.max())}


def compare(rankings, qids, saved):
    """Agreement with a saved run: order/set differences, max score error where order agrees."""
    order = sets = 0
    error = 0.0
    for qid, (docnos, scores) in zip(qids, rankings):
        ref_docs, ref_scores = saved[qid]
        same = np.array_equal(docnos, ref_docs)
        order += not same
        sets += set(docnos) != set(ref_docs)
        if same:
            error = max(error, float(np.max(np.abs(np.asarray(scores, dtype=np.float64) - ref_scores))))
    return {"queries": len(qids), "order_differences": order, "set_differences": sets,
            "max_abs_score_error_when_order_agrees": error}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", choices=["C1", "C2"], required=True)
    parser.add_argument("--model", choices=["M1", "M2", "GN"], required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, default=Path("../data/nyt-tc-2026"))
    parser.add_argument("--index-dir", type=Path, default=Path("../data/part3_indexes"))
    parser.add_argument("--input-results-dir", type=Path, default=Path("../data/part3_results"))
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the repository")
    parser.add_argument("--alpha", type=float, choices=FUSION_ALPHAS, default=0.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="3 known-item + 3 topic queries, one repeat")
    args = parser.parse_args(argv)
    output = external_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    official_dir, index_dir = args.official_dir.resolve(), args.index_dir.resolve()
    results_dir, model_path = args.input_results_dir.resolve(), args.embedding_model.resolve()
    memory_before = memory_snapshot()
    started = perf_counter()
    official, queries, bm25, dense = load_artifacts(official_dir, index_dir, args.collection, args.model, model_path)
    loading_seconds = perf_counter() - started
    selected = pd.concat([queries.iloc[:3], queries.iloc[1000:1003]]) if args.smoke else queries
    qids, texts = selected["qid"].tolist(), selected["query"].tolist()
    methods = [("linear", args.alpha), ("rrf", None)]
    timings, first = {m: [] for m, _ in methods}, {}
    try:
        row_of = dict(zip(bm25.docnos, range(len(bm25.docnos))))
        for text in texts[:args.warmup]:  # untimed: pages in matrix/postings, starts BLAS threads
            for method, alpha in methods:
                hybrid_search(text, bm25, dense, row_of, method, alpha)
        for repeat in range(1 if args.smoke else args.repeats):
            for method, alpha in (methods if repeat % 2 == 0 else methods[::-1]):  # alternate pass order
                times, outputs = run_pass(texts, bm25, dense, row_of, method, alpha)
                timings[method].append(times)
                first.setdefault(method, outputs)
        memory_after = memory_snapshot()
    finally:
        dense.close()
    # Untimed comparison with the saved runs, which are never rewritten.
    prefix, label = args.collection.lower(), args.model.lower()
    alpha_code = f"a{args.alpha:.1f}".replace(".", "p")
    files = {"bm25": f"{prefix}_bm25.csv", "dense": f"{prefix}_dense_{label}.csv",
             "linear": f"{prefix}_hybrid_{label}_{alpha_code}.csv", "rrf": f"{prefix}_rrf_{label}.csv"}
    saved = {k: load_rankings(results_dir / f, queries, official["docno"]) for k, f in files.items()}
    validation = {
        "bm25_first_stage": compare([o[1] for o in first["linear"]], qids, saved["bm25"]),
        "dense_first_stage": compare([o[2] for o in first["linear"]], qids, saved["dense"]),
        "linear": compare([o[0] for o in first["linear"]], qids, saved["linear"]),
        "rrf": compare([o[0] for o in first["rrf"]], qids, saved["rrf"]),
    }
    from threadpoolctl import threadpool_info
    summary = {
        "collection": args.collection, "model": args.model, "smoke": args.smoke,
        "queries": len(texts), "query_ids": qids if args.smoke else "known-item then topics, official order",
        "batch_size": 1, "warmup_queries_untimed": min(args.warmup, len(texts)),
        "repeats": len(timings["linear"]),
        "methods": {
            method: {"alpha": alpha, "file_compared": files[method],
                     "per_repeat_total": [describe(t[:, 0] * 1e3) for t in timings[method]],
                     "pooled_total": describe(np.concatenate([t[:, 0] for t in timings[method]]) * 1e3),
                     "mean_stage_ms": {s: float(np.concatenate([t[:, i] for t in timings[method]]).mean() * 1e3)
                                       for i, s in enumerate(STAGES)},
                     "per_query_total_ms": [(t[:, 0] * 1e3).tolist() for t in timings[method]]}
            for method, alpha in methods},
        "timing_includes": ["query embedding (once)", "full BM25 first stage + top-100",
                            "full dense cosine first stage + top-100", "candidate union",
                            "candidate score lookup (linear) or RRF", "min-max + fusion", "final top-100 ranking"],
        "timing_excludes": ["hashing and loading", "warm-up", "comparison with saved runs", "JSON writing"],
        "validation_against_saved_runs": validation,
        "loading_seconds_untimed": loading_seconds,
        "environment": {"platform": platform.platform(), "logical_cpus": os.cpu_count(),
                        "python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                        "threadpools": threadpool_info(),
                        "thread_env": {k: os.environ.get(k) for k in
                                       ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}},
        "memory_before": memory_before, "memory_after": memory_after,
        "source_sha256": {f: sha256_file(results_dir / f) for f in files.values()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, summary)  # exclusive creation; never overwrites
    for method in timings:
        print(f"{args.collection}/{args.model} {method}: "
              f"{summary['methods'][method]['pooled_total']['mean_ms']:.3f} ms mean end-to-end")
    print("Validation:", json.dumps(validation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
