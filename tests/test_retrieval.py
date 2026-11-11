"""Focused synthetic checks for Parts 3.1-3.4; assignment data are unchanged."""

import contextlib
import csv
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

from src.document_collections import clean_articles
from src.preprocessing import preprocess_text
from src.retrieval import (
    bm25_weights, build_index, ensure_memory, external_path, load_index, matched_texts,
    publish_summary, read_table, run_pipeline, run_queries, save_index, validate_results,
    DenseIndex, build_dense_documents, embed_query_batch, load_dense_index,
    normalize_embeddings, run_dense_pipeline, run_dense_queries, sha256_file, weighted_average_embeddings,
    QueryExpander, expansion_centroid, model_parameters, run_expansion_pipeline,
    FUSION_ALPHAS, fusion_candidates, linear_fusion, minmax_scores, rrf_scores,
    score_fusion_candidates,
)


class RetrievalTests(unittest.TestCase):
    def test_tfidf_sublinear_cosine_and_same_query_analyzer(self):
        index = build_index("tfidf", ["alpha alpha beta", "beta gamma", "alpha gamma gamma"],
                            ["d0", "d1", "d2"])
        self.assertIs(index.vectorizer.analyzer, preprocess_text)
        self.assertTrue(index.vectorizer.sublinear_tf)
        self.assertTrue(sparse.isspmatrix_csc(index.weights))
        docnos, scores = index.search("ALPHA alpha", 3)
        expected = {"d0": (1 + math.log(2)) / math.sqrt((1 + math.log(2)) ** 2 + 1),
                    "d1": 0, "d2": 1 / math.sqrt(1 + (1 + math.log(2)) ** 2)}
        for docno, score in zip(docnos, scores):
            self.assertAlmostEqual(score, expected[docno], places=12)
        self.assertEqual(docnos.tolist(), ["d0", "d2", "d1"])
        # BM25's distinct-term rule must not change TF-IDF's query TF weighting.
        original = index.search("alpha beta", 3)
        repeated = index.search("alpha alpha beta", 3)
        original_scores = dict(zip(*original))
        repeated_scores = dict(zip(*repeated))
        self.assertNotAlmostEqual(original_scores["d0"], repeated_scores["d0"], places=12)

    def test_shared_preprocessing_preserves_stopwords_numbers_and_acronyms(self):
        for model in ["tfidf", "bm25"]:
            with self.subTest(model=model):
                index = build_index(model, ["The U.S. team's 2025 plan", "another story"], ["d1", "d2"])
                self.assertEqual(set(index.vectorizer.vocabulary_),
                                 {"the", "us", "team", "2025", "plan", "another", "story"})
                self.assertEqual(index.search("U.S. TEAM'S", 1)[0].tolist(), ["d1"])
                self.assertNotIn("missing", index.vectorizer.vocabulary_)
                index.search("missing", 1)
                self.assertNotIn("missing", index.vectorizer.vocabulary_)

    def test_bm25_against_appendix_formula_with_distinct_query_terms(self):
        docs = ["alpha alpha beta", "beta gamma", "alpha"]
        index = build_index("bm25", docs, ["d0", "d1", "d2"])
        docnos, scores = index.search("alpha alpha beta", 3)
        expected = {}
        for i, document in enumerate(docs):
            words = document.split()
            value = 0
            for term in ["alpha", "beta"]:
                tf = words.count(term)
                df = sum(term in d.split() for d in docs)
                value += math.log(1 + (3 - df + .5) / (df + .5)) * tf * 2.2 / (
                    tf + 1.2 * (.25 + .75 * len(words) / 2))
            expected[f"d{i}"] = value
        for docno, score in zip(docnos, scores):
            self.assertAlmostEqual(score, expected[docno], places=12)
        self.assertEqual(docnos.tolist(), sorted(expected, key=lambda d: (-expected[d], d)))
        without_repetition = index.search("alpha beta", 3)
        np.testing.assert_array_equal(docnos, without_repetition[0])
        np.testing.assert_array_equal(scores, without_repetition[1])
        self.assertAlmostEqual(index.metadata["average_document_length"], 2)

    def test_bm25_idf_uses_document_frequency_and_keeps_common_terms_positive(self):
        # Every document has length 2 and tf=1, so the TF/length factor is 1.
        counts = sparse.csr_matrix([[1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 0]])
        weights, average = bm25_weights(counts)
        self.assertEqual(average, 2)
        for term, df in enumerate([4, 2, 2]):
            start, stop = weights.indptr[term:term + 2]
            expected_idf = math.log(1 + (4 - df + .5) / (df + .5))
            np.testing.assert_allclose(weights.data[start:stop], expected_idf, rtol=1e-14)
            self.assertGreater(expected_idf, 0)

    def test_bm25_retains_document_tf_and_penalizes_unrelated_document_length(self):
        documents = ["apple", "apple apple", "apple filler filler filler", "banana"]
        index = build_index("bm25", documents, ["single", "double", "long", "absent"])
        self.assertEqual(index.metadata["average_document_length"], 2)
        apple = index.vectorizer.vocabulary_["apple"]
        counts = index.vectorizer.transform(documents)
        self.assertEqual(int(counts[1, apple]), 2)
        docnos, scores = index.search("APPLE apple", 4)
        values = dict(zip(docnos, scores))
        self.assertGreater(values["double"], values["single"])
        self.assertGreater(values["single"], values["long"])
        self.assertEqual(values["absent"], 0)
        idf = math.log(1 + (4 - 3 + .5) / (3 + .5))
        expected_double = idf * (2 * 2.2) / (2 + 1.2 * (.25 + .75 * 2 / 2))
        self.assertAlmostEqual(values["double"], expected_double, places=12)
        np.testing.assert_array_equal(index.search("apple", 4)[1], scores)

    def test_bm25_collection_statistics_and_empty_documents(self):
        weights, average = bm25_weights(sparse.csr_matrix([[1, 0], [0, 0], [0, 2]]))
        self.assertEqual(average, 1)
        self.assertEqual(weights.getrow(1).nnz, 0)
        self.assertTrue(np.isfinite(weights.data).all())
        index1 = build_index("tfidf", ["shared", "shared"], ["a", "b"])
        index2 = build_index("tfidf", ["shared", "different"], ["a", "b"])
        self.assertLess(index1.vectorizer.idf_[index1.vectorizer.vocabulary_["shared"]],
                        index2.vectorizer.idf_[index2.vectorizer.vocabulary_["shared"]])

    def test_bm25_canonicalizes_counts_without_mutating_inputs(self):
        counts = sparse.csr_matrix(([1, 2, 0, 1], [0, 0, 1, 1], [0, 3, 4]), shape=(2, 2))
        before = counts.data.copy()
        actual, average = bm25_weights(counts)
        expected, _ = bm25_weights(sparse.csr_matrix([[3, 0], [0, 1]]))
        self.assertEqual(average, 2)
        np.testing.assert_array_equal(actual.indptr, expected.indptr)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.data, expected.data)
        np.testing.assert_array_equal(counts.data, before)
        for matrix in [[[-1]], [[np.nan]], [[.5]]]:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                bm25_weights(sparse.csr_matrix(matrix))
        for k1, b in [(0, .75), (1.2, 2), (float("nan"), .75)]:
            with self.subTest(k1=k1, b=b), self.assertRaises(ValueError):
                bm25_weights(sparse.csr_matrix([[1]]), k1=k1, b=b)

    def test_independent_collections_have_distinct_bm25_statistics_and_vocabularies(self):
        first = build_index("bm25", ["alpha alpha", "alpha beta"], ["C1-000000", "C1-000001"])
        second = build_index("bm25", ["alpha gamma", "gamma gamma gamma gamma gamma"],
                             ["C2-000000", "C2-000001"])
        self.assertNotIn("gamma", first.vectorizer.vocabulary_)
        self.assertNotIn("beta", second.vectorizer.vocabulary_)
        self.assertEqual(first.metadata["average_document_length"], 2)
        self.assertEqual(second.metadata["average_document_length"], 3.5)
        old = first.search("alpha", 2)
        second.search("alpha gamma", 2)
        after = first.search("alpha", 2)
        np.testing.assert_array_equal(old[0], after[0])
        np.testing.assert_array_equal(old[1], after[1])

    def test_top_100_zero_scores_ties_and_no_dense_matrix_conversion(self):
        docnos = [f"C1-{i:06d}" for i in range(104, -1, -1)]
        for model in ["tfidf", "bm25"]:
            index = build_index(model, ["same text"] * 105, docnos)
            with patch.object(sparse.csc_matrix, "toarray", side_effect=AssertionError("dense")), \
                 patch.object(sparse.csc_matrix, "todense", side_effect=AssertionError("dense")):
                for query in ["", "outofvocabulary", "same"]:
                    result, scores = index.search(query)
                    self.assertEqual(result.tolist(), sorted(docnos)[:100])
                    self.assertEqual(len(set(result)), 100)
                    self.assertTrue(np.all(np.diff(scores) <= 0))
                    if query != "same":
                        self.assertTrue(np.all(scores == 0))
            with self.assertRaises(ValueError):
                index.search("same", 106)

    def test_persistence_reuses_identical_scores_and_rejects_wrong_inputs(self):
        with TemporaryDirectory() as temporary:
            for model in ["tfidf", "bm25"]:
                index = build_index(model, ["first first word", "second word"], ["d1", "d2"])
                path = Path(temporary) / model
                save_index(path, index, {"source": "fixture"})
                loaded = load_index(path, {"source": "fixture"}, model)
                before = index.search("first word", 2)
                after = loaded.search("first word", 2)
                np.testing.assert_array_equal(before[0], after[0])
                np.testing.assert_array_equal(before[1], after[1])
                if model == "bm25":
                    repeated = loaded.search("FIRST first word", 2)
                    np.testing.assert_array_equal(after[0], repeated[0])
                    np.testing.assert_array_equal(after[1], repeated[1])
                with self.assertRaises(ValueError):
                    load_index(path, {"source": "changed"}, model)
                with self.assertRaises(FileExistsError):
                    save_index(path, index, {"source": "fixture"})

    def test_streamed_join_matches_global_part1_cleaning_across_chunk_boundaries(self):
        def text(label):
            return label + " one two three four five six seven eight nine ten"
        raw = pd.DataFrame({
            "web_url": ["outside", "u1", "u1", "u2", "u3", "u2", "u3"],
            "text": [text("duplicate"), text("first"), text("discarded"),
                     text("discarded"), "short", text("valid"), text("third")],
        })
        collection = pd.DataFrame({"docno": ["d3", "d2", "d1"], "web_url": ["u3", "u2", "u1"]})
        expected = clean_articles(raw)[0].set_index("web_url")["text"]
        with TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "articles.h5"
            raw.to_hdf(dataset, key="data", format="table")
            for chunk_size in [1, 2, 3, 20]:
                docnos, details = [], {}
                values = list(matched_texts(dataset, collection, docnos, details, chunk_size=chunk_size))
                self.assertEqual(dict(zip(docnos, values)),
                                 {r.docno: expected[r.web_url] for r in collection.itertuples()})
                self.assertEqual(details["missing_article_texts"], 0)
                self.assertEqual(len(docnos), 3)

    def test_missing_text_and_duplicate_official_urls_fail_explicitly(self):
        with TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "articles.h5"
            pd.DataFrame({"web_url": ["u1"], "text": ["short"]}).to_hdf(dataset, key="data", format="table")
            collection = pd.DataFrame({"docno": ["d1"], "web_url": ["u1"]})
            with self.assertRaisesRegex(ValueError, "Missing cleaned article texts"):
                list(matched_texts(dataset, collection, [], {}))
            with self.assertRaisesRegex(ValueError, "duplicate URLs"):
                list(matched_texts(dataset, pd.concat([collection, collection]), [], {}))

    def test_csv_schema_actual_scores_and_unique_ranks(self):
        docnos = [f"d{i:03d}" for i in range(101)]
        index = build_index("tfidf", ["alpha"] + ["beta"] * 100, docnos)
        queries = pd.DataFrame({"qid": ["q1", "q2"], "query": ["alpha", "unknown"]})
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.csv"
            metrics = run_queries(index, queries, path)
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["qid", "docno", "rank", "score"])
                rows = list(reader)
            self.assertEqual(len(rows), 200)
            self.assertEqual(rows[0], {"qid": "q1", "docno": "d000", "rank": "1", "score": "1"})
            self.assertEqual([int(r["rank"]) for r in rows[:100]], list(range(1, 101)))
            self.assertEqual(len({r["docno"] for r in rows[:100]}), 100)
            self.assertTrue(metrics["all_queries_returned_100"])
            self.assertEqual(metrics["zero_score_queries"], 1)
            with self.assertRaises(FileExistsError):
                run_queries(index, queries, path)

    def test_result_validator_rejects_missing_values_bad_identifiers_ranks_and_scores(self):
        rows = [["q1", "a", 1, 3], ["q1", "b", 2, 2], ["q1", "c", 3, 0],
                ["q2", "d", 1, 1], ["q2", "c", 2, 0], ["q2", "a", 3, 0]]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.csv"

            def write(records):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["qid", "docno", "rank", "score"])
                    writer.writerows(records)

            write(rows)
            self.assertTrue(validate_results(path, ["q1", "q2"], ["a", "b", "c", "d"], 3)["passed"])
            corruptions = [(0, 0, ""), (0, 0, "unknown"), (1, 1, "a"),
                           (1, 1, "outside"), (1, 2, 3), (1, 3, 4), (1, 3, "nan")]
            for i, column, value in corruptions:
                bad = [r.copy() for r in rows]
                bad[i][column] = value
                write(bad)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    validate_results(path, ["q1", "q2"], ["a", "b", "c", "d"], 3)
            for bad in [rows[:-1], rows[:3], rows + rows[:1]]:
                write(bad)
                with self.assertRaises(ValueError):
                    validate_results(path, ["q1", "q2"], ["a", "b", "c", "d"], 3)

    def test_memory_guard_reserves_headroom_and_allocation_budget(self):
        with patch("src.retrieval.memory_snapshot", return_value={"system_available_bytes": 128 * 1024**2}):
            with self.assertRaises(MemoryError):
                ensure_memory()
        with patch("src.retrieval.memory_snapshot", return_value={"system_available_bytes": 1024**3}):
            ensure_memory(16 * 1024**2)
            with self.assertRaises(MemoryError):
                ensure_memory(1024**3)
        with patch("src.retrieval.memory_snapshot", return_value={"system_available_bytes": None}):
            ensure_memory()

    def test_memory_failure_does_not_publish_partial_results(self):
        index = build_index("tfidf", ["alpha"] * 100, [f"d{i}" for i in range(100)])
        queries = pd.DataFrame({"qid": ["q"], "query": ["alpha"]})
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.csv"
            with patch.object(index, "search", side_effect=MemoryError("synthetic")) as search:
                with self.assertRaises(MemoryError):
                    run_queries(index, queries, path)
                search.assert_called_once()
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_summary_publication_rejects_a_concurrent_change(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "retrieval_summary.json"
            path.write_text('{"state":"initial"}', encoding="utf-8")
            expected = path.read_bytes()
            path.write_text('{"state":"another writer"}', encoding="utf-8")
            changed = path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                publish_summary(path, {"state": "new"}, expected)
            self.assertEqual(path.read_bytes(), changed)
            self.assertEqual(list(Path(temporary).iterdir()), [path])
            publish_summary(path, {"state": "new"}, changed)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"state": "new"})

    def test_end_to_end_smoke_and_cached_second_run(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            official = root / "official"
            official.mkdir()
            articles = []
            for period, name in [("P1", "C1"), ("P2", "C2")]:
                ids = [f"{name}-{i:06d}" for i in range(105)]
                urls = [f"https://example/{name}/{i}" for i in range(105)]
                articles.extend({"web_url": url, "text": f"{name} article {i} with alpha beta gamma one two three four five"}
                                for i, url in enumerate(urls))
                pd.DataFrame({"docno": ids, "web_url": urls}).to_csv(official / f"collection_{period}.tsv", sep="\t", index=False)
                pd.DataFrame({"qid": [f"KI{period}-{i}" for i in range(1000)],
                              "query": ["alpha beta"] * 1000}).to_csv(official / f"queries_ki_{period}.tsv", sep="\t", index=False)
            pd.DataFrame({"tid": [f"T{i:02d}" for i in range(32)], "query": ["gamma"] * 32}).to_csv(official / "topics.tsv", sep="\t", index=False)
            dataset = root / "articles.h5"
            pd.DataFrame(articles).to_hdf(dataset, key="data", format="table")
            arguments = (dataset, official, root / "indexes")
            with patch.dict("src.retrieval.EXPECTED_COLLECTION_COUNTS", {"C1": 105, "C2": 105}), \
                 contextlib.redirect_stdout(io.StringIO()):
                first = run_pipeline(*arguments, root / "results", smoke=True, chunk_size=11)
                with patch("src.retrieval.matched_texts", side_effect=AssertionError("unexpected rebuild")):
                    second = run_pipeline(*arguments, root / "results_cached", smoke=True)
                    before = run_pipeline(*arguments, root / "appended", smoke=True,
                                          collection_names=["C1"], combined_output=True)
                    c1_bytes = {p: p.read_bytes() for p in (root / "appended").glob('c1_*.csv')}
                    appended = run_pipeline(*arguments, root / "appended", smoke=True,
                                            collection_names=["C2"], combined_output=True,
                                            append_results=True)
                    self.assertEqual(appended["collections"], ["C1", "C2"])
                    self.assertEqual(len(appended["runs"]), 4)
                    self.assertEqual(appended["runs"][:2], before["runs"])
                    for key, value in before["indexes"].items():
                        self.assertEqual(appended["indexes"][key], value)
                    for key in ["memory_before", "memory_after"]:
                        self.assertEqual(appended[key], before[key])
                    for path, content in c1_bytes.items():
                        self.assertEqual(path.read_bytes(), content)
                    summary_path = root / "appended/retrieval_summary.json"
                    summary_bytes = summary_path.read_bytes()
                    with self.assertRaisesRegex(ValueError, "existing collection"):
                        run_pipeline(*arguments, root / "appended", smoke=True,
                                     collection_names=["C1"], combined_output=True,
                                     append_results=True)
                    self.assertEqual(summary_path.read_bytes(), summary_bytes)
                    # C1-only execution must not read or depend on C2 input files.
                    (official / "collection_P2.tsv").unlink()
                    (official / "queries_ki_P2.tsv").unlink()
                    c1_only = run_pipeline(*arguments, root / "results_c1", smoke=True,
                                           collection_names=["C1"])
                    self.assertEqual(set(c1_only["indexes"]), {"C1_tfidf", "C1_bm25"})
                    self.assertEqual(len(c1_only["runs"]), 4)
                    self.assertEqual(set(c1_only["query_file_sha256"]), {"queries_ki_P1.tsv", "topics.tsv"})
                    combined = run_pipeline(*arguments, root / "combined_c1", smoke=True,
                                            collection_names=["C1"], combined_output=True)
                    self.assertEqual(len(combined["runs"]), 2)
                    self.assertEqual({r["filename"] for r in combined["runs"]},
                                     {"c1_tfidf.csv", "c1_bm25.csv"})
                    self.assertTrue(all(r["queries"] == 6 and r["rows"] == 600
                                        and r["verification"]["passed"] for r in combined["runs"]))
                with self.assertRaises(FileExistsError):
                    run_pipeline(*arguments, root / "results", smoke=True)
            self.assertEqual(len(first["indexes"]), 4)
            self.assertEqual(len(first["runs"]), 8)
            self.assertTrue(first["all_queries_returned_100"])
            self.assertTrue(all(r["queries"] == 3 and r["rows"] == 300 for r in first["runs"]))
            self.assertTrue(all(i["reused"] for i in second["indexes"].values()))
            saved = json.loads((root / "results/retrieval_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["top_k"], 100)

    def test_invalid_tables_empty_corpus_and_repository_outputs_rejected(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.tsv"
            pd.DataFrame({"qid": ["q", "q"], "query": ["one", "two"]}).to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "duplicate qid"):
                read_table(path, ["qid", "query"], "qid")
        with self.assertRaises(ValueError):
            external_path(Path(__file__).resolve().parents[1] / "indexes")
        with self.assertRaises(ValueError):
            build_index("bm25", ["", ""], ["a", "b"])


def tiny_embeddings():
    model = KeyedVectors(vector_size=3)
    model.add_vectors(["apple", "banana", "novel", "zero", "opposite"],
                      np.asarray([[4, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [-4, 0, 0]],
                                 dtype=np.float32))
    return model


class DenseRetrievalTests(unittest.TestCase):
    def test_raw_weighted_average_excludes_oov_from_denominator(self):
        counts = sparse.csr_matrix([[2, 1, 50], [0, 0, 2], [0, 0, 0]])
        vectors = np.asarray([[2, 0], [0, 3], [9, 9]], dtype=np.float32)
        actual = weighted_average_embeddings(counts, [3, 1, 100], vectors,
                                              usable=[True, True, False], block_size=1)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, [[12 / 7, 3 / 7], [0, 0], [0, 0]], rtol=1e-6)
        # A known zero word vector remains in the denominator, as the formula requires.
        zero_word = weighted_average_embeddings(sparse.csr_matrix([[1, 1]]), [1, 1],
                                                  np.asarray([[2, 0], [0, 0]], dtype=np.float32))
        np.testing.assert_array_equal(zero_word, [[1, 0]])

    def test_document_raw_tf_and_original_vector_lengths_are_preserved(self):
        texts = ["apple apple banana missing", "banana banana", "missing"]
        tfidf = TfidfVectorizer(analyzer=preprocess_text, sublinear_tf=True).fit(texts)
        output = np.empty((3, 3), dtype=np.float32)
        details = build_dense_documents(iter(texts), ["d1", "d2", "d3"], tiny_embeddings(),
                                         tfidf, output, block_size=1)
        apple_idf = tfidf.idf_[tfidf.vocabulary_["apple"]]
        banana_idf = tfidf.idf_[tfidf.vocabulary_["banana"]]
        expected = np.asarray([2 * apple_idf * 4, banana_idf, 0])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(output[0], expected, rtol=1e-6)
        np.testing.assert_array_equal(output[1:], [[0, 1, 0], [0, 0, 0]])
        self.assertEqual(details["zero_document_embeddings"], 1)
        self.assertEqual(details["document_matrix_shape"], [3, 3])

    def test_query_oov_occurrences_empty_queries_and_collection_unseen_terms(self):
        model = tiny_embeddings()
        tfidf = TfidfVectorizer(analyzer=preprocess_text, sublinear_tf=True).fit(["apple banana", "apple"])
        vectors, stats = embed_query_batch(["apple apple missing", "missing missing", ""], model, tfidf, 2)
        np.testing.assert_array_equal(vectors, [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(stats["query_tokens"], 5)
        self.assertEqual(stats["oov_tokens"], 3)
        self.assertEqual(stats["oov_token_fraction"], .6)
        self.assertEqual(stats["queries_with_oov_fraction"], 2 / 3)
        self.assertEqual(stats["queries_without_embedding"], 2)
        vectors, stats = embed_query_batch(["apple novel"], model, tfidf, 2)
        expected = np.asarray([4., 0., 1 + math.log(3)])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(vectors[0], expected, rtol=1e-6)
        self.assertEqual(stats["oov_tokens"], 0)
        self.assertEqual(stats["queries_without_embedding"], 0)

    def test_zero_and_cancelling_embeddings_score_zero_and_negative_cosine_is_valid(self):
        model = tiny_embeddings()
        tfidf = TfidfVectorizer(analyzer=preprocess_text).fit(["apple opposite zero"])
        documents = np.asarray([[1, 0, 0], [0, 0, 0], [-1, 0, 0]], dtype=np.float32)
        index = DenseIndex(documents, np.asarray(["d2", "d1", "d0"]), model, tfidf, 1, False, {})
        rankings, stats = index.search_batch(["missing", "zero", "apple opposite"], 3)
        for docnos, scores in rankings:
            self.assertEqual(docnos.tolist(), ["d0", "d1", "d2"])
            np.testing.assert_array_equal(scores, [0, 0, 0])
        self.assertEqual(stats["queries_without_embedding"], 3)
        rankings, _ = index.search_batch(["apple"], 3)
        self.assertEqual(rankings[0][0].tolist(), ["d2", "d1", "d0"])
        np.testing.assert_array_equal(rankings[0][1], [1, 0, -1])

    def test_batched_top_100_csv_and_oov_totals(self):
        model = tiny_embeddings()
        tfidf = TfidfVectorizer(analyzer=preprocess_text).fit(["apple banana"])
        documents = np.zeros((105, 3), dtype=np.float32)
        documents[:60, 0] = 1
        documents[60:, 1] = 1
        docnos = np.asarray([f"C1-{i:06d}" for i in range(104, -1, -1)])
        index = DenseIndex(documents, docnos, model, tfidf, 1, False, {})
        queries = pd.DataFrame({"qid": ["q1", "q2", "q3", "q4", "q5"],
                                "query": ["apple", "banana", "missing", "", "apple apple"]})
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "dense.csv"
            with patch.object(index, "search_batch", wraps=index.search_batch) as search:
                result = run_dense_queries(index, queries, path, batch_size=2)
                self.assertEqual([len(call.args[0]) for call in search.call_args_list], [2, 2, 1])
            self.assertEqual(result["rows"], 500)
            self.assertTrue(result["verification"]["passed"])
            self.assertEqual(result["maximum_score_matrix_shape"], [2, 105])
            self.assertEqual(result["maximum_score_matrix_bytes"], 840)
            self.assertEqual(result["oov"]["oov_token_fraction"], .2)
            self.assertEqual(result["oov"]["queries_with_oov_fraction"], .2)
            self.assertEqual(result["oov"]["queries_without_embedding"], 2)
            table = pd.read_csv(path)
            self.assertTrue(table["score"].between(-1, 1).all())
            self.assertEqual(table.loc[table.qid == "q3", "docno"].tolist(), sorted(docnos)[:100])
            self.assertEqual(table.loc[table.qid == "q1", "docno"].tolist(),
                             table.loc[table.qid == "q5", "docno"].tolist())
            self.assertEqual(table.iloc[0]["docno"], "C1-000045")
            with self.assertRaises(FileExistsError):
                run_dense_queries(index, queries, path)

    def test_dense_cache_readonly_round_trip_and_wrong_provenance_or_corruption(self):
        model = tiny_embeddings()
        tfidf = TfidfVectorizer(analyzer=preprocess_text).fit(["apple banana"])
        documents = normalize_embeddings(np.asarray([[4, 1, 0], [0, 1, 0]], dtype=np.float32))
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory / "documents.npy", documents)
            (directory / "docnos.json").write_text(json.dumps(["d1", "d2"]), encoding="utf-8")
            metadata = {"provenance": {"source": "fixture"}, "idf_document_count": 1,
                        "google_news": False, "documents_sha256": sha256_file(directory / "documents.npy"),
                        "docnos_sha256": sha256_file(directory / "docnos.json")}
            (directory / "dense_index.json").write_text(json.dumps(metadata), encoding="utf-8")
            loaded = load_dense_index(directory, {"source": "fixture"}, model, tfidf)
            try:
                self.assertIsInstance(loaded.documents, np.memmap)
                self.assertFalse(loaded.documents.flags.writeable)
                actual, _ = loaded.search_batch(["apple banana"], 2)
                expected_query = normalize_embeddings(np.asarray([[4, 1, 0]], dtype=np.float32))[0]
                expected_scores = documents @ expected_query
                np.testing.assert_allclose(actual[0][1], expected_scores, rtol=1e-6)
            finally:
                loaded.close()
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_dense_index(directory, {"source": "changed"}, model, tfidf)
            with (directory / "documents.npy").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_dense_index(directory, {"source": "fixture"}, model, tfidf)

    def test_google_news_uses_existing_case_lookup_without_download(self):
        model = KeyedVectors(vector_size=3)
        model.add_vectors(["Apple", "US"], np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
        tfidf = TfidfVectorizer(analyzer=preprocess_text).fit(["apple us"])
        vectors, stats = embed_query_batch(["apple", "U.S."], model, tfidf, 1, google_news=True)
        np.testing.assert_array_equal(vectors, [[1, 0, 0], [0, 1, 0]])
        self.assertEqual(stats["oov_token_fraction"], 0)

    def test_preprocessed_gn_subset_uses_direct_keys_for_documents_and_queries(self):
        model = KeyedVectors(vector_size=3)
        model.add_vectors(["apple", "Banana"],
                          np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
        texts = ["apple banana", "banana"]
        tfidf = TfidfVectorizer(analyzer=preprocess_text).fit(texts)
        documents = np.empty((2, 3), dtype=np.float32)
        with patch("src.google_news.lookup_gn_key", side_effect=AssertionError("Unexpected case lookup")):
            details = build_dense_documents(iter(texts), ["d1", "d2"], model, tfidf,
                                            documents, google_news=False)
            queries, stats = embed_query_batch(texts, model, tfidf, 2, google_news=False)
        np.testing.assert_array_equal(documents, [[1, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(queries, documents)
        self.assertEqual(details["observed_terms_with_embeddings"], 1)
        self.assertEqual(stats["oov_tokens"], 2)
        self.assertEqual(stats["queries_without_embedding"], 1)

    def test_preprocessed_gn_option_rejects_incompatible_models(self):
        for model_label, model_format in [("M1", "keyedvectors"), ("GN", "word2vec")]:
            with self.assertRaisesRegex(ValueError, "Preprocessed GN keys require"):
                run_dense_pipeline("unused", "unused", "unused", "unused", "unused",
                                   "unused", model_label, gn_format=model_format,
                                   gn_preprocessed=True)

    def test_dense_memory_failure_does_not_publish_partial_csv(self):
        model = tiny_embeddings()
        tfidf = TfidfVectorizer(analyzer=preprocess_text).fit(["apple"])
        index = DenseIndex(np.tile(np.asarray([[1, 0, 0]], dtype=np.float32), (100, 1)),
                           np.asarray([str(i) for i in range(100)]), model, tfidf, 1, False, {})
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "dense.csv"
            with patch.object(index, "search_batch", side_effect=MemoryError("synthetic")):
                with self.assertRaises(MemoryError):
                    run_dense_queries(index, pd.DataFrame({"qid": ["q"], "query": ["apple"]}), path)
            self.assertEqual(list(Path(temporary).iterdir()), [])


class QueryExpansionTests(unittest.TestCase):
    def test_centroid_averages_unit_vectors_and_preserves_query_occurrences(self):
        model = tiny_embeddings()
        before = model.vectors.copy()
        np.testing.assert_allclose(expansion_centroid(["apple", "banana", "missing"], model),
                                   [.5, .5, 0])
        np.testing.assert_allclose(expansion_centroid(["apple", "apple", "banana"], model),
                                   [2 / 3, 1 / 3, 0])
        for tokens in [[], ["missing"], ["zero"], ["apple", "opposite"]]:
            self.assertIsNone(expansion_centroid(tokens, model))
        np.testing.assert_array_equal(model.vectors, before)

    def test_neighbors_restrict_vocabulary_exclude_originals_and_break_ties(self):
        model = KeyedVectors(vector_size=2)
        model.add_vectors(["query", "alpha", "beta", "side", "negative", "modelonly", "zero"],
                          np.asarray([[4, 0], [2, 0], [3, 0], [0, 1], [-1, 0], [9, 0], [0, 0]],
                                     dtype=np.float32))
        before = model.vectors.copy()
        expander = QueryExpander(model, ["query", "beta", "alpha", "side", "negative", "zero", "oov"])
        weights, details = expander.expand("QUERY query oov", 5, .5)
        self.assertEqual([entry["term"] for entry in details["expansion"]],
                         ["alpha", "beta", "side", "negative"])
        self.assertEqual(weights, {"query": 1, "oov": 1, "alpha": .5, "beta": .5, "side": 0, "negative": -.5})
        self.assertFalse(details["fallback"])
        # A model-known query term need not be in the collection to form a centroid.
        weights, _ = expander.expand("modelonly", 5, 1)
        self.assertIn("alpha", weights)
        np.testing.assert_array_equal(model.vectors, before)

    def test_empty_candidate_set_is_safe(self):
        expander = QueryExpander(tiny_embeddings(), ["missing", "zero"])
        weights, details = expander.expand("apple", 20, 1)
        self.assertEqual(weights, {"apple": 1})
        self.assertEqual(details["expansion"], [])
        self.assertFalse(details["fallback"])

    def test_lambda_scaling_m_prefixes_and_cache_do_not_repeat_neighbor_search(self):
        terms = [f"neighbor{i:02d}" for i in range(25)]
        model = KeyedVectors(vector_size=2)
        model.add_vectors(["query", *terms],
                          np.asarray([[1, 0]] + [[1, i / 30] for i in range(25)], dtype=np.float32))
        expander = QueryExpander(model, terms)
        weights5, details5 = expander.expand("query query", 5, .5)
        with patch.object(expander, "_prepare_candidates", side_effect=AssertionError("unexpected neighbor rebuild")):
            weights10, details10 = expander.expand("query query", 10, 1)
            _, details20 = expander.expand("query query", 20, .5)
        self.assertEqual(expander.neighbor_searches, 1)
        self.assertEqual(expander.cache_hits, 2)
        self.assertEqual([len(d["expansion"]) for d in [details5, details10, details20]], [5, 10, 20])
        self.assertEqual(weights5["query"], 1)
        for entry in details5["expansion"]:
            self.assertAlmostEqual(weights5[entry["term"]], .5 * entry["cosine"])
            self.assertAlmostEqual(weights10[entry["term"]], 2 * weights5[entry["term"]])

    def test_weighted_bm25_scores_match_appendix_b_hand_calculation(self):
        index = build_index("bm25", ["apple apple banana", "banana", "apple"], ["d0", "d1", "d2"])
        docnos, scores = index.search_weighted({"apple": 1, "banana": .25, "missing": 17}, 3)
        idf = math.log(1 + 1.5 / 2.5)
        average = 5 / 3
        expected = {}
        for docno, apple, banana, length in [("d0", 2, 1, 3), ("d1", 0, 1, 1), ("d2", 1, 0, 1)]:
            norm = 1.2 * (1 - .75 + .75 * length / average)
            expected[docno] = idf * (apple * 2.2 / (apple + norm) + .25 * banana * 2.2 / (banana + norm))
        for docno, score in zip(docnos, scores):
            self.assertAlmostEqual(score, expected[docno], places=12)
        self.assertEqual(docnos.tolist(), sorted(expected, key=lambda term: (-expected[term], term)))
        repeated = index.search("apple apple banana", 3)
        distinct = index.search_weighted({"apple": 1, "banana": 1}, 3)
        np.testing.assert_array_equal(repeated[0], distinct[0])
        np.testing.assert_array_equal(repeated[1], distinct[1])
        with self.assertRaises(ValueError):
            index.search_weighted({"apple": float("nan")})
        with self.assertRaises(ValueError):
            build_index("tfidf", ["apple"], ["d"]).search_weighted({"apple": 1})

    def test_oov_zero_or_cancelling_centroid_falls_back_to_original_bm25(self):
        index = build_index("bm25", ["missing zero apple", "opposite apple", "banana"], ["d0", "d1", "d2"])
        expander = QueryExpander(tiny_embeddings(), index.vectorizer.vocabulary_)
        for query in ["missing missing", "zero", "apple opposite", ""]:
            weights, details = expander.expand(query, 20, .5)
            self.assertTrue(details["fallback"])
            self.assertEqual(details["expansion"], [])
            expected = index.search(query, 3)
            actual = index.search_weighted(weights, 3)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])

    def test_persistent_candidate_cache_reused_and_mismatches_rejected(self):
        model = tiny_embeddings()
        expander = QueryExpander(model, ["apple", "banana", "opposite"])
        expected = expander.expand("apple", 5, .5)
        provenance = {"collection": "C1", "model_hash": "synthetic"}
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.json"
            expander.save_cache(path, provenance)
            other = QueryExpander(model, expander.vocabulary)
            other.load_cache(path, provenance)
            with patch.object(other, "_prepare_candidates", side_effect=AssertionError("unexpected rebuild")):
                self.assertEqual(other.expand("apple", 5, .5), expected)
            self.assertEqual(other.neighbor_searches, 0)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                QueryExpander(model, expander.vocabulary).load_cache(path, {"collection": "C2"})
            bad = json.loads(path.read_text(encoding="utf-8"))
            bad["entries"][0]["neighbors"][0]["term"] = "apple"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid expansion cache"):
                QueryExpander(model, expander.vocabulary).load_cache(path, provenance)

    def test_expansion_google_news_subset_uses_direct_lookup_only(self):
        model = KeyedVectors(vector_size=2)
        model.add_vectors(["apple", "banana", "Capitalized"],
                          np.asarray([[1, 0], [1, 1], [1, 0]], dtype=np.float32))
        expander = QueryExpander(model, ["apple", "banana", "capitalized"])
        with patch("src.google_news.lookup_gn_key", side_effect=AssertionError("Unexpected case lookup")):
            weights, _ = expander.expand("apple", 5, .5)
            _, missing = expander.expand("capitalized", 5, .5)
        self.assertEqual(set(weights), {"apple", "banana"})
        self.assertTrue(missing["fallback"])

    def test_pipeline_reuses_saved_indexes_on_each_collection_and_cached_neighbors(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            official, indexes = root / "official", root / "indexes"
            official.mkdir()
            indexes.mkdir()
            model = KeyedVectors(vector_size=300)
            vectors = np.zeros((3, 300), dtype=np.float32)
            vectors[0, 0], vectors[1, :2], vectors[2, 1] = 4, [1, 1], 3
            model.add_vectors(["apple", "banana", "novel"], vectors)
            model_path = root / "subset.kv"
            model.save(str(model_path), separately=["vectors"])
            pd.DataFrame({"tid": [f"T{i:02d}" for i in range(32)],
                          "query": ["apple", "banana", "missing"] + ["apple"] * 29}).to_csv(
                              official / "topics.tsv", sep="\t", index=False)
            for collection, period, word in [("C1", "P1", "banana"), ("C2", "P2", "novel")]:
                docnos = [f"{collection}-{i:06d}" for i in range(105)]
                collection_path = official / f"collection_{period}.tsv"
                pd.DataFrame({"docno": docnos, "web_url": [f"u{i}" for i in range(105)]}).to_csv(
                    collection_path, sep="\t", index=False)
                pd.DataFrame({"qid": [f"KI{period}-{i}" for i in range(1000)],
                              "query": ["apple apple", "missing", word] + ["apple"] * 997}).to_csv(
                                  official / f"queries_ki_{period}.tsv", sep="\t", index=False)
                index = build_index("bm25", [f"apple {word}"] * 105, docnos)
                provenance = {"smoke": False, "collection": collection,
                              "collection_sha256": sha256_file(collection_path),
                              "preprocessing_sha256": sha256_file(Path(__file__).resolve().parents[1] / "src/preprocessing.py"),
                              "model_parameters": model_parameters("bm25")}
                save_index(indexes / f"{collection}_bm25", index, provenance)
            original_files = {p: sha256_file(p) for p in indexes.rglob("*") if p.is_file()}
            with patch.dict("src.retrieval.EXPECTED_COLLECTION_COUNTS", {"C1": 105, "C2": 105}), \
                 patch("src.retrieval.build_index", side_effect=AssertionError("unexpected index build")), \
                 patch("src.retrieval.matched_texts", side_effect=AssertionError("unexpected document read")), \
                 contextlib.redirect_stdout(io.StringIO()):
                first = run_expansion_pipeline(official, indexes, root / "first", model_path, "GN", smoke=True)
                self.assertEqual(len(first["runs"]), 6)
                for run in first["runs"]:
                    self.assertEqual(run["verification"]["rows"], 600)
                    self.assertEqual(run["verification"]["queries"], 6)
                    self.assertEqual(run["fallback_queries"], 2)
                cache_bytes = Path(first["candidate_cache"]).read_bytes()
                with patch.object(QueryExpander, "_prepare_candidates", side_effect=AssertionError("unexpected neighbors")):
                    reused = run_expansion_pipeline(official, indexes, root / "reused", model_path, "GN",
                                                    smoke=True, expansion_sizes=[10], expansion_lambdas=[1])
                self.assertEqual(reused["new_neighbor_searches"], 0)
                self.assertEqual(Path(first["candidate_cache"]).read_bytes(), cache_bytes)
                c2 = run_expansion_pipeline(official, indexes, root / "c2", model_path, "GN", collection="C2",
                                            smoke=True, expansion_sizes=[5], expansion_lambdas=[.5])
                self.assertNotEqual(first["candidate_cache"], c2["candidate_cache"])
                self.assertEqual(c2["topic_examples"][0]["expansion"][0]["term"], "novel")
                with self.assertRaises(FileExistsError):
                    run_expansion_pipeline(official, indexes, root / "first", model_path, "GN", smoke=True)
            for path, digest in original_files.items():
                self.assertEqual(sha256_file(path), digest)


class HybridRetrievalTests(unittest.TestCase):
    def endpoint_indexes(self):
        """Disjoint top-100 lists with strictly ordered, controllable scores."""
        texts = ["apple " * (i + 1) + "banana" for i in range(240)]
        docnos = np.asarray([f"d{i:03d}" for i in range(240)])
        bm25 = build_index("bm25", texts, docnos)
        tfidf = TfidfVectorizer(analyzer=preprocess_text, sublinear_tf=True).fit(texts)
        cosine = np.linspace(.95, -.95, 240, dtype=np.float32)
        documents = np.column_stack([cosine, np.sqrt(1 - cosine ** 2), np.zeros(240)]).astype(np.float32)
        dense = DenseIndex(documents, docnos, tiny_embeddings(), tfidf, 240, False, {})
        return bm25, dense

    def make_indexes(self):
        texts = ["apple apple apple", "apple apple banana", "apple banana", "missing", "banana novel"]
        bm25 = build_index("bm25", texts, ["a", "b", "c", "d", "e"])
        tfidf = TfidfVectorizer(analyzer=preprocess_text, sublinear_tf=True).fit(texts)
        # Dense and lexical caches can have different row orders.
        documents = np.asarray([[0, 0, 1], [1, 0, 0], [.6, .8, 0], [.8, .6, 0], [0, 0, 0]],
                               dtype=np.float32)
        dense = DenseIndex(documents, np.asarray(["e", "c", "a", "b", "d"]),
                           tiny_embeddings(), tfidf, 5, False, {})
        return bm25, dense

    def test_union_contains_only_original_candidates_and_is_deterministic(self):
        union = fusion_candidates(["c", "a"], ["b", "c"])
        self.assertEqual(union.tolist(), ["a", "b", "c"])
        hundred = [f"d{i:03d}" for i in range(100)]
        other = [f"d{i:03d}" for i in range(80, 180)]
        self.assertEqual(len(fusion_candidates(hundred, other)), 180)
        with self.assertRaises(ValueError):
            fusion_candidates(["a", "a"], ["b"])

    def test_independent_minmax_uses_all_union_scores_including_negative_cosines(self):
        bm25 = np.asarray([10., 20., 100.])
        dense = np.asarray([-.6, .2, .4])
        np.testing.assert_allclose(minmax_scores(bm25), [0, 1 / 9, 1])
        np.testing.assert_allclose(minmax_scores(dense), [0, .8, 1])
        np.testing.assert_allclose(linear_fusion(bm25, dense, .3), [0, .3 / 9 + .7 * .8, 1])
        np.testing.assert_array_equal(bm25, [10, 20, 100])
        np.testing.assert_array_equal(dense, [-.6, .2, .4])

    def test_equal_scores_are_zero_and_do_not_receive_half_weight_or_epsilon(self):
        np.testing.assert_array_equal(minmax_scores([7, 7, 7]), [0, 0, 0])
        np.testing.assert_allclose(linear_fusion([7, 7, 7], [-1, 0, 1], .5), [0, .25, .5])
        np.testing.assert_array_equal(linear_fusion([7, 7], [-2, -2], .3), [0, 0])

    def test_alpha_endpoints_and_lexical_ties(self):
        bm25, dense = [0, 10, 5], [.8, .2, .5]
        np.testing.assert_array_equal(linear_fusion(bm25, dense, 0), minmax_scores(dense))
        np.testing.assert_array_equal(linear_fusion(bm25, dense, 1), minmax_scores(bm25))
        from src.retrieval import rank_scores
        docnos = np.asarray(["c", "a", "b"])
        tie_order = np.argsort(docnos)
        self.assertEqual(rank_scores(linear_fusion(bm25, dense, 0), docnos, tie_order, 3)[0].tolist(), ["c", "b", "a"])
        self.assertEqual(rank_scores(linear_fusion(bm25, dense, 1), docnos, tie_order, 3)[0].tolist(), ["a", "b", "c"])
        np.testing.assert_allclose(linear_fusion([0, 10], [1, 0], .5), [.5, .5])
        self.assertEqual(FUSION_ALPHAS, tuple(i / 10 for i in range(11)))

    def test_alpha_one_reproduces_original_bm25_top_100_over_full_union(self):
        from src.retrieval import rank_scores
        bm25, dense = self.endpoint_indexes()
        bm_docs, _ = bm25.search("apple", 100)
        de_docs, _ = dense.search_batch(["apple"], 100)[0][0]
        record = score_fusion_candidates("apple", bm25, dense, bm_docs, de_docs)
        self.assertEqual(len(record["docnos"]), 200)
        scores = linear_fusion(record["bm25"], record["dense"], 1)
        ranked, _ = rank_scores(scores, record["docnos"], np.arange(200), 100)
        np.testing.assert_array_equal(ranked, bm_docs)

    def test_alpha_zero_reproduces_original_dense_top_100_and_introduces_dense_only_hits(self):
        from src.retrieval import rank_scores
        bm25, dense = self.endpoint_indexes()
        bm_docs, _ = bm25.search("apple", 100)
        de_docs, _ = dense.search_batch(["apple"], 100)[0][0]
        record = score_fusion_candidates("apple", bm25, dense, bm_docs, de_docs)
        self.assertFalse(set(bm_docs) & set(de_docs))
        self.assertEqual(len(record["docnos"]), 200)
        scores = linear_fusion(record["bm25"], record["dense"], 0)
        ranked, _ = rank_scores(scores, record["docnos"], np.arange(200), 100)
        np.testing.assert_array_equal(ranked, de_docs)
        self.assertNotIn(ranked[0], bm_docs)

    def test_endpoint_ties_use_official_docno_instead_of_source_list_position(self):
        from src.retrieval import rank_scores
        docnos = np.asarray(["d3", "d1", "d2"])
        order = np.argsort(docnos)
        bm25, dense = [5, 5, 2], [.1, .9, .9]
        bm_endpoint = rank_scores(linear_fusion(bm25, dense, 1), docnos, order, 3)[0].tolist()
        de_endpoint = rank_scores(linear_fusion(bm25, dense, 0), docnos, order, 3)[0].tolist()
        self.assertEqual(bm_endpoint, ["d1", "d3", "d2"])
        self.assertEqual(de_endpoint, ["d1", "d2", "d3"])

    def test_rrf_uses_original_one_based_ranks_with_no_missing_rank_contribution(self):
        candidates = np.asarray(["a", "b", "c", "outside"])
        scores = rrf_scores(candidates, ["a", "b"], ["b", "c"])
        np.testing.assert_allclose(scores, [1 / 61, 1 / 62 + 1 / 61, 1 / 62, 0], rtol=0, atol=1e-16)
        from src.retrieval import rank_scores
        self.assertEqual(rank_scores(scores, candidates, np.arange(4), 4)[0].tolist(), ["b", "a", "c", "outside"])

    def test_missing_side_candidates_get_exact_nonzero_scores_with_separate_row_maps(self):
        bm25, dense = self.make_indexes()
        record = score_fusion_candidates("apple apple", bm25, dense, ["a", "b"], ["c", "b"])
        self.assertEqual(record["docnos"].tolist(), ["a", "b", "c"])
        # c was not a BM25 top-2 hit, but its true query score is positive.
        average_length = 11 / 5
        idf = math.log(1 + (5 - 3 + .5) / (3 + .5))
        expected_c = idf * 2.2 / (1 + 1.2 * (.25 + .75 * 2 / average_length))
        self.assertAlmostEqual(record["bm25"][2], expected_c, places=12)
        # a was not a dense top-2 hit, but its exact cosine is .6, not zero.
        np.testing.assert_allclose(record["dense"], [.6, .8, 1], rtol=1e-6)
        np.testing.assert_allclose(record["rrf"], [1 / 61, 2 / 62, 1 / 61])
        all_docnos, all_scores = bm25.search("apple", 5)
        expected = dict(zip(all_docnos, all_scores))
        np.testing.assert_allclose(record["bm25"], [expected[d] for d in record["docnos"]], rtol=1e-12)

    def test_zero_embedding_query_is_safe_and_rrf_still_uses_saved_ranks(self):
        bm25, dense = self.make_indexes()
        record = score_fusion_candidates("outofvocabulary", bm25, dense, ["a", "b"], ["b", "c"])
        np.testing.assert_array_equal(record["bm25"], [0, 0, 0])
        np.testing.assert_array_equal(record["dense"], [0, 0, 0])
        np.testing.assert_array_equal(linear_fusion(record["bm25"], record["dense"], .5), [0, 0, 0])
        self.assertGreater(record["rrf"][1], record["rrf"][0])

    def test_fusion_rejects_invalid_scores_and_alpha(self):
        for bad in [[], [float("nan")], [float("inf")]]:
            with self.assertRaises(ValueError):
                minmax_scores(bad)
        for alpha in [-.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                linear_fusion([0, 1], [1, 0], alpha)
        with self.assertRaises(ValueError):
            linear_fusion([0, 1], [1], .5)


if __name__ == "__main__":
    unittest.main()
