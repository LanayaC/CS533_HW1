"""Sparse retrieval, averaged embeddings, and embedding-based query expansion.

Run from the repository root with configurable input/output paths::

    python -m src.retrieval --dataset ../data/articles_mapped.h5 \
        --official-dir ../data/nyt-tc-2026 --index-dir ../data/part3_indexes \
        --output-dir ../data/part3_results

Use --smoke with separate destinations to index 300 documents per collection
and run three queries per set. Existing result directories are never replaced.
Use --collections C1 or --collections C2 to process one collection independently.
With --expansion-model, --smoke selects three queries per set and reuses the
full saved BM25 index. Expansion uses direct keys, including the local GN subset.
"""

import argparse
from contextlib import ExitStack
import copy
import csv
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .document_collections import EXPECTED_COLLECTION_COUNTS, clean_articles
from .preprocessing import preprocess_text


REPOSITORY = Path(__file__).resolve().parents[1]
TOP_K = 100
K1, B = 1.2, 0.75
INDEX_VERSION = 3
BM25_IDF = "log(1 + (N-df+0.5)/(df+0.5))"
MEMORY_HEADROOM_BYTES = 256 * 1024 ** 2


def model_parameters(model):
    """Cache identity includes all fixed scoring parameters."""
    if model == "tfidf":
        return {"sublinear_tf": True, "norm": "l2", "smooth_idf": True}
    if model == "bm25":
        return {"k1": K1, "b": B, "idf": BM25_IDF,
                "query_terms": "Distinct original query terms each have weight 1; no k3 factor"}
    raise ValueError(f"Unknown model: {model}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def memory_snapshot():
    """Process RSS/peak working set and system availability, when measurable."""
    result = {"rss_bytes": None, "process_peak_rss_bytes": None,
              "system_available_bytes": None, "system_total_bytes": None}
    try:
        import psutil
        info = psutil.Process().memory_info()
        result["rss_bytes"] = int(info.rss)
        peak = getattr(info, "peak_wset", None)
        if peak is not None:
            result["process_peak_rss_bytes"] = int(peak)
        memory = psutil.virtual_memory()
        result.update(system_available_bytes=int(memory.available),
                      system_total_bytes=int(memory.total))
    except ImportError:
        pass
    if result["process_peak_rss_bytes"] is None and sys.platform != "win32":
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["process_peak_rss_bytes"] = int(peak * (1 if sys.platform == "darwin" else 1024))
    return result


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def ensure_memory(additional_bytes=0):
    """Stop before a known allocation would consume the available headroom."""
    available = memory_snapshot().get("system_available_bytes")
    if available is not None and available < additional_bytes + MEMORY_HEADROOM_BYTES:
        raise MemoryError(
            f"{available:,} bytes available; need {additional_bytes:,} additional "
            f"bytes plus {MEMORY_HEADROOM_BYTES:,} bytes of headroom"
        )


def merge_summaries(previous, current):
    """Add a disjoint collection without replacing earlier measurements."""
    for key in ["smoke", "top_k", "versions", "source_paths", "tie_rule"]:
        if previous[key] != current[key]:
            raise ValueError(f"Existing summary has incompatible {key}")
    if set(previous["collections"]) & set(current["collections"]):
        raise ValueError("Refusing to replace an existing collection's measurements")
    if set(previous["indexes"]) & set(current["indexes"]):
        raise ValueError("Refusing to replace existing index measurements")
    if {r["filename"] for r in previous["runs"]} & {r["filename"] for r in current["runs"]}:
        raise ValueError("Refusing to replace existing run measurements")
    for name, digest in current["query_file_sha256"].items():
        if name in previous["query_file_sha256"] and previous["query_file_sha256"][name] != digest:
            raise ValueError(f"Query file changed since the earlier run: {name}")
    merged = copy.deepcopy(previous)
    merged["collections"].extend(current["collections"])
    merged["indexes"].update(current["indexes"])
    merged["runs"].extend(current["runs"])
    merged["query_file_sha256"].update(current["query_file_sha256"])
    merged["all_queries_returned_100"] = all(r["all_queries_returned_100"] for r in merged["runs"])
    context_keys = ["collections", "versions", "platform", "logical_cpus",
                    "memory_before", "memory_after", "source_paths"]
    previous_context = {key: previous[key] for key in context_keys if key in previous}
    current_context = {key: current[key] for key in context_keys if key in current}
    merged["executions"] = copy.deepcopy(previous.get("executions", [previous_context]))
    merged["executions"].append(current_context)
    return merged


def publish_summary(path, summary, expected_bytes=None):
    """Replace a summary only when an explicitly appended snapshot still matches."""
    path = Path(path)
    if expected_bytes is None:
        write_json(path, summary)
        return
    staged = path.parent / f".{path.name}.in_progress"
    handle = staged.open("x", encoding="utf-8", newline="\n")
    try:
        with handle:
            json.dump(summary, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
        if path.read_bytes() != expected_bytes:
            raise RuntimeError("Summary changed during analysis; refusing to overwrite it")
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def external_path(path):
    path = Path(path).resolve()
    if path.is_relative_to(REPOSITORY):
        raise ValueError("Indexes and results must be outside the repository")
    return path


def read_table(path, columns, key):
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if not set(columns).issubset(table.columns):
        raise ValueError(f"Missing columns in {path}: {columns}")
    if table.empty or table[key].duplicated().any():
        raise ValueError(f"Empty table or duplicate {key} in {path}")
    if any(table[column].str.strip().eq("").any() for column in columns):
        raise ValueError(f"Empty required value in {path}")
    return table


def matched_texts(dataset, collection, docnos, details, hdf_key="data", chunk_size=10000):
    """Stream Part 1-cleaned texts matched by exact official web_url.

    Global SHA-256 text fingerprints preserve text-deduplication order across
    chunks without retaining every article. clean_articles applies the shared
    short-text and within-chunk rules. Text deduplication precedes URL
    deduplication; only official URLs need tracking at the final join stage.
    The yielded order and corresponding official docnos are recorded together.
    """
    if collection["web_url"].duplicated().any():
        raise ValueError("Official collection contains duplicate URLs")
    wanted = dict(zip(collection["web_url"], collection["docno"]))
    seen_texts, matched = set(), set()
    digest = hashlib.sha256()
    with pd.HDFStore(dataset, mode="r") as store:
        if store.get_storer(hdf_key).format_type != "table":
            raise ValueError("Streaming requires a table-format HDF5 dataset")
        for chunk in store.select(hdf_key, columns=["web_url", "text"],
                                  chunksize=chunk_size):
            ensure_memory()
            keep = []
            for text in chunk["text"]:
                fingerprint = hashlib.sha256(text.encode("utf-8")).digest() if isinstance(text, str) else None
                first = fingerprint is not None and fingerprint not in seen_texts
                keep.append(first)
                if first:
                    seen_texts.add(fingerprint)
            cleaned, _ = clean_articles(chunk.loc[keep])
            selected = cleaned.loc[cleaned["web_url"].isin(wanted), ["web_url", "text"]]
            for url, text in selected.itertuples(index=False, name=None):
                if url in matched:
                    continue
                docno = wanted[url]
                matched.add(url)
                docnos.append(docno)
                digest.update(json.dumps([docno, text], ensure_ascii=False).encode("utf-8"))
                digest.update(b"\n")
                yield text
            if len(matched) == len(wanted):
                break
    if len(matched) != len(wanted):
        missing = sorted(set(wanted) - matched)
        raise ValueError(f"Missing cleaned article texts for {len(missing)} official URLs: {missing[:3]}")
    details.update(matched_documents=len(matched), missing_article_texts=0,
                   joined_docno_text_sha256=digest.hexdigest())


def bm25_weights(counts, k1=K1, b=B):
    """Appendix B BM25 document-term weights; no cosine normalization.

    idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5)).
    weight(t,d) = idf(t) * tf(t,d)*(k1+1) /
                  (tf(t,d) + k1*(1-b+b*len(d)/avgdl)).
    """
    if not np.isfinite(k1) or k1 <= 0 or not np.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("BM25 requires finite k1 > 0 and 0 <= b <= 1")
    if sparse.issparse(counts):
        ensure_memory(4 * (counts.data.nbytes + counts.nnz * 8 + 8 * (sum(counts.shape) + 2)))
    counts = sparse.csr_matrix(counts, copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if (min(counts.shape) == 0 or not np.isfinite(counts.data).all()
            or np.any(counts.data < 0) or np.any(counts.data % 1 != 0)):
        raise ValueError("BM25 requires nonnegative finite integer term counts")
    lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    average = float(lengths.mean())
    if average <= 0:
        raise ValueError("Cannot index an entirely empty corpus")
    frequency = np.bincount(counts.indices, minlength=counts.shape[1])
    idf = np.log1p((counts.shape[0] - frequency + 0.5) / (frequency + 0.5))
    weights = counts.astype(np.float64, copy=True)
    # Bounded blocks avoid allocating one document-length value per posting.
    for first in range(0, counts.shape[0], 4096):
        ensure_memory()
        last = min(first + 4096, counts.shape[0])
        start, stop = counts.indptr[first], counts.indptr[last]
        norm = k1 * (1 - b + b * lengths[first:last] / average)
        denominator = weights.data[start:stop] + np.repeat(norm, np.diff(counts.indptr[first:last + 1]))
        weights.data[start:stop] *= (k1 + 1) / denominator
        weights.data[start:stop] *= idf[counts.indices[start:stop]]
    return weights.tocsc(), average


def rank_scores(scores, docnos, tie_order, top_k=TOP_K):
    """Rank one document-score vector, using official docno for tied scores."""
    if top_k <= 0 or top_k > len(docnos):
        raise ValueError("top_k must be positive and no larger than the collection")
    if scores.shape != (len(docnos),) or not np.isfinite(scores).all():
        raise ValueError("Invalid or nonfinite retrieval scores")
    cutoff = np.partition(scores, len(scores) - top_k)[len(scores) - top_k]
    better = np.flatnonzero(scores > cutoff)
    ties = tie_order[scores[tie_order] == cutoff][:top_k - len(better)]
    chosen = np.concatenate([better, ties])
    chosen = chosen[np.lexsort((docnos[chosen], -scores[chosen]))]
    return docnos[chosen], scores[chosen]


@dataclass
class RetrievalIndex:
    model: str
    vectorizer: object
    weights: sparse.csc_matrix
    docnos: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.tie_order = np.argsort(self.docnos, kind="stable")
        if self.weights.shape != (len(self.docnos), len(self.vectorizer.vocabulary_)):
            raise ValueError("Index dimensions do not match vocabulary/docnos")
        if len(set(self.docnos)) != len(self.docnos):
            raise ValueError("Duplicate indexed docnos")

    def search(self, query, top_k=TOP_K):
        if top_k <= 0 or top_k > len(self.docnos):
            raise ValueError("top_k must be positive and no larger than the collection")
        vector = self.vectorizer.transform([query]).tocsr()
        if self.model == "bm25":
            # Our stated interpretation: each distinct preprocessed original
            # query term has weight 1. Document counts remain untouched.
            vector.data.fill(1)
        return self._rank_columns(vector.indices, vector.data, top_k)

    def search_weighted(self, term_weights, top_k=TOP_K):
        """Score preprocessed term weights against existing Appendix B postings.

        Unknown collection terms contribute zero. Signed finite weights are
        supported: expansion uses lambda * cosine without positive-only filtering.
        """
        if self.model != "bm25":
            raise ValueError("Weighted query expansion requires a BM25 index")
        columns, weights = [], []
        for term, weight in term_weights.items():
            if not isinstance(term, str) or not np.isfinite(weight):
                raise ValueError("Query weights require string terms and finite weights")
            if term in self.vectorizer.vocabulary_:
                columns.append(self.vectorizer.vocabulary_[term])
                weights.append(float(weight))
        return self._rank_columns(columns, weights, top_k)

    def _rank_columns(self, columns, query_weights, top_k):
        scores = np.zeros(len(self.docnos), dtype=np.float64)
        for term, query_weight in zip(columns, query_weights):
            start, stop = self.weights.indptr[term:term + 2]
            documents = self.weights.indices[start:stop]
            scores[documents] += self.weights.data[start:stop] * query_weight
        return rank_scores(scores, self.docnos, self.tie_order, top_k)

    def score_candidates(self, query, docnos, positions=None):
        """Exact original-query BM25 on arbitrary candidates, including unseen ranks."""
        if self.model != "bm25":
            raise ValueError("Candidate BM25 scoring requires a BM25 index")
        positions = positions if positions is not None else dict(zip(self.docnos, range(len(self.docnos))))
        rows = np.asarray([positions[docno] for docno in docnos], dtype=np.int64)
        terms = self.vectorizer.transform([query]).indices
        # Select query postings first, then candidate rows. Each distinct original
        # term has weight 1; no document terms or BM25 statistics are rebuilt.
        return np.asarray(self.weights[:, terms][rows].sum(axis=1)).ravel()


def normalize_embeddings(matrix):
    """L2-normalize float32 rows in place; zero rows remain zero."""
    norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64))
    np.divide(matrix, norms[:, None], out=matrix, where=norms[:, None] > 0)
    return matrix


def weighted_average_embeddings(counts, idf, vectors, *, usable=None, output=None,
                                normalize=False, block_size=2048):
    """RAW TF-IDF averages using sparse products, never sublinear TF.

    Only model-vocabulary terms contribute to numerator AND denominator.
    Word vectors retain their original lengths. IDF is supplied by the
    collection-specific vectorizer; only the resulting averages are normalized.
    Blocks bound temporary dense memory. output may be a float32 .npy memmap.
    """
    counts = sparse.csr_matrix(counts, dtype=np.float32)
    vectors = np.asarray(vectors, dtype=np.float32)
    idf = np.asarray(idf, dtype=np.float32)
    if (block_size <= 0 or vectors.ndim != 2 or vectors.shape[0] != counts.shape[1]
            or idf.shape != (counts.shape[1],)):
        raise ValueError("Invalid averaging dimensions or block size")
    if not np.isfinite(vectors).all() or not np.isfinite(idf).all() or np.any(idf <= 0):
        raise ValueError("Vectors must be finite and IDF must be finite and positive")
    if not np.isfinite(counts.data).all() or np.any(counts.data < 0) or np.any(counts.data % 1):
        raise ValueError("Dense retrieval requires raw nonnegative integer term frequencies")
    if usable is not None:
        usable = np.asarray(usable, dtype=bool)
        if usable.shape != idf.shape:
            raise ValueError("Invalid vocabulary membership mask")
        idf = idf * usable
    shape = (counts.shape[0], vectors.shape[1])
    if output is None:
        ensure_memory(int(np.prod(shape)) * 4)
        output = np.empty(shape, dtype=np.float32)
    if output.shape != shape or output.dtype != np.float32:
        raise ValueError("Expected a float32 document-by-dimension output matrix")
    for first in range(0, counts.shape[0], block_size):
        last = min(first + block_size, counts.shape[0])
        ensure_memory((last - first) * vectors.shape[1] * 8)
        weights = counts[first:last].multiply(idf).tocsr()
        averages = weights @ vectors
        denominator = np.asarray(weights.sum(axis=1)).ravel()
        np.divide(averages, denominator[:, None], out=averages,
                  where=denominator[:, None] > 0)
        averages[denominator == 0] = 0
        if not np.isfinite(averages).all():
            raise ValueError("Nonfinite averaged embeddings")
        output[first:last] = normalize_embeddings(averages) if normalize else averages
    return output


def embedding_keys(tokens, model, google_news=False):
    if google_news:
        from .google_news import lookup_gn_key
        return [lookup_gn_key(model, token) for token in tokens]
    return [token if token in model else None for token in tokens]


def embedding_rows(model, keys):
    """Gather raw word vectors once per vocabulary/batch, not per document."""
    if not keys:
        return np.empty((0, model.vector_size), dtype=np.float32)
    ensure_memory(len(keys) * model.vector_size * 4)
    rows = model.vectors[[model.key_to_index[key] for key in keys]]
    rows = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(rows).all():
        raise ValueError("Nonfinite word vectors in the embedding model")
    return rows


def expansion_centroid(tokens, model):
    """Mean of unit word vectors over query token occurrences, with no TF-IDF.

    Repetitions contribute to the centroid; BM25 original-term weights remain
    distinct. OOV and zero-length vectors cannot supply a cosine direction.
    Return None for no usable vectors or an exactly cancelling centroid.
    """
    keys = [token for token in tokens if token in model]
    if not keys:
        return None
    vectors = embedding_rows(model, keys)
    lengths = np.linalg.norm(vectors.astype(np.float64), axis=1)
    usable = lengths > 0
    if not np.any(usable):
        return None
    centroid = (vectors[usable] / lengths[usable, None]).mean(axis=0)
    return centroid if np.linalg.norm(centroid) > 0 else None


class QueryExpander:
    """Direct-token lookup, collection-restricted neighbors, and reusable cache.

    Cache the top 20 cosine neighbors before applying m or lambda. Candidate
    vectors are copied and unit-normalized once, lazily; models stay unchanged.
    Ties use ascending original token labels. Negative cosines are retained.
    """

    def __init__(self, model, collection_vocabulary):
        self.model = model
        self.vocabulary = set(collection_vocabulary)
        self.cache = {}
        self.cache_hits = 0
        self.neighbor_searches = 0
        self.terms = self.vectors = None

    def _prepare_candidates(self):
        terms = sorted(term for term in self.vocabulary if term in self.model)
        vectors = embedding_rows(self.model, terms)
        usable = np.any(vectors, axis=1)
        self.terms = np.asarray(terms, dtype=str)[usable]
        self.vectors = normalize_embeddings(vectors if usable.all() else vectors[usable])

    def neighbors(self, query):
        tokens = tuple(preprocess_text(query))
        if tokens in self.cache:
            self.cache_hits += 1
            return self.cache[tokens]
        centroid = expansion_centroid(tokens, self.model)
        neighbors = []
        if centroid is not None:
            if self.terms is None:
                self._prepare_candidates()
            # One candidate-score vector per query, never all queries x vocabulary.
            direction = (centroid / np.linalg.norm(centroid)).astype(np.float32)
            scores = self.vectors @ direction
            np.clip(scores, -1, 1, out=scores)
            eligible = np.flatnonzero(~np.isin(self.terms, list(set(tokens))))
            if len(eligible):
                terms = self.terms[eligible]
                chosen, similarities = rank_scores(scores[eligible], terms, np.arange(len(terms)),
                                                   min(20, len(terms)))
                neighbors = [{"term": str(term), "cosine": float(score)}
                             for term, score in zip(chosen, similarities)]
            self.neighbor_searches += 1
        record = {"tokens": list(tokens), "fallback": centroid is None, "neighbors": neighbors}
        self.cache[tokens] = record
        return record

    def expand(self, query, m=5, expansion_lambda=0.5):
        if m not in (5, 10, 20) or expansion_lambda not in (0.5, 1.0):
            raise ValueError("Expansion requires m in {5,10,20} and lambda in {0.5,1.0}")
        record = self.neighbors(query)
        weights = dict.fromkeys(record["tokens"], 1.0)
        expansion = [{**entry, "weight": expansion_lambda * entry["cosine"]}
                     for entry in record["neighbors"][:m]]
        weights.update({entry["term"]: entry["weight"] for entry in expansion})
        return weights, {"fallback": record["fallback"], "expansion": expansion}

    def load_cache(self, path, provenance):
        """Reject stale model/collection identities and malformed cached lists."""
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        if saved["provenance"] != provenance or saved["max_terms"] != 20:
            raise ValueError("Expansion cache model/collection/preprocessing mismatch")
        for record in saved["entries"]:
            tokens = tuple(record["tokens"])
            entries = record["neighbors"]
            terms = [entry["term"] for entry in entries]
            scores = [entry["cosine"] for entry in entries]
            if (tokens in self.cache or len(terms) > 20 or len(set(terms)) != len(terms)
                    or any(term in tokens or term not in self.vocabulary or term not in self.model
                           for term in terms)
                    or any(not np.isfinite(score) or not -1 <= score <= 1 for score in scores)
                    or entries != sorted(entries, key=lambda e: (-e["cosine"], e["term"]))
                    or (record["fallback"] and entries)):
                raise ValueError("Invalid expansion cache entries")
            self.cache[tokens] = record

    def save_cache(self, path, provenance, expected_bytes=None):
        saved = {"provenance": provenance, "max_terms": 20,
                 "entries": [self.cache[key] for key in sorted(self.cache)]}
        publish_summary(path, saved, expected_bytes)


def oov_statistics(counts):
    result = dict(counts)
    result["oov_token_fraction"] = counts["oov_tokens"] / counts["query_tokens"] if counts["query_tokens"] else 0.0
    result["queries_with_oov_fraction"] = counts["queries_with_oov"] / counts["queries"] if counts["queries"] else 0.0
    return result


def embed_query_batch(texts, model, tfidf_vectorizer, document_count, *, google_news=False):
    """Reuse smooth sklearn IDF: 1 + ln((N+1)/(df+1)), independently per collection.

    A model-known query term absent from the collection has df=0 and receives
    1+ln(N+1). It is not an embedding OOV. Repeated query tokens use RAW TF.
    OOV rates count token occurrences; empty token sequences have OOV rate 0.
    """
    tokenized = [preprocess_text(text) for text in texts]
    terms = list(dict.fromkeys(token for tokens in tokenized for token in tokens))
    keys = embedding_keys(terms, model, google_news)
    resolved = dict(zip(terms, keys))
    usable_terms = [term for term in terms if resolved[term] is not None]
    stats = {"queries": len(texts), "query_tokens": sum(map(len, tokenized)),
             "oov_tokens": sum(resolved[token] is None for tokens in tokenized for token in tokens),
             "queries_with_oov": sum(any(resolved[t] is None for t in tokens) for tokens in tokenized)}
    if usable_terms:
        counter = CountVectorizer(analyzer=preprocess_text,
                                  vocabulary={term: i for i, term in enumerate(usable_terms)},
                                  dtype=np.float32)
        counts = counter.transform(texts)
        unknown_idf = 1 + np.log(document_count + 1)
        vocabulary = tfidf_vectorizer.vocabulary_
        idf = [tfidf_vectorizer.idf_[vocabulary[t]] if t in vocabulary else unknown_idf
               for t in usable_terms]
        vectors = embedding_rows(model, [resolved[t] for t in usable_terms])
        embeddings = weighted_average_embeddings(counts, idf, vectors, normalize=True)
    else:
        embeddings = np.zeros((len(texts), model.vector_size), dtype=np.float32)
    stats["queries_without_embedding"] = int(np.count_nonzero(~np.any(embeddings, axis=1)))
    return embeddings, oov_statistics(stats)


@dataclass
class DenseIndex:
    documents: np.ndarray
    docnos: np.ndarray
    model: object
    tfidf_vectorizer: object
    idf_document_count: int
    google_news: bool
    metadata: dict

    def __post_init__(self):
        self.docnos = np.asarray(self.docnos, dtype=str)
        if (self.documents.shape != (len(self.docnos), self.model.vector_size)
                or self.documents.dtype != np.float32
                or len(set(self.docnos)) != len(self.docnos)):
            raise ValueError("Invalid dense matrix dimensions/dtype or duplicate docnos")
        self.tie_order = np.argsort(self.docnos, kind="stable")

    def search_batch(self, texts, top_k=TOP_K):
        ensure_memory(len(texts) * len(self.docnos) * 4)
        queries, stats = embed_query_batch(texts, self.model, self.tfidf_vectorizer,
                                           self.idf_document_count, google_news=self.google_news)
        similarities = queries @ self.documents.T  # One bounded batch-by-document matrix.
        if not np.isfinite(similarities).all():
            raise ValueError("Nonfinite cosine similarities")
        np.clip(similarities, -1, 1, out=similarities)  # Float32 roundoff only.
        rankings = [rank_scores(row, self.docnos, self.tie_order, top_k) for row in similarities]
        return rankings, stats

    def close(self):
        mapping = getattr(self.documents, "_mmap", None)
        if mapping is not None:
            mapping.close()

    def score_candidates(self, query_vector, docnos, positions=None):
        """Exact cosine on saved unit document rows, regardless of previous rank."""
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.shape != (self.model.vector_size,) or not np.isfinite(query_vector).all():
            raise ValueError("Invalid candidate query embedding")
        positions = positions if positions is not None else dict(zip(self.docnos, range(len(self.docnos))))
        rows = np.asarray([positions[docno] for docno in docnos], dtype=np.int64)
        scores = self.documents[rows] @ query_vector
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite candidate cosine scores")
        return np.clip(scores, -1, 1)


def build_dense_documents(documents, docnos, model, tfidf_vectorizer, output,
                          *, google_news=False, block_size=2048):
    """Use the existing collection vocabulary/IDF, but build RAW sparse counts."""
    counter = CountVectorizer(analyzer=preprocess_text,
                              vocabulary=tfidf_vectorizer.vocabulary_, dtype=np.float32)
    counts = counter.transform(documents)
    if counts.shape[0] != len(docnos) or counts.shape[0] != output.shape[0]:
        raise ValueError("Document iterator and official docnos have different lengths")
    ensure_memory(2 * counts.indices.nbytes)
    used = np.unique(counts.indices)
    terms = tfidf_vectorizer.get_feature_names_out()[used]
    keys = embedding_keys(terms, model, google_news)
    keep = [i for i, key in enumerate(keys) if key is not None]
    columns = used[keep]
    vectors = embedding_rows(model, [keys[i] for i in keep])
    selected = counts[:, columns]
    del counts
    weighted_average_embeddings(selected, tfidf_vectorizer.idf_[columns], vectors,
                                output=output, normalize=True, block_size=block_size)
    zero_documents = sum(int(np.count_nonzero(~np.any(output[i:i + block_size], axis=1)))
                         for i in range(0, len(docnos), block_size))
    return {"zero_document_embeddings": zero_documents,
            "observed_terms_with_embeddings": len(columns),
            "document_matrix_shape": list(output.shape), "document_matrix_dtype": str(output.dtype)}


def load_dense_index(directory, provenance, model, tfidf_vectorizer):
    directory = Path(directory)
    metadata = json.loads((directory / "dense_index.json").read_text(encoding="utf-8"))
    if metadata["provenance"] != provenance:
        raise ValueError("Dense cache model/collection/preprocessing/IDF mismatch")
    if sha256_file(directory / "documents.npy") != metadata["documents_sha256"]:
        raise ValueError("Dense document matrix checksum mismatch")
    if sha256_file(directory / "docnos.json") != metadata["docnos_sha256"]:
        raise ValueError("Dense document identifiers checksum mismatch")
    docnos = json.loads((directory / "docnos.json").read_text(encoding="utf-8"))
    documents = np.load(directory / "documents.npy", mmap_mode="r", allow_pickle=False)
    return DenseIndex(documents, np.asarray(docnos), model, tfidf_vectorizer,
                      metadata["idf_document_count"], metadata["google_news"], metadata)


def run_dense_queries(index, queries, output, batch_size=16):
    """Bound cosine scoring to batch_size x N and report occurrence-based OOV."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing existing dense results: {output}")
    if batch_size <= 0 or queries.empty or queries["qid"].duplicated().any():
        raise ValueError("Use nonempty unique queries and a positive batch size")
    totals = dict.fromkeys(["queries", "query_tokens", "oov_tokens", "queries_with_oov",
                            "queries_without_embedding"], 0)
    seconds, rows = 0.0, 0
    staged = output.parent / f".{output.name}.in_progress"
    handle = staged.open("x", encoding="utf-8", newline="")
    try:
        with handle:
            writer = csv.writer(handle)
            writer.writerow(["qid", "docno", "rank", "score"])
            for first in range(0, len(queries), batch_size):
                batch = queries.iloc[first:first + batch_size]
                start = perf_counter()
                rankings, stats = index.search_batch(batch["query"].tolist())
                seconds += perf_counter() - start
                for key in totals:
                    totals[key] += stats[key]
                for qid, (docnos, scores) in zip(batch["qid"], rankings):
                    for rank, (docno, score) in enumerate(zip(docnos, scores), 1):
                        writer.writerow([qid, docno, rank, format(float(score), ".17g")])
                        rows += 1
        verification = validate_results(staged, queries["qid"], index.docnos)
        os.link(staged, output)
    finally:
        staged.unlink()
    return {"filename": output.name, "queries": len(queries), "rows": rows,
            "verification": verification, "oov": oov_statistics(totals),
            "query_seconds": seconds, "mean_query_seconds": seconds / len(queries),
            "query_timing": "Embedding, batched cosine scoring, ranking; excludes CSV writing/validation",
            "maximum_score_matrix_shape": [min(batch_size, len(queries)), len(index.docnos)],
            "maximum_score_matrix_bytes": min(batch_size, len(queries)) * len(index.docnos) * 4,
            "memory_after_queries": memory_snapshot()}


def run_dense_pipeline(dataset, official_dir, index_dir, dense_index_dir, output_dir,
                       embedding_model, model_label, *, collection="C1", smoke=False,
                       hdf_key="data", chunk_size=10000, query_batch_size=16,
                       embedding_block_size=2048, gn_format="word2vec",
                       gn_preprocessed=False):
    """Process exactly one embedding-model/collection pair; never change sparse results."""
    if collection not in EXPECTED_COLLECTION_COUNTS or model_label not in {"M1", "M2", "M2s", "GN"}:
        raise ValueError("Unknown collection or embedding model label")
    if min(chunk_size, query_batch_size, embedding_block_size) <= 0:
        raise ValueError("Chunk and batch sizes must be positive")
    if gn_preprocessed and (model_label != "GN" or gn_format != "keyedvectors"):
        raise ValueError("Preprocessed GN keys require GN in native keyedvectors format")
    # The extracted GN subset already maps the Part 2 case variants to original
    # preprocessed tokens. Resolve it directly; do not apply casing a second time.
    case_lookup = model_label == "GN" and not gn_preprocessed
    dataset, official_dir = Path(dataset).resolve(), Path(official_dir).resolve()
    model_path = Path(embedding_model).resolve()
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"Embedding model must already exist locally: {model_path}")
    output_dir, dense_index_dir = external_path(output_dir), external_path(dense_index_dir)
    if output_dir.exists():
        raise FileExistsError(f"Use a new dense results directory: {output_dir}")
    tfidf_directory = Path(index_dir).resolve() / f"{collection}_tfidf"
    cached = json.loads((tfidf_directory / "index.json").read_text(encoding="utf-8"))
    period = {"C1": "P1", "C2": "P2"}[collection]
    collection_path = official_dir / f"collection_{period}.tsv"
    official = read_table(collection_path, ["docno", "web_url"], "docno")
    if len(official) != EXPECTED_COLLECTION_COUNTS[collection]:
        raise ValueError("Unexpected official document count")
    if (cached["model"] != "tfidf" or cached["documents"] != len(official)
            or cached["provenance"]["collection"] != collection or cached["provenance"]["smoke"]):
        raise ValueError("Dense retrieval needs the full collection's TF-IDF vectorizer")
    dataset_hash = sha256_file(dataset)
    preprocessing_hash = sha256_file(REPOSITORY / "src/preprocessing.py")
    for key, expected in [("dataset_sha256", dataset_hash),
                          ("preprocessing_sha256", preprocessing_hash), ("hdf_key", hdf_key)]:
        if cached["provenance"][key] != expected:
            raise ValueError(f"TF-IDF input mismatch: {key}")
    if cached["provenance"]["collection_sha256"] != sha256_file(collection_path):
        raise ValueError("Official collection changed since TF-IDF indexing")
    vectorizer = joblib.load(tfidf_directory / "vectorizer.joblib")
    if (not isinstance(vectorizer, TfidfVectorizer) or vectorizer.analyzer is not preprocess_text
            or not vectorizer.use_idf or not vectorizer.smooth_idf):
        raise ValueError("Expected the shared analyzer and sklearn smooth IDF convention")
    selected = official.iloc[:300].copy() if smoke else official
    known = read_table(official_dir / f"queries_ki_{period}.tsv", ["qid", "query"], "qid")
    topics = read_table(official_dir / "topics.tsv", ["tid", "query"], "tid").rename(columns={"tid": "qid"})
    if len(known) != 1000 or len(topics) != 32:
        raise ValueError("Expected 1,000 known-item queries and 32 topic queries")
    if smoke:
        known, topics = known.iloc[:3].copy(), topics.iloc[:3].copy()
    # Native Gensim files may have array sidecars. Hash them as part of cache identity.
    model_files = [model_path, *sorted(model_path.parent.glob(model_path.name + ".*"))]
    provenance = {
        "dense_version": 1, "collection": collection, "model": model_label, "smoke": smoke,
        "dataset_sha256": dataset_hash, "preprocessing_sha256": preprocessing_hash,
        "tfidf_metadata_sha256": sha256_file(tfidf_directory / "index.json"),
        "idf_vectorizer_sha256": sha256_file(tfidf_directory / "vectorizer.joblib"),
        "selected_docnos_sha256": hashlib.sha256("\n".join(selected["docno"]).encode()).hexdigest(),
        "model_file_sha256": {p.name: sha256_file(p) for p in model_files if p.is_file()},
        "gn_format": gn_format if model_label == "GN" else None,
    }
    if gn_preprocessed:
        provenance["gn_preprocessed"] = True
    before = memory_snapshot()
    started = perf_counter()
    ensure_memory(2 * model_path.stat().st_size)
    if model_label == "GN":
        from .google_news import load_gn_vectors
        model = load_gn_vectors(model_path, model_format=gn_format)
    else:
        from gensim.models import Word2Vec
        model = Word2Vec.load(str(model_path)).wv
    model_load_seconds = perf_counter() - started
    if model.vector_size != 300:
        raise ValueError("Assignment models must have 300-dimensional word vectors")
    directory = dense_index_dir / f"{collection}_{model_label}"
    reused = directory.exists()
    print(f"Dense {collection}/{model_label}: {'loading cached document vectors' if reused else 'building document vectors'}", flush=True)
    if not reused:
        directory.mkdir(parents=True, exist_ok=False)
        matrix = np.lib.format.open_memmap(directory / "documents.npy", mode="w+",
                                          dtype=np.float32, shape=(len(selected), 300))
        docnos, details = [], {}
        started = perf_counter()
        try:
            documents = matched_texts(dataset, selected, docnos, details, hdf_key, chunk_size)
            dense_details = build_dense_documents(documents, docnos, model, vectorizer, matrix,
                                                   google_news=case_lookup,
                                                   block_size=embedding_block_size)
            if set(docnos) != set(selected["docno"]):
                raise ValueError("Dense document vectors do not match official docnos")
            matrix.flush()
        finally:
            matrix._mmap.close()
        write_json(directory / "docnos.json", docnos)
        metadata = {"provenance": provenance, "documents": len(docnos),
                    "idf_document_count": len(official), "google_news": case_lookup,
                    "embedding_lookup": "Part 2 case variants" if case_lookup else "Direct preprocessed token",
                    "idf_convention": "sklearn smooth IDF: 1+ln((N+1)/(df+1)); df=0 for unseen query terms",
                    "term_frequency": "Raw token counts for documents and queries; no sublinear TF",
                    "normalized": True, "embedding_seconds": perf_counter() - started,
                    "document_matrix_bytes": len(docnos) * 300 * 4,
                    "documents_sha256": sha256_file(directory / "documents.npy"),
                    "docnos_sha256": sha256_file(directory / "docnos.json"),
                    "memory_after_embedding": memory_snapshot(), **details, **dense_details}
        write_json(directory / "dense_index.json", metadata)  # Completion marker is written last.
    index = load_dense_index(directory, provenance, model, vectorizer)
    try:
        if set(index.docnos) != set(selected["docno"]):
            raise ValueError("Dense cache does not match official docnos")
        output_dir.mkdir(parents=True, exist_ok=False)
        runs = []
        for query_set, queries in [("known_item", known), ("topics", topics)]:
            filename = f"{collection.lower()}_{model_label.lower()}_{query_set}.csv"
            metrics = run_dense_queries(index, queries, output_dir / filename, query_batch_size)
            runs.append({"model": model_label, "collection": collection, "query_set": query_set, **metrics})
            print(f"{filename}: {metrics['queries']} queries, {metrics['rows']} ranked rows; OOV token fraction {metrics['oov']['oov_token_fraction']:.4f}", flush=True)
        summary = {"model": model_label, "collection": collection, "smoke": smoke,
                   "document_cache": str(directory), "cache_reused": reused,
                   "model_loading_seconds": model_load_seconds, "document_embeddings": index.metadata,
                   "runs": runs, "memory_before": before, "memory_after": memory_snapshot()}
        write_json(output_dir / "dense_summary.json", summary)
        return summary
    finally:
        index.close()


def build_index(model, documents, docnos):
    """Fit only this collection; the analyzer is shared with query transform."""
    if model == "tfidf":
        vectorizer = TfidfVectorizer(analyzer=preprocess_text, sublinear_tf=True,
                                     norm="l2", use_idf=True, smooth_idf=True,
                                     dtype=np.float64)
        matrix = vectorizer.fit_transform(documents)
        ensure_memory(matrix.data.nbytes + matrix.indices.nbytes + 8 * sum(matrix.shape))
        weights = matrix.tocsc()
        extra = model_parameters(model)
    elif model == "bm25":
        vectorizer = CountVectorizer(analyzer=preprocess_text, dtype=np.int32)
        counts = vectorizer.fit_transform(documents)
        weights, average = bm25_weights(counts)
        extra = {**model_parameters(model), "average_document_length": average}
    else:
        raise ValueError(f"Unknown model: {model}")
    return RetrievalIndex(model, vectorizer, weights, np.asarray(docnos, dtype=str), extra)


def save_index(directory, index, provenance):
    directory = Path(directory)
    directory.mkdir()  # Exclusive creation; a missing manifest marks an incomplete index.
    with (directory / "weights.npz").open("xb") as handle:
        sparse.save_npz(handle, index.weights, compressed=False)
    with (directory / "vectorizer.joblib").open("xb") as handle:
        joblib.dump(index.vectorizer, handle)
    write_json(directory / "docnos.json", index.docnos.tolist())
    matrix_bytes = sum(a.nbytes for a in [index.weights.data, index.weights.indices, index.weights.indptr])
    index.metadata.update(index_version=INDEX_VERSION, model=index.model,
                          provenance=provenance, documents=len(index.docnos),
                          vocabulary_size=len(index.vectorizer.vocabulary_),
                          nonzero_postings=int(index.weights.nnz),
                          sparse_matrix_bytes=int(matrix_bytes))
    write_json(directory / "index.json", index.metadata)


def load_index(directory, provenance, model):
    directory = Path(directory)
    with (directory / "index.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if (metadata["index_version"] != INDEX_VERSION or metadata["model"] != model
            or metadata["provenance"] != provenance):
        raise ValueError(f"Index configuration/input mismatch: {directory}")
    with (directory / "docnos.json").open(encoding="utf-8") as handle:
        docnos = np.asarray(json.load(handle), dtype=str)
    ensure_memory(2 * metadata.get("sparse_matrix_bytes", 0))
    vectorizer = joblib.load(directory / "vectorizer.joblib")
    weights = sparse.load_npz(directory / "weights.npz")
    if weights.format != "csc":
        raise ValueError("Expected CSC term postings")
    return RetrievalIndex(model, vectorizer, weights, docnos, metadata)


def validate_results(path, expected_qids, official_docnos, top_k=TOP_K):
    """Stream a saved run and enforce the complete ranking/identifier contract."""
    expected_qids = list(expected_qids)
    expected = set(expected_qids)
    allowed = set(official_docnos)
    if not expected or len(expected) != len(expected_qids):
        raise ValueError("Expected query IDs must be nonempty and unique")
    seen_queries, seen_documents = set(), set()
    current, rank, rows, previous = None, 0, 0, float("inf")
    columns = ["qid", "docno", "rank", "score"]
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError("Unexpected result CSV columns")
        for row in reader:
            if None in row or any(row.get(key) in (None, "") for key in columns):
                raise ValueError("Missing result values or extra columns")
            qid, docno = row["qid"], row["docno"]
            if qid != current:
                if current is not None and rank != top_k:
                    raise ValueError(f"Expected {top_k} results for {current}")
                if qid not in expected or qid in seen_queries:
                    raise ValueError(f"Unexpected or repeated query group: {qid}")
                current, rank, previous = qid, 0, float("inf")
                seen_queries.add(qid)
                seen_documents = set()
            value = float(row["score"])
            if (int(row["rank"]) != rank + 1 or rank >= top_k
                    or not np.isfinite(value) or value > previous):
                raise ValueError(f"Invalid rank or score for {qid}")
            if docno not in allowed or docno in seen_documents:
                raise ValueError(f"Invalid or duplicate official docno for {qid}: {docno}")
            seen_documents.add(docno)
            rank, previous, rows = rank + 1, value, rows + 1
    if rank != top_k or seen_queries != expected or rows != len(expected) * top_k:
        raise ValueError("Missing queries or incorrect result row count")
    return {"passed": True, "queries": len(seen_queries), "rows": rows,
            "unique_official_documents_per_query": top_k}


def run_queries(index, queries, output):
    """Publish a validated CSV exclusively, retaining genuine zero-score ties."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing existing results: {output}")
    if queries.empty or queries["qid"].duplicated().any():
        raise ValueError("Queries must be nonempty with unique IDs")
    seconds = 0.0
    zero_queries = below_100_positive = rows = 0
    minimum = TOP_K
    staged = output.parent / f".{output.name}.in_progress"
    handle = staged.open("x", encoding="utf-8", newline="")
    try:
        with handle:
            writer = csv.writer(handle)
            writer.writerow(["qid", "docno", "rank", "score"])
            for number, (qid, text) in enumerate(queries[["qid", "query"]].itertuples(index=False, name=None)):
                if number % 50 == 0:
                    ensure_memory(3 * len(index.docnos) * 8)
                start = perf_counter()
                docnos, scores = index.search(text)
                seconds += perf_counter() - start
                minimum = min(minimum, len(docnos))
                zero_queries += not np.any(scores)
                below_100_positive += np.count_nonzero(scores > 0) < TOP_K
                for rank, (docno, score) in enumerate(zip(docnos, scores), 1):
                    writer.writerow([qid, docno, rank, format(float(score), ".17g")])
                    rows += 1
        verification = validate_results(staged, queries["qid"], index.docnos)
        os.link(staged, output)
    finally:
        staged.unlink()  # Only this attempt's exclusively created staging file.
    return {"filename": Path(output).name, "queries": len(queries), "rows": rows,
            "minimum_documents_per_query": minimum, "all_queries_returned_100": minimum == TOP_K,
            "query_seconds": seconds, "mean_query_seconds": seconds / len(queries),
            "zero_score_queries": int(zero_queries),
            "queries_with_fewer_than_100_positive_scores": int(below_100_positive),
            "verification": verification,
            "timing": "Query preprocessing, scoring, and ranking; excludes CSV writing",
            "memory_after_queries": memory_snapshot()}


def run_expansion_queries(index, queries, expander, output, m, expansion_lambda):
    """Retrieve using cached neighbors and the unchanged sparse BM25 postings."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing existing expansion results: {output}")
    if queries.empty or queries["qid"].duplicated().any():
        raise ValueError("Use nonempty queries with unique IDs")
    staged = output.parent / f".{output.name}.in_progress"
    seconds, fallbacks, short_lists = 0.0, 0, 0
    handle = staged.open("x", encoding="utf-8", newline="")
    try:
        with handle:
            writer = csv.writer(handle)
            writer.writerow(["qid", "docno", "rank", "score"])
            for qid, query in queries[["qid", "query"]].itertuples(index=False, name=None):
                ensure_memory(3 * len(index.docnos) * 8)
                started = perf_counter()
                weights, details = expander.expand(query, m, expansion_lambda)
                docnos, scores = index.search_weighted(weights)
                seconds += perf_counter() - started
                fallbacks += details["fallback"]
                short_lists += not details["fallback"] and len(details["expansion"]) < m
                for rank, (docno, score) in enumerate(zip(docnos, scores), 1):
                    writer.writerow([qid, docno, rank, format(float(score), ".17g")])
        verification = validate_results(staged, queries["qid"], index.docnos)
        os.link(staged, output)
    finally:
        staged.unlink()
    return {"filename": output.name, "m": m, "lambda": expansion_lambda,
            "queries": len(queries), "verification": verification,
            "fallback_queries": int(fallbacks), "queries_with_fewer_expansions": int(short_lists),
            "query_seconds": seconds, "mean_query_seconds": seconds / len(queries),
            "query_timing": "Cached expansion weighting, BM25 scoring and ranking; excludes CSV I/O",
            "memory_after_queries": memory_snapshot()}


def run_expansion_pipeline(official_dir, index_dir, output_dir, embedding_model, model_label,
                           *, collection="C1", smoke=False, expansion_sizes=(5, 10, 20),
                           expansion_lambdas=(0.5, 1.0), expansion_cache=None):
    """Part 3.3: one model/collection pair, loading existing indexes only.

    Smoke mode selects three known-item and three topic queries against the
    saved index. No document text is read and no index is built. The independent
    m/lambda options allow P1 tuning followed by a fixed configuration on P2.
    GN must be the local native subset keyed by original preprocessed tokens.
    """
    if collection not in EXPECTED_COLLECTION_COUNTS or model_label not in {"M1", "M2", "GN"}:
        raise ValueError("Expansion requires C1/C2 and M1/M2/GN")
    sizes, lambdas = sorted(set(expansion_sizes)), sorted(set(expansion_lambdas))
    if not sizes or not lambdas or not set(sizes) <= {5, 10, 20} or not set(lambdas) <= {0.5, 1.0}:
        raise ValueError("Expansion requires m in {5,10,20} and lambda in {0.5,1.0}")
    official_dir, index_dir = Path(official_dir).resolve(), Path(index_dir).resolve()
    output_dir = external_path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Use a new expansion results directory: {output_dir}")
    period = {"C1": "P1", "C2": "P2"}[collection]
    collection_path = official_dir / f"collection_{period}.tsv"
    official = read_table(collection_path, ["docno", "web_url"], "docno")
    directory = index_dir / f"{collection}_bm25"
    metadata = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    preprocessing_hash = sha256_file(REPOSITORY / "src/preprocessing.py")
    if (len(official) != EXPECTED_COLLECTION_COUNTS[collection]
            or metadata["documents"] != len(official) or metadata["provenance"]["smoke"]
            or metadata["provenance"]["collection"] != collection
            or metadata["provenance"]["collection_sha256"] != sha256_file(collection_path)
            or metadata["provenance"]["preprocessing_sha256"] != preprocessing_hash
            or metadata["provenance"]["model_parameters"] != model_parameters("bm25")):
        raise ValueError("Expansion needs the matching full collection's Appendix B BM25 index")
    known_path = official_dir / f"queries_ki_{period}.tsv"
    topics_path = official_dir / "topics.tsv"
    known = read_table(known_path, ["qid", "query"], "qid")
    topics = read_table(topics_path, ["tid", "query"], "tid").rename(columns={"tid": "qid"})
    if len(known) != 1000 or len(topics) != 32:
        raise ValueError("Expected 1,000 known-item queries and 32 topics")
    if smoke:
        known, topics = known.iloc[:3].copy(), topics.iloc[:3].copy()
    queries = pd.concat([known, topics], ignore_index=True)
    if queries["qid"].duplicated().any():
        raise ValueError("Known-item and topic query IDs must be disjoint")
    model_path = Path(embedding_model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model_files = [model_path, *sorted(model_path.parent.glob(model_path.name + ".*"))]
    provenance = {"expansion_version": 1, "collection": collection, "model": model_label,
                  "numpy_version": np.__version__,
                  "preprocessing_sha256": preprocessing_hash,
                  "index_files_sha256": {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()},
                  "model_files_sha256": {p.name: sha256_file(p) for p in model_files if p.is_file()},
                  "lookup": "Direct preprocessed tokens",
                  "centroid": "Mean unit word vectors over token occurrences; no IDF"}
    cache_path = external_path(expansion_cache) if expansion_cache else (
        external_path(index_dir) / f"{collection}_{model_label}_expansion_candidates{'_smoke' if smoke else ''}.json")
    if cache_path == model_path or cache_path.is_relative_to(directory) or cache_path.is_relative_to(output_dir):
        raise ValueError("Use a separate external expansion-candidate cache path")
    before = memory_snapshot()
    index = load_index(directory, metadata["provenance"], "bm25")
    if (set(index.docnos) != set(official["docno"])
            or index.vectorizer.analyzer is not preprocess_text):
        raise ValueError("BM25 cache has incompatible docnos or preprocessing")
    ensure_memory(2 * model_path.stat().st_size)
    started = perf_counter()
    if model_label == "GN":
        from .google_news import load_gn_vectors
        model = load_gn_vectors(model_path, model_format="keyedvectors")
    else:
        from gensim.models import Word2Vec
        model = Word2Vec.load(str(model_path)).wv
    if model.vector_size != 300:
        raise ValueError("Assignment models must have 300-dimensional vectors")
    model_seconds = perf_counter() - started
    expander = QueryExpander(model, index.vectorizer.vocabulary_)
    expected_bytes = cache_path.read_bytes() if cache_path.exists() else None
    if expected_bytes is not None:
        expander.load_cache(cache_path, provenance)
    cached_before = len(expander.cache)
    started = perf_counter()
    for query in queries["query"]:
        expander.neighbors(query)
    neighbor_seconds = perf_counter() - started
    if expected_bytes is None or len(expander.cache) != cached_before:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        expander.save_cache(cache_path, provenance, expected_bytes)
    output_dir.mkdir(parents=True, exist_ok=False)
    runs = []
    for m in sizes:
        for expansion_lambda in lambdas:
            filename = f"{collection.lower()}_bm25_qe_{model_label.lower()}_m{m}_lambda{expansion_lambda:g}.csv"
            metrics = run_expansion_queries(index, queries, expander, output_dir / filename, m, expansion_lambda)
            runs.append(metrics)
            print(f"{filename}: {metrics['verification']['queries']} queries, {metrics['verification']['rows']} rows, "
                  f"{metrics['fallback_queries']} fallbacks", flush=True)
    examples = []
    for qid, query in topics[["qid", "query"]].iloc[:3].itertuples(index=False, name=None):
        _, details = expander.expand(query, sizes[0], lambdas[0])
        examples.append({"qid": qid, "query": query, "m": sizes[0], "lambda": lambdas[0], **details})
    summary = {"model": model_label, "collection": collection, "smoke": smoke,
               "document_index_rebuilt": False, "provenance": provenance,
               "candidate_cache": str(cache_path), "cached_queries_before": cached_before,
               "new_neighbor_searches": expander.neighbor_searches,
               "neighbor_seconds": neighbor_seconds, "model_loading_seconds": model_seconds,
               "m_values": sizes, "lambda_values": lambdas, "runs": runs,
               "query_file_sha256": {p.name: sha256_file(p) for p in [known_path, topics_path]},
               "query_ids": {"known_item": known["qid"].tolist(), "topics": topics["qid"].tolist()},
               "topic_examples": examples, "memory_before": before, "memory_after": memory_snapshot(),
               "candidate_rule": "Collection vocabulary with nonzero model vectors; exclude all original tokens; "
                                 "highest cosine first, lexical ties, retain negative cosines; at most m terms",
               "fallback_rule": "No usable or cancelling centroid: distinct original BM25 terms at weight 1"}
    write_json(output_dir / "expansion_summary.json", summary)
    return summary


FUSION_ALPHAS = tuple(i / 10 for i in range(11))


def fusion_candidates(bm25_top, dense_top):
    """Sorted union of original rankings, retaining at most 200 candidates.

    The union lets dense retrieval introduce documents outside BM25's top 100.
    Every member subsequently receives both actual scores for linear fusion.
    """
    if (not len(bm25_top) or not len(dense_top) or len(set(bm25_top)) != len(bm25_top)
            or len(set(dense_top)) != len(dense_top)):
        raise ValueError("Fusion needs nonempty rankings without duplicate documents")
    return np.asarray(sorted(set(bm25_top) | set(dense_top)), dtype=str)


def minmax_scores(scores):
    """Normalize one model across the entire union; an equal-score model is zero."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Min-max normalization requires nonempty finite scores")
    low, high = scores.min(), scores.max()
    return (scores - low) / (high - low) if high != low else np.zeros_like(scores)


def linear_fusion(bm25_scores, dense_scores, alpha):
    """Part 3.4 linear fusion with independent, per-query union normalization.

    Alpha=1 preserves BM25 score order; alpha=0 preserves dense score order.
    Ranking breaks ties by ascending official docno, including a constant-score
    model's all-zero normalization. Float32 cosine recomputation can reorder
    numerically indistinguishable dense hits relative to a saved batched run.
    """
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Fusion alpha must lie in [0,1]")
    bm25, dense = minmax_scores(bm25_scores), minmax_scores(dense_scores)
    if bm25.shape != dense.shape:
        raise ValueError("Fusion score vectors must describe the same candidates")
    return alpha * bm25 + (1 - alpha) * dense


def rrf_scores(candidates, bm25_top, dense_top):
    """RRF with k=60 and original 1-based ranks; absent documents add zero."""
    contributions = dict.fromkeys(candidates, 0.0)
    for ranking in [bm25_top, dense_top]:
        if len(set(ranking)) != len(ranking):
            raise ValueError("RRF source rankings must not contain duplicate documents")
        for rank, docno in enumerate(ranking, 1):
            if docno in contributions:
                contributions[docno] += 1 / (60 + rank)
    return np.asarray([contributions[docno] for docno in candidates], dtype=np.float64)


def score_fusion_candidates(query, bm25, dense, bm25_top, dense_top, *,
                            query_vector=None, bm25_positions=None, dense_positions=None):
    """Recompute BOTH models on every union member, including missing-side hits."""
    candidates = fusion_candidates(bm25_top, dense_top)
    if query_vector is None:
        vectors, _ = embed_query_batch([query], dense.model, dense.tfidf_vectorizer,
                                       dense.idf_document_count, google_news=dense.google_news)
        query_vector = vectors[0]
    return {"docnos": candidates,
            "bm25": bm25.score_candidates(query, candidates, bm25_positions),
            "dense": dense.score_candidates(query_vector, candidates, dense_positions),
            "rrf": rrf_scores(candidates, bm25_top, dense_top)}


def load_rankings(path, queries, official_docnos):
    """Read validated original top-100 runs; preserve their ranks for RRF."""
    validate_results(path, queries["qid"], official_docnos)
    table = pd.read_csv(path, dtype={"qid": str, "docno": str, "rank": np.int64, "score": np.float64})
    return {qid: (group["docno"].to_numpy(dtype=str), group["score"].to_numpy())
            for qid, group in table.groupby("qid", sort=False)}


def run_hybrid_pipeline(official_dir, index_dir, input_results_dir, output_dir, embedding_model,
                        model_label, *, collection="C1", smoke=False, alphas=FUSION_ALPHAS,
                        query_batch_size=16):
    """Fuse saved top-100 runs using saved indexes and exact candidate scores.

    Candidate scores are computed once per query and saved for inspection and
    reuse. No document indexes or document embeddings are constructed. Smoke
    mode uses three known-item and three topic queries from the full saved runs.
    """
    alphas = tuple(sorted(set(alphas)))
    if (collection not in EXPECTED_COLLECTION_COUNTS or model_label not in {"M1", "M2", "GN"}
            or not alphas or any(alpha not in FUSION_ALPHAS for alpha in alphas)
            or query_batch_size <= 0):
        raise ValueError("Hybrid retrieval needs C1/C2, M1/M2/GN, tenths in [0,1], and a positive batch size")
    official_dir, index_dir = Path(official_dir).resolve(), Path(index_dir).resolve()
    input_results_dir, output_dir = Path(input_results_dir).resolve(), external_path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Use a new hybrid results directory: {output_dir}")
    period = {"C1": "P1", "C2": "P2"}[collection]
    collection_path = official_dir / f"collection_{period}.tsv"
    official = read_table(collection_path, ["docno", "web_url"], "docno")
    known_path, topics_path = official_dir / f"queries_ki_{period}.tsv", official_dir / "topics.tsv"
    known = read_table(known_path, ["qid", "query"], "qid")
    topics = read_table(topics_path, ["tid", "query"], "tid").rename(columns={"tid": "qid"})
    if len(official) != EXPECTED_COLLECTION_COUNTS[collection] or len(known) != 1000 or len(topics) != 32:
        raise ValueError("Unexpected official collection or query counts")
    queries = pd.concat([known, topics], ignore_index=True)
    if queries["qid"].duplicated().any():
        raise ValueError("Hybrid query IDs must be unique")
    bm25_path = input_results_dir / f"{collection.lower()}_bm25.csv"
    dense_path = input_results_dir / f"{collection.lower()}_dense_{model_label.lower()}.csv"
    bm25_runs = load_rankings(bm25_path, queries, official["docno"])
    dense_runs = load_rankings(dense_path, queries, official["docno"])
    if smoke:
        queries = pd.concat([known.iloc[:3], topics.iloc[:3]], ignore_index=True)
    bm25_dir, tfidf_dir = index_dir / f"{collection}_bm25", index_dir / f"{collection}_tfidf"
    dense_dir = index_dir / f"{collection}_{model_label}"
    bm25_meta = json.loads((bm25_dir / "index.json").read_text(encoding="utf-8"))
    tfidf_meta = json.loads((tfidf_dir / "index.json").read_text(encoding="utf-8"))
    dense_meta = json.loads((dense_dir / "dense_index.json").read_text(encoding="utf-8"))
    preprocessing_hash = sha256_file(REPOSITORY / "src/preprocessing.py")
    collection_hash = sha256_file(collection_path)
    for metadata in [bm25_meta, tfidf_meta]:
        if (metadata["documents"] != len(official) or metadata["provenance"]["smoke"]
                or metadata["provenance"]["collection"] != collection
                or metadata["provenance"]["collection_sha256"] != collection_hash
                or metadata["provenance"]["preprocessing_sha256"] != preprocessing_hash):
            raise ValueError("Hybrid lexical index collection/preprocessing mismatch")
    provenance = dense_meta["provenance"]
    if (bm25_meta["provenance"]["model_parameters"] != model_parameters("bm25")
            or dense_meta["documents"] != len(official) or not dense_meta["normalized"]
            or provenance["collection"] != collection or provenance["model"] != model_label
            or provenance["smoke"] or provenance["preprocessing_sha256"] != preprocessing_hash
            or provenance["tfidf_metadata_sha256"] != sha256_file(tfidf_dir / "index.json")
            or provenance["idf_vectorizer_sha256"] != sha256_file(tfidf_dir / "vectorizer.joblib")
            or not (provenance["dataset_sha256"] == bm25_meta["provenance"]["dataset_sha256"]
                    == tfidf_meta["provenance"]["dataset_sha256"])):
        raise ValueError("Hybrid requires matching saved BM25 and normalized dense indexes")
    model_path = Path(embedding_model).resolve()
    model_files = [model_path, *sorted(model_path.parent.glob(model_path.name + ".*"))]
    model_hashes = {p.name: sha256_file(p) for p in model_files if p.is_file()}
    if not model_path.is_file() or model_hashes != provenance["model_file_sha256"]:
        raise ValueError("Embedding model differs from the saved dense index")
    if model_label == "GN" and (dense_meta["google_news"] or not provenance.get("gn_preprocessed")):
        raise ValueError("Hybrid GN requires the directly keyed local subset")
    before = memory_snapshot()
    started = perf_counter()
    bm25 = load_index(bm25_dir, bm25_meta["provenance"], "bm25")
    vectorizer = joblib.load(tfidf_dir / "vectorizer.joblib")
    if (bm25.vectorizer.analyzer is not preprocess_text or vectorizer.analyzer is not preprocess_text
            or not vectorizer.smooth_idf or not vectorizer.use_idf):
        raise ValueError("Hybrid indexes must use shared preprocessing and collection smooth IDF")
    ensure_memory(2 * model_path.stat().st_size)
    if model_label == "GN":
        from .google_news import load_gn_vectors
        model = load_gn_vectors(model_path, model_format="keyedvectors")
    else:
        from gensim.models import Word2Vec
        model = Word2Vec.load(str(model_path)).wv
    if model.vector_size != 300:
        raise ValueError("Assignment embeddings must be 300-dimensional")
    dense = load_dense_index(dense_dir, provenance, model, vectorizer)
    try:
        if set(bm25.docnos) != set(official["docno"]) or set(dense.docnos) != set(official["docno"]):
            raise ValueError("Hybrid index docnos differ from the official collection")
        loading_seconds = perf_counter() - started
        bm_positions = dict(zip(bm25.docnos, range(len(bm25.docnos))))
        dense_positions = dict(zip(dense.docnos, range(len(dense.docnos))))
        records, shared_seconds, zero_queries = [], 0.0, 0
        missing_bm25, missing_dense = 0, 0
        for first in range(0, len(queries), query_batch_size):
            batch = queries.iloc[first:first + query_batch_size]
            ensure_memory()
            started = perf_counter()
            embeddings, stats = embed_query_batch(batch["query"].tolist(), model, vectorizer,
                                                   dense.idf_document_count, google_news=dense.google_news)
            shared_seconds += perf_counter() - started
            zero_queries += stats["queries_without_embedding"]
            for (qid, query), embedding in zip(batch[["qid", "query"]].itertuples(index=False, name=None), embeddings):
                started = perf_counter()
                bm_docs, bm_scores = bm25_runs[qid]
                de_docs, de_scores = dense_runs[qid]
                record = score_fusion_candidates(query, bm25, dense, bm_docs, de_docs,
                                                  query_vector=embedding, bm25_positions=bm_positions,
                                                  dense_positions=dense_positions)
                shared_seconds += perf_counter() - started
                positions = dict(zip(record["docnos"], range(len(record["docnos"]))))
                if (not np.allclose(record["bm25"][[positions[d] for d in bm_docs]], bm_scores, rtol=1e-12, atol=1e-10)
                        or not np.allclose(record["dense"][[positions[d] for d in de_docs]], de_scores, rtol=0, atol=2e-6)):
                    raise ValueError(f"Recomputed hybrid scores disagree with the saved runs for {qid}")
                missing_bm25 += len(set(de_docs) - set(bm_docs))
                missing_dense += len(set(bm_docs) - set(de_docs))
                records.append({"qid": qid, **record})
        output_dir.mkdir(parents=True, exist_ok=False)
        # Auditable exact scores, saved once; all alpha values reuse these arrays.
        offsets = np.cumsum([0, *[len(record["docnos"]) for record in records]], dtype=np.int64)
        with (output_dir / "candidate_scores.npz").open("xb") as handle:
            np.savez_compressed(handle, qids=queries["qid"].to_numpy(dtype=str), offsets=offsets,
                                **{key: np.concatenate([record[key] for record in records])
                                   for key in ["docnos", "bm25", "dense", "rrf"]})
        methods = [(f"a{alpha:.1f}".replace(".", "p"), alpha) for alpha in alphas] + [("rrf", None)]
        runs, staged_files = [], []
        try:
            with ExitStack() as stack:
                writers, timings = {}, dict.fromkeys([name for name, _ in methods], 0.0)
                for name, alpha in methods:
                    filename = (f"{collection.lower()}_rrf_{model_label.lower()}.csv" if alpha is None else
                                f"{collection.lower()}_hybrid_{model_label.lower()}_{name}.csv")
                    target = output_dir / filename
                    staged = output_dir / f".{filename}.in_progress"
                    handle = stack.enter_context(staged.open("x", encoding="utf-8", newline=""))
                    staged_files.append(staged)
                    writers[name] = csv.writer(handle)
                    writers[name].writerow(["qid", "docno", "rank", "score"])
                    runs.append({"method": "rrf" if alpha is None else "linear", "alpha": alpha,
                                 "filename": filename})
                for record in records:
                    tie_order = np.arange(len(record["docnos"]))  # Union is already lexical.
                    for name, alpha in methods:
                        started = perf_counter()
                        scores = record["rrf"] if alpha is None else linear_fusion(record["bm25"], record["dense"], alpha)
                        docnos, ranked_scores = rank_scores(scores, record["docnos"], tie_order)
                        timings[name] += perf_counter() - started
                        for rank, (docno, score) in enumerate(zip(docnos, ranked_scores), 1):
                            writers[name].writerow([record["qid"], docno, rank, format(float(score), ".17g")])
            for run, (name, _) in zip(runs, methods):
                staged = output_dir / f".{run['filename']}.in_progress"
                run.update(verification=validate_results(staged, queries["qid"], official["docno"]),
                           fusion_seconds=timings[name], shared_scoring_seconds=shared_seconds,
                           mean_query_seconds=(shared_seconds + timings[name]) / len(queries))
            for run in runs:
                staged, target = output_dir / f".{run['filename']}.in_progress", output_dir / run["filename"]
                os.link(staged, target)
                run["sha256"] = sha256_file(target)
        finally:
            for staged in staged_files:
                staged.unlink(missing_ok=True)
        summary = {"model": model_label, "collection": collection, "smoke": smoke,
                   "queries": len(queries), "runs": runs, "alphas": list(alphas), "rrf_k": 60,
                   "indexes_rebuilt": False, "document_embeddings_rebuilt": False,
                   "loading_seconds": loading_seconds, "shared_scoring_seconds": shared_seconds,
                   "candidate_union_sizes": {"minimum": min(np.diff(offsets).tolist()),
                                             "maximum": max(np.diff(offsets).tolist())},
                   "exact_missing_side_scores": {"bm25": missing_bm25, "dense": missing_dense},
                   "saved_source_scores_agree": True, "queries_without_embedding": zero_queries,
                   "candidate_cache": {"path": str(output_dir / "candidate_scores.npz"),
                                       "sha256": sha256_file(output_dir / "candidate_scores.npz")},
                   "source_sha256": {str(p): sha256_file(p) for p in [bm25_path, dense_path, known_path, topics_path,
                                        collection_path, bm25_dir / "index.json", dense_dir / "dense_index.json"]},
                   "dense_provenance": provenance, "memory_before": before, "memory_after": memory_snapshot(),
                   "timing_note": "Per-result latency includes query embedding, union construction, exact candidate "
                                  "BM25/cosine scoring, normalization, fusion and ranking. Shared scoring is performed "
                                  "only once. Excludes loading, CSV I/O, validation and the original first-stage "
                                  "retrieval that produced the saved top-100 lists.",
                   "normalization": "Independent min-max over the entire union; exactly equal scores become zero",
                   "rrf_rule": "Original 1-based top-100 ranks only; absent documents contribute zero",
                   "tie_rule": "Descending fused score, then ascending official docno"}
        write_json(output_dir / "hybrid_summary.json", summary)
        print(f"Hybrid {collection}/{model_label}: {len(runs)} validated results, {len(queries)} queries each", flush=True)
        return summary
    finally:
        dense.close()


def run_pipeline(dataset, official_dir, index_dir, output_dir, *, hdf_key="data",
                 chunk_size=10000, smoke=False, collection_names=("C1", "C2"),
                 combined_output=False, append_results=False):
    dataset, official_dir = Path(dataset).resolve(), Path(official_dir).resolve()
    index_dir, output_dir = external_path(index_dir), external_path(output_dir)
    if chunk_size <= 0 or index_dir == output_dir:
        raise ValueError("Use a positive chunk size and distinct index/results directories")
    summary_path = output_dir / "retrieval_summary.json"
    previous, previous_bytes = None, None
    if append_results:
        previous_bytes = summary_path.read_bytes()
        previous = json.loads(previous_bytes)
    elif output_dir.exists():
        raise FileExistsError(f"Use a new results directory: {output_dir}")
    collection_names = tuple(collection_names)
    if (not collection_names or len(set(collection_names)) != len(collection_names)
            or not set(collection_names).issubset({"C1", "C2"})):
        raise ValueError("Select C1, C2, or both without duplicates")
    collections, queries, collection_hashes = {}, {}, {}
    query_paths = [official_dir / "topics.tsv"]
    for name in collection_names:
        period = {"C1": "P1", "C2": "P2"}[name]
        path = official_dir / f"collection_{period}.tsv"
        table = read_table(path, ["docno", "web_url"], "docno")
        if len(table) != EXPECTED_COLLECTION_COUNTS[name]:
            raise ValueError(f"Unexpected full {name} document count: {len(table)}")
        collections[name] = table.iloc[:300].copy() if smoke else table
        collection_hashes[name] = sha256_file(path)
        query_path = official_dir / f"queries_ki_{period}.tsv"
        query_paths.append(query_path)
        table = read_table(query_path, ["qid", "query"], "qid")
        if len(table) != 1000:
            raise ValueError(f"Expected 1,000 {period} known-item queries")
        queries[name] = table.iloc[:3].copy() if smoke else table
    topics = read_table(official_dir / "topics.tsv", ["tid", "query"], "tid").rename(columns={"tid": "qid"})
    if len(topics) != 32:
        raise ValueError("Expected 32 topic queries")
    if smoke:
        topics = topics.iloc[:3].copy()
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    common = {"dataset_sha256": sha256_file(dataset), "hdf_key": hdf_key,
              "preprocessing_sha256": sha256_file(REPOSITORY / "src/preprocessing.py"),
              "versions": versions, "smoke": smoke}
    summary = {"smoke": smoke, "collections": list(collection_names),
               "top_k": TOP_K, "indexes": {}, "runs": [],
               "versions": versions, "platform": platform.platform(),
               "logical_cpus": os.cpu_count(), "memory_before": memory_snapshot(),
               "memory_note": "RSS snapshots and lifetime process peak, when available; not per-phase peak RAM",
               "memory_headroom_bytes": MEMORY_HEADROOM_BYTES,
               "tie_rule": "Descending score, then ascending official docno; zero-score documents remain eligible",
               "source_paths": {"dataset": str(dataset), "official_dir": str(official_dir),
                                "index_dir": str(index_dir)},
               "query_file_sha256": {p.name: sha256_file(p) for p in query_paths}}
    if previous is not None:
        merge_summaries(previous, summary)  # Validate compatibility before building or writing.
        for metadata in previous["indexes"].values():
            for key in ["dataset_sha256", "preprocessing_sha256", "hdf_key"]:
                if metadata["provenance"][key] != common[key]:
                    raise ValueError(f"Input changed since the earlier run: {key}")
    for name in collection_names:
        for model in ["tfidf", "bm25"]:
            filenames = ([f"{name.lower()}_{model}.csv"] if combined_output else
                         [f"{name}_{model}_{group}.csv" for group in ["known_item", "topics"]])
            for filename in filenames:
                for path in [output_dir / filename, output_dir / f".{filename}.in_progress"]:
                    if path.exists():
                        raise FileExistsError(f"Refusing existing results or active staging file: {path}")
    ensure_memory()
    index_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=append_results)
    for name, collection in collections.items():
        provenance = {**common, "collection_sha256": collection_hashes[name],
                      "collection": name, "selected_documents": len(collection)}
        for model in ["tfidf", "bm25"]:
            key = f"{name}_{model}"
            model_provenance = {**provenance, "model_parameters": model_parameters(model)}
            directory = index_dir / key
            print(f"{key}: {'loading reusable index' if directory.exists() else 'building index'}", flush=True)
            started = perf_counter()
            reused = directory.exists()
            if reused:
                index = load_index(directory, model_provenance, model)
            else:
                docnos, details = [], {}
                before = memory_snapshot()
                documents = matched_texts(dataset, collection, docnos, details, hdf_key, chunk_size)
                index = build_index(model, documents, docnos)
                index.metadata.update(indexing_seconds=perf_counter() - started,
                                      indexing_timing="HDF5 streaming, cleaned URL/text join, preprocessing, vocabulary and sparse weights; excludes persistence",
                                      memory_before_indexing=before,
                                      memory_after_indexing=memory_snapshot(), **details)
                save_index(directory, index, model_provenance)
            if set(index.docnos) != set(collection["docno"]):
                raise ValueError(f"Cached index docnos differ from official {name}")
            summary["indexes"][key] = {**index.metadata, "reused": reused,
                                      "build_or_load_and_persist_seconds": perf_counter() - started,
                                      "directory": str(directory)}
            if combined_output:
                all_queries = pd.concat([queries[name][["qid", "query"]], topics[["qid", "query"]]], ignore_index=True)
                query_sets = [("known_item_and_topics", all_queries)]
            else:
                query_sets = [("known_item", queries[name]), ("topics", topics)]
            for query_name, query_table in query_sets:
                filename = f"{name.lower()}_{model}.csv" if combined_output else f"{name}_{model}_{query_name}.csv"
                output = output_dir / filename
                metrics = run_queries(index, query_table, output)
                summary["runs"].append({"collection": name, "model": model, "query_set": query_name, **metrics})
                print(f"{output.name}: {metrics['queries']} queries, {metrics['rows']} rows", flush=True)
            print(f"{key}: {index.metadata['vocabulary_size']} terms; indexing {index.metadata['indexing_seconds']:.3f}s", flush=True)
            del index
            gc.collect()
    summary["all_queries_returned_100"] = all(r["all_queries_returned_100"] for r in summary["runs"])
    summary["memory_after"] = memory_snapshot()
    if previous is not None:
        summary = merge_summaries(previous, summary)
    publish_summary(summary_path, summary, previous_bytes)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Required for sparse and dense document indexing")
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hdf-key", default="data")
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--combined-output", action="store_true",
                        help="One validated CSV per collection/method, combining known-item and topic queries")
    parser.add_argument("--append-results", action="store_true",
                        help="Add a new collection to an existing results directory and preserve prior measurements")
    parser.add_argument("--collections", dest="collection_names", nargs="+",
                        choices=["C1", "C2"], default=["C1", "C2"])
    parser.add_argument("--dense-model", choices=["M1", "M2", "M2s", "GN"])
    parser.add_argument("--embedding-model", type=Path, help="Existing native Word2Vec or pretrained GN file")
    parser.add_argument("--dense-index-dir", type=Path, help="External reusable document-vector cache")
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--embedding-block-size", type=int, default=2048)
    parser.add_argument("--gn-format", choices=["word2vec", "keyedvectors"], default="word2vec")
    parser.add_argument("--gn-preprocessed", action="store_true",
                        help="GN subset keys are already preprocessed tokens; use direct lookup")
    parser.add_argument("--expansion-model", choices=["M1", "M2", "GN"],
                        help="Part 3.3 weighted BM25 using an existing index; no dataset needed")
    parser.add_argument("--expansion-sizes", nargs="+", type=int, choices=[5, 10, 20], default=[5, 10, 20])
    parser.add_argument("--expansion-lambdas", nargs="+", type=float, choices=[0.5, 1.0], default=[0.5, 1.0])
    parser.add_argument("--expansion-cache", type=Path, help="Reusable lambda-independent top-20 query neighbors")
    parser.add_argument("--hybrid-model", choices=["M1", "M2", "GN"],
                        help="Part 3.4 linear fusion and RRF using saved retrieval artifacts")
    parser.add_argument("--input-results-dir", type=Path, help="Saved BM25 and dense top-100 CSV directory")
    parser.add_argument("--fusion-alphas", nargs="+", type=float, choices=FUSION_ALPHAS, default=list(FUSION_ALPHAS))
    args = parser.parse_args()
    try:
        if args.hybrid_model:
            if (args.dense_model or args.expansion_model or args.dense_index_dir or args.append_results
                    or args.combined_output or len(args.collection_names) != 1
                    or args.embedding_model is None or args.input_results_dir is None):
                parser.error("Hybrid runs need one collection, --embedding-model, and --input-results-dir")
            run_hybrid_pipeline(args.official_dir, args.index_dir, args.input_results_dir, args.output_dir,
                                args.embedding_model, args.hybrid_model, collection=args.collection_names[0],
                                smoke=args.smoke, alphas=args.fusion_alphas, query_batch_size=args.query_batch_size)
        elif args.expansion_model:
            if (args.dense_model or args.dense_index_dir or args.append_results or args.combined_output
                    or len(args.collection_names) != 1 or args.embedding_model is None):
                parser.error("Expansion needs one --collections value and --embedding-model; no dense/append flags")
            run_expansion_pipeline(
                args.official_dir, args.index_dir, args.output_dir, args.embedding_model, args.expansion_model,
                collection=args.collection_names[0], smoke=args.smoke, expansion_sizes=args.expansion_sizes,
                expansion_lambdas=args.expansion_lambdas, expansion_cache=args.expansion_cache,
            )
        elif args.dataset is None:
            parser.error("Sparse and dense indexing require --dataset")
        elif args.dense_model:
            if len(args.collection_names) != 1 or args.embedding_model is None or args.dense_index_dir is None:
                parser.error("Dense runs require one --collections value, --embedding-model, and --dense-index-dir")
            if args.append_results or args.combined_output:
                parser.error("Dense runs use separate new results and per-query-set OOV statistics")
            run_dense_pipeline(
                args.dataset, args.official_dir, args.index_dir, args.dense_index_dir,
                args.output_dir, args.embedding_model, args.dense_model,
                collection=args.collection_names[0], smoke=args.smoke, hdf_key=args.hdf_key,
                chunk_size=args.chunk_size, query_batch_size=args.query_batch_size,
                embedding_block_size=args.embedding_block_size, gn_format=args.gn_format,
                gn_preprocessed=args.gn_preprocessed,
            )
        else:
            if (args.embedding_model or args.dense_index_dir or args.gn_preprocessed or args.expansion_cache
                    or args.input_results_dir):
                parser.error("Embedding paths require --dense-model")
            options = vars(args).copy()
            for key in ["dense_model", "embedding_model", "dense_index_dir", "query_batch_size",
                        "embedding_block_size", "gn_format", "gn_preprocessed", "expansion_model",
                        "expansion_sizes", "expansion_lambdas", "expansion_cache", "hybrid_model",
                        "input_results_dir", "fusion_alphas"]:
                options.pop(key)
            run_pipeline(**options)
    except MemoryError as error:
        print(f"Stopped for insufficient memory: {error}. Completed indexes/results were preserved.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
