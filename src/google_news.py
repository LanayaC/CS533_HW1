"""Part 2.4: offline Google News vector loading and shared lowercase lookup.

All later GN comparisons should resolve tokens with lookup_gn_key before
indexing vectors. This module never downloads a model or trains embeddings.
"""

import argparse
from collections.abc import Container
import json
from pathlib import Path

from gensim.models import KeyedVectors


def gn_lookup_candidates(token: str) -> tuple[str, ...]:
    """Return the assignment's ordered case variants for a lowercase token.

    Preserve underscores. Only underscore-separated phrases receive the final
    component-wise capitalization (new_york -> New_York). Deduplicate variants
    without changing their priority. Invalid inputs are errors, not OOV words.
    """
    if not isinstance(token, str):
        raise TypeError("GN lookup requires a lowercase string token.")
    if not token or any(character.isspace() for character in token):
        raise ValueError("GN lookup requires a nonempty token without whitespace; use underscores for phrases.")
    if token != token.lower():
        raise ValueError("GN lookup requires a lowercase token.")
    candidates = [token, token.capitalize(), token.upper()]
    if "_" in token:
        candidates.append("_".join(part.capitalize() for part in token.split("_")))
    return tuple(dict.fromkeys(candidates))


def lookup_gn_key(vocabulary: Container[str], token: str) -> str | None:
    """Return the first existing GN case variant, or None when all are missing.

    Accept KeyedVectors or another case-sensitive membership container. Return
    the actual key so callers use vocabulary[key] for vectors or similarity
    queries. Do not average variants, change vocabulary casing, or guess words.
    """
    for candidate in gn_lookup_candidates(token):
        if candidate in vocabulary:
            return candidate
    return None


def load_gn_vectors(model_path, *, model_format="word2vec") -> KeyedVectors:
    """Load an existing local pretrained file; no network or training calls.

    word2vec: original binary .bin or .bin.gz, read in full as float32.
    keyedvectors: an uncompressed native .kv cache with separate arrays,
    loaded using read-only memory mapping. Keep its array sidecars together.
    """
    if model_format not in {"word2vec", "keyedvectors"}:
        raise ValueError("model_format must be 'word2vec' or 'keyedvectors'.")
    path = Path(model_path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"GN must be a nonempty existing local file: {path}")
    if model_format == "keyedvectors":
        return KeyedVectors.load(str(path), mmap="r")
    return KeyedVectors.load_word2vec_format(str(path), binary=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Existing pretrained GN file")
    parser.add_argument("--model-format", choices=["word2vec", "keyedvectors"], default="word2vec")
    parser.add_argument("--words", nargs="+", required=True, help="Lowercase tokens; underscores for phrases")
    args = parser.parse_args(argv)
    # Reject invalid queries before loading a potentially large file.
    for token in args.words:
        gn_lookup_candidates(token)
    vectors = load_gn_vectors(args.model, model_format=args.model_format)
    lookups = []
    for token in args.words:
        key = lookup_gn_key(vectors, token)
        lookups.append({
            "token": token, "matched_key": key,
            "status": "found" if key is not None else "missing",
            "candidates": gn_lookup_candidates(token),
        })
    print(json.dumps({
        "model_path": str(args.model.expanduser().resolve()),
        "model_format": args.model_format,
        "vocabulary_size": len(vectors), "vector_size": vectors.vector_size,
        "lookups": lookups,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
