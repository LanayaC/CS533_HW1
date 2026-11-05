"""The two known-item eligibility checks in Appendix A.4."""

import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .document_collections import URL_PATTERN


_ALPHABETIC_TOKEN = re.compile(r"[a-z]+")


def _extract_slug(url: object) -> str | None:
    if not isinstance(url, str):
        return None
    match = URL_PATTERN.match(url)
    return match.group("slug") if match else None


def _qualifying_token_count(slug: object) -> int:
    if not isinstance(slug, str):
        return 0
    return sum(
        1
        for token in slug.lower().split("-")
        if _ALPHABETIC_TOKEN.fullmatch(token)
        and len(token) > 1
        and token not in ENGLISH_STOP_WORDS
    )


def check_known_item_eligibility(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Check slug uniqueness and the three-qualifying-token threshold.

    Supply ALL cleaned articles, not a period or collection subset. Only the
    web_url column is required. Slugs are extracted using Appendix A.2's exact
    URL_PATTERN, with the optional .html suffix excluded. Count occurrences
    of each extracted slug across every supplied row, including articles
    outside C1/C2. Uniqueness compares extracted slug strings exactly; it does
    not depend on the URL's date/path or normalize the slug's case.

    For the second condition, lowercase each slug, split on hyphens, and keep
    only whole tokens matching [a-z]+. Count tokens longer than one character
    that are not in scikit-learn's ENGLISH_STOP_WORDS. Repeated qualifying
    tokens each count; this is not a count of distinct words. Tokens with
    digits, apostrophes, other punctuation, or non-ASCII letters are discarded
    whole. These rules intentionally do not call the Part 1.3 tokenizer.

    Missing/nonmatching URLs have no slug, zero counts, and fail both checks.
    Return one row per input row, preserving index labels (even duplicates)
    and input order. Do not modify the input or perform any further cleaning.

    Output columns are web_url, slug, slug_occurrences, qualifying_token_count,
    unique_slug, enough_qualifying_tokens, and eligible. The last column is
    the conjunction of the two checks. It does NOT check membership in C1/C2,
    rank/select articles, or construct final queries.

    Examples
    --------
    From the repository root, with the complete existing cleaned DataFrame::

        from src.known_item_eligibility import check_known_item_eligibility

        eligibility = check_known_item_eligibility(clean_df)
    """
    result = clean_df.loc[:, ["web_url"]].copy()
    result["slug"] = result["web_url"].map(_extract_slug)
    occurrences = result["slug"].value_counts(dropna=True)
    result["slug_occurrences"] = (
        result["slug"].map(occurrences).fillna(0).astype("int64")
    )
    result["qualifying_token_count"] = (
        result["slug"].map(_qualifying_token_count).astype("int64")
    )
    result["unique_slug"] = result["slug_occurrences"].eq(1)
    result["enough_qualifying_tokens"] = result["qualifying_token_count"].ge(3)
    result["eligible"] = result["unique_slug"] & result["enough_qualifying_tokens"]
    return result
