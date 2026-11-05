"""Incremental token statistics for CS533 Assignment 1, Part 1.3."""

import pandas as pd

from .preprocessing import preprocess_text


def count_tokens_by_period(
    clean_df: pd.DataFrame,
    text_col: str = "text",
    date_col: str = "pub_date",
) -> pd.DataFrame:
    """Count all tokens and each period's distinct tokens in cleaned articles.

    P1 includes publication years 2005 through 2014; P2 includes 2015
    through 2025. Missing/unparseable dates and years outside those ranges
    are excluded. Missing or empty text contributes no tokens, following
    ``preprocess_text``. Every repeated token and stop word contributes to
    the total. Each period has its own vocabulary, so a word occurring in
    both periods counts as distinct once in each period.

    Dates are parsed once into a publication-year Series. Texts are then
    tokenized one article at a time: only the current article's token list
    and the two vocabulary sets are retained, not a corpus of token lists.
    Exact distinct counts require memory proportional to vocabulary size;
    the year Series requires memory proportional to the article count.
    The input DataFrame is neither modified nor cleaned again.

    Parameters
    ----------
    clean_df : pandas.DataFrame
        Already-cleaned articles containing publication dates and text.
    text_col, date_col : str
        Text and publication-date column names.

    Returns
    -------
    pandas.DataFrame
        Rows ``P1`` and ``P2`` (index name ``period``), with integer columns
        ``total_tokens`` and ``distinct_tokens``. Both rows are present even
        when a period is empty, in which case its counts are zero.

    Examples
    --------
    From the repository root, using the existing cleaned DataFrame::

        from src.token_counts import count_tokens_by_period

        token_counts = count_tokens_by_period(clean_df)
    """
    texts = clean_df[text_col]
    years = pd.to_datetime(
        clean_df[date_col], errors="coerce", format="mixed"
    ).dt.year
    totals = {"P1": 0, "P2": 0}
    vocabularies = {"P1": set(), "P2": set()}

    for year, text in zip(years, texts):
        if pd.isna(year):
            continue
        if 2005 <= year <= 2014:
            period = "P1"
        elif 2015 <= year <= 2025:
            period = "P2"
        else:
            continue

        tokens = preprocess_text(text)
        totals[period] += len(tokens)
        vocabularies[period].update(tokens)
        del tokens

    return pd.DataFrame(
        {
            "total_tokens": [totals["P1"], totals["P2"]],
            "distinct_tokens": [len(vocabularies["P1"]), len(vocabularies["P2"])],
        },
        index=pd.Index(["P1", "P2"], name="period"),
        dtype="int64",
    )
