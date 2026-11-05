"""Monthly article counts for the descriptive analysis in Part 1.1."""

from collections.abc import Iterable

import pandas as pd


def monthly_article_counts(
    articles: pd.DataFrame,
    years: Iterable[int],
    categories: Iterable[str] | None = None,
    date_col: str = "pub_date",
    category_col: str = "mapped_section",
) -> pd.DataFrame:
    """Count article rows for every month of the selected years.

    Parameters
    ----------
    articles : pandas.DataFrame
        The already-loaded article table. It is not modified or deduplicated.
    years : iterable of int
        Years to include. Repeated years are counted once and sorted.
    categories : iterable of str or None
        Exact category labels, in the desired column order. When omitted,
        count all rows, including those with a missing category, in a single
        ``article_count`` column. Repeated labels produce only one column.
    date_col, category_col : str
        Publication-date and category column names. The category column is
        required only when categories are selected.

    Returns
    -------
    pandas.DataFrame
        Integer counts indexed by month-start timestamps (index name
        ``month``), with all 12 months of each requested year. With categories
        selected, each label has its own column; absent categories and months
        receive zeros. Missing or unparseable dates are excluded. Zero means
        no matching records, not proof that the month's coverage is complete.

    Examples
    --------
    Overall monthly coverage in the two anomalous years::

        monthly_article_counts(df, years=[2009, 2025])

    Sports coverage, including 2023 as a comparison year::

        monthly_article_counts(
            df,
            years=[2023, 2024, 2025],
            categories=["Sports & Athletics"],
        )
    """
    # Parse a separate Series; mixed string formats are accepted. Already
    # parsed dates retain their timezone and publication-year interpretation.
    dates = pd.to_datetime(articles[date_col], errors="coerce", format="mixed")
    monthly_index = pd.DatetimeIndex(
        [
            pd.Timestamp(year=year, month=month, day=1)
            for year in sorted(set(years))
            for month in range(1, 13)
        ],
        name="month",
    )
    group_index = pd.MultiIndex.from_arrays(
        [monthly_index.year, monthly_index.month], names=["year", "month"]
    )
    working = pd.DataFrame({"year": dates.dt.year, "month": dates.dt.month})

    if categories is None:
        counts = (
            working.groupby(["year", "month"])
            .size()
            .reindex(group_index, fill_value=0)
            .to_frame("article_count")
        )
    else:
        selected_categories = list(dict.fromkeys(categories))
        working["category"] = articles[category_col]
        working = working.loc[working["category"].isin(selected_categories)]
        counts = (
            working.groupby(["year", "month", "category"], observed=True)
            .size()
            .unstack("category", fill_value=0)
            .reindex(
                index=group_index, columns=selected_categories, fill_value=0
            )
        )
        counts.columns.name = category_col

    counts.index = monthly_index
    return counts.astype("int64")
