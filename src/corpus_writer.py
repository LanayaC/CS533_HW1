"""Incremental, failure-safe Word2Vec corpus writing for Part 1.4."""

from contextlib import ExitStack
import os
from pathlib import Path
import shutil

import pandas as pd

from .preprocessing import preprocess_text


def write_training_corpora(
    clean_df: pd.DataFrame,
    output_dir: str | Path = Path(r"C:\Projects\IR\data\corpora"),
    text_col: str = "text",
    date_col: str = "pub_date",
) -> pd.DataFrame:
    """Write all supplied articles into P1 and P2 Word2Vec training files.

    P1 covers 2005-2014 and P2 covers 2015-2025, inclusively. Every input row
    is written once, preserving its relative order within its period, without
    sampling, deduplication, or input mutation. Dates are validated before
    any files are created: missing, invalid, or out-of-range dates raise
    ValueError rather than silently dropping articles.

    Each article is passed to the existing preprocess_text() and immediately
    written as space-separated tokens plus one LF newline, in UTF-8. Empty
    token lists produce blank lines and still count as articles. Stop words
    and repeated tokens are retained. Memory holds a publication-year Series
    and the current article's tokens, never all tokenized articles.

    Safety protocol
    ---------------
    Existing corpus files are never replaced. Both corpora are first written,
    flushed, synced, and closed inside .corpus_write_in_progress/ under the
    output directory. This directory also guards against concurrent writers.
    The finished files are published using exclusive hard links, so a final
    filename never contains partially written text. This requires a filesystem
    supporting hard links, such as NTFS; there is no copying fallback.

    The pair is complete only when BOTH final files exist and the staging
    directory is ABSENT. Two file publications cannot form one atomic action.
    Caught errors (including KeyboardInterrupt) roll back this attempt's final
    links and remove staging. Forced termination may leave the staging marker
    and zero, one, or two complete final files. A subsequent call refuses to
    proceed until those artifacts are inspected and recovered manually. If
    rollback itself fails, the marker is retained. This does not promise
    recovery from disk corruption or power loss.

    Parameters
    ----------
    clean_df : pandas.DataFrame
        Already-cleaned articles, normally all 970,720 rows. No fixed row count
        is enforced, allowing small synthetic inputs and reusable tests.
    output_dir : str or pathlib.Path
        Created if needed. Defaults to C:\\Projects\\IR\\data\\corpora,
        outside the repository. Files are corpus_P1.txt and corpus_P2.txt.
    text_col, date_col : str
        Text and publication-date column names.

    Returns
    -------
    pandas.DataFrame
        P1 and P2 rows (index name period), with integer article_count and
        token_count columns. Empty periods have empty files and zero counts.

    Examples
    --------
    From the repository root, with the existing cleaned DataFrame::

        from src.corpus_writer import write_training_corpora

        summary = write_training_corpora(clean_df)
    """
    texts = clean_df[text_col]
    years = pd.to_datetime(
        clean_df[date_col], errors="coerce", format="mixed"
    ).dt.year
    invalid_dates = years.isna() | ~years.between(2005, 2025)
    if invalid_dates.any():
        raise ValueError(
            f"{int(invalid_dates.sum())} articles have missing, invalid, or "
            "out-of-range publication dates; no corpus files were written."
        )

    output_dir = Path(output_dir).resolve()
    staging_dir = output_dir / ".corpus_write_in_progress"
    final_paths = {
        period: output_dir / f"corpus_{period}.txt" for period in ("P1", "P2")
    }
    if os.path.lexists(staging_dir):
        raise FileExistsError(
            f"Incomplete or active corpus write: {staging_dir}. Inspect any "
            "existing corpora and staging files before recovering manually."
        )
    for path in final_paths.values():
        if os.path.lexists(path):
            raise FileExistsError(f"Refusing to overwrite existing corpus: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never remove staging belonging to an earlier writer.
    staging_dir.mkdir()
    staged_paths = {
        period: staging_dir / path.name for period, path in final_paths.items()
    }
    article_counts = {"P1": 0, "P2": 0}
    token_counts = {"P1": 0, "P2": 0}

    try:
        with ExitStack() as stack:
            handles = {
                period: stack.enter_context(
                    path.open("x", encoding="utf-8", newline="\n")
                )
                for period, path in staged_paths.items()
            }
            for year, text in zip(years, texts):
                period = "P1" if year <= 2014 else "P2"
                tokens = preprocess_text(text)
                handles[period].write(" ".join(tokens) + "\n")
                article_counts[period] += 1
                token_counts[period] += len(tokens)
                del tokens

            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())

        summary = pd.DataFrame(
            {
                "article_count": [article_counts["P1"], article_counts["P2"]],
                "token_count": [token_counts["P1"], token_counts["P2"]],
            },
            index=pd.Index(["P1", "P2"], name="period"),
            dtype="int64",
        )
        for period, path in final_paths.items():
            os.link(staged_paths[period], path)
    except BaseException:
        # Check ownership by file identity, including interruption immediately
        # after a link is created. Never remove an unrelated conflicting file.
        for period, path in final_paths.items():
            staged_path = staged_paths[period]
            if staged_path.exists() and path.exists() and path.samefile(staged_path):
                path.unlink()
        # This private directory was created above beneath resolved output_dir.
        # If rollback raises, execution never reaches removal of the marker.
        shutil.rmtree(staging_dir)
        raise

    # Remove the incomplete marker only after BOTH complete files are published.
    shutil.rmtree(staging_dir)
    return summary
