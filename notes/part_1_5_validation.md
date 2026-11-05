# Reproduce Part 1.5 collection validation

Run these commands in PowerShell from the repository root, with the existing
Python environment containing pandas and tables. Both input paths are configurable.

```powershell
.\.venv\Scripts\python.exe -B -m src.document_collections `
  --dataset "..\data\articles_mapped.h5" `
  --official-dir "..\data\nyt-tc-2026"
```

With another Python environment or data location:

```powershell
python -B -m src.document_collections `
  --dataset "D:\news-data\articles_mapped.h5" `
  --official-dir "D:\news-data\nyt-tc-2026"
```

Use `--hdf-key` if the HDF5 key differs from its default, `data`. Use `--examples`
to change the number of example differences printed (default 5, or 0 for none).
The command is independent of Jupyter and does not execute or modify the notebook.
It loads the original HDF5 into memory; allow enough RAM for the original article
table and temporary cleaning/reconstruction data.

## Exact processing rules

1. Load the original articles using `pandas.read_hdf`.
2. Remove articles with fewer than 10 whitespace-separated text tokens. Count
   using `text.fillna("").str.count(r"\S+")`; do not use `preprocess_text()` here.
3. Remove duplicate text, keeping the first surviving occurrence.
4. Remove duplicate `web_url`, keeping the first surviving occurrence.
5. Reconstruct C1/C2 from these cleaned articles using Appendix A.2's exact URL
   regex and Table 2's 33 exact paths (32 topics). The two books paths map to books.
6. Derive the year from `pub_date`, not the URL. C1 is 2005-2014, C2 is 2015-2025.
7. Sort each collection by full publication timestamp and then URL. Assign
   `C1-000000`, `C1-000001`, etc., or the equivalent C2 prefix. Return the columns
   `docno`, `web_url`, `year`, `topic`.
8. Only after reconstruction, load `collection_P1.tsv` and `collection_P2.tsv`
   from the official directory for validation. The official files and topics.tsv
   are never used to choose or label reconstructed articles.

## Understanding the report

The command prints the original count, removals at each cleaning step, final
cleaned count, each reconstructed collection size, and validation differences.

| Check | Expected |
| --- | ---: |
| Final cleaned articles | 970,720 |
| C1 documents | 173,603 |
| C2 documents | 210,664 |
| Missing, extra, mismatched records in each collection | 0 |

Records are matched by URL. Missing records are official URLs absent from the
reconstruction; extra records are reconstructed URLs absent from the official
collection. Mismatched records share a URL but disagree on docno, year, or topic.
A record with several mismatched fields counts once. The report also prints a
few detailed differences when present. Duplicate URLs/docnos or missing required
values raise an error rather than yielding ambiguous comparison counts.

Exit code 0 means all expected counts match and both collections have zero
differences. Exit code 1 indicates a validation failure; inspect the printed
counts and examples. Loading/parsing errors also cause a nonzero exit status.

No reconstructed datasets, caches, or reports are written by this command.
Official files are read only, and the original DataFrame is not modified.

## Verified results

The full command was run against the existing original HDF5 and official TSVs.
It loaded **1,007,004** original articles and completed with exit code **0**.

| Cleaning step | Removed | Remaining |
| --- | ---: | ---: |
| Fewer than 10 whitespace-separated tokens | 28,799 | 978,205 |
| Duplicate text (keep first) | 6,914 | 971,291 |
| Duplicate URL (keep first) | 571 | 970,720 |

| Collection | Reconstructed | Official | Missing | Extra | Mismatched |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 173,603 | 173,603 | 0 | 0 | 0 |
| C2 | 210,664 | 210,664 | 0 | 0 | 0 |

Measured elapsed time was **101.1 seconds** with the existing Python 3.12.7
environment (pandas 3.1.0, tables 3.11.1). Runtime will vary by machine and disk
cache state. All expected sizes and all shared records matched; no differences
were hidden by relying on collection sizes alone.

## Use the functions with an existing cleaned DataFrame

```python
from src.document_collections import build_collections, compare_collections

collections = build_collections(clean_df)
c1 = collections["C1"]
c2 = collections["C2"]
summary, differences = compare_collections(collections, official_dir)
```

`clean_articles(original_df)` is also available and returns the cleaned DataFrame
and a per-step removal report. No function writes reconstructed datasets. If you
later save returned DataFrames, choose a directory outside the repository.

Run the synthetic tests without reading the real dataset:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```
