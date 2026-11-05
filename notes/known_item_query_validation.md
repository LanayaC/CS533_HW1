# Reproduce Appendix A.4 known-item selection and validation

Run from the repository root in PowerShell, using the existing environment with
pandas, tables, and scikit-learn. Both input paths are configurable:

```powershell
.\.venv\Scripts\python.exe -B -m src.known_item_queries `
  --dataset "..\data\articles_mapped.h5" `
  --official-dir "..\data\nyt-tc-2026"
```

Another environment can use `python -B -m src.known_item_queries` with the same
arguments. `--hdf-key` defaults to `data`; `--examples` controls the maximum number
of printed examples per difference type (default 5, or 0 for none).

## Workflow

1. Read the original HDF5. Reuse `clean_articles()` to remove short text, duplicate
   text, and duplicate URLs, in that exact order, keeping the first duplicates.
2. Reuse `build_collections()` to reconstruct C1/C2 from Appendix A.2 and Table 2.
3. Reuse `check_known_item_eligibility()` on ALL cleaned articles, not separate
   collection subsets. This preserves the global slug-uniqueness condition.
4. Join eligibility to each collection by the original `web_url` and retain
   eligible members only. Raise an error if a collection URL lacks eligibility.
5. For each candidate, compute the following on the complete, unchanged URL:

   ```python
   int(hashlib.md5(web_url.encode("utf-8")).hexdigest()[:12], 16)
   ```

6. Sort each period independently by that integer and select the first 1,000.
   For equal hash values, stable sorting preserves reconstructed collection
   order; Appendix A.4 does not specify a tie-breaker. Insufficient candidates
   raise an error rather than silently producing fewer queries.
7. Build query text from the lowercased slug, split on hyphens, keeping only whole
   `[a-z]+` tokens. Keep stop words and single-letter words in the final query;
   the three-qualifying-token filter applies only to eligibility.
8. Assign `KI1-0001` through `KI1-1000` and `KI2-0001` through `KI2-1000` in hash
   order. Return exactly `qid`, `query`, `docno`, `year` columns.
9. ONLY AFTER selection, read the official `queries_ki_P1.tsv` and
   `queries_ki_P2.tsv` for comparison. Official collections and query files
   never determine the reconstructed selection or query text.

The command prints the cleaning counts, collection sizes, selected query counts,
and missing/extra/mismatched query counts. It loads the full dataset into RAM,
does not depend on notebook variables, and does not execute the notebook.

## Comparison definitions

Match records by `qid`. Missing IDs are in the official file only; extra IDs
are in the selection only. A shared ID with a different `query`, `docno`, or
`year` is mismatched and counts once even if multiple fields disagree. Comparing
docno and query under each qid checks both the selected articles and their order.
Physical TSV row order does not matter because the qid records the intended order.

Exit code 0 means the expected cleaned/collection counts match, each query set
has 1,000 rows, and every official comparison has zero differences. A validation
failure returns 1; input/loading errors also produce a nonzero exit status.

No query datasets or qrels are written. Input DataFrames, the notebook, and the
official files are not modified. All returned datasets remain in memory; any
later export should use a location outside the repository.

## Verified results

The full command completed successfully with exit code **0** in **98.8 seconds**.
It loaded 1,007,004 original articles, retained 970,720 after the exact three
cleaning steps, and reconstructed 173,603 C1 and 210,664 C2 documents.

| Period | Query IDs | Selected | Official | Missing | Extra | Mismatched |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P1 / C1 | KI1-0001 through KI1-1000 | 1,000 | 1,000 | 0 | 0 | 0 |
| P2 / C2 | KI2-0001 through KI2-1000 | 1,000 | 1,000 | 0 | 0 | 0 |

Every selected qid matched the official query text, target docno, and year.
All **44 synthetic tests passed**, including the nine new tests for MD5
selection and query validation. Runtime will vary with machine and disk cache.

## Use existing in-memory results

```python
from src.known_item_queries import (
    select_known_item_queries,
    compare_known_item_queries,
)

queries = select_known_item_queries(collections, eligibility)
ki1 = queries["C1"]
ki2 = queries["C2"]
summary, differences = compare_known_item_queries(queries, official_dir)
```

The default `n_queries=1000` implements the assignment. A smaller positive value
is available for synthetic tests. Eligibility must come from the full cleaned
dataset, and collections must come from the independent reconstruction.

Run synthetic tests without loading the real dataset:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```
