# CS533 Homework 1

Information Retrieval: Word Embeddings, Semantic Drift, and News Retrieval.
Part 1 data preparation and independent test-collection reconstruction are
implemented. Part 2 training has not started. The student report draft is
[report/part1_draft.md](report/part1_draft.md).

The final completion pass passed **74 synthetic tests**. All independent
collection, known-item query, known-item qrels, topic qrels, and topic-definition
comparisons had **zero differences** from official files. Both existing corpora
matched the current tokenizer for all 970,720 cleaned articles. The consolidated
review took **217.9 seconds**, including full corpus-text verification; runtime
varies by machine and disk cache. See the report for source labels and statistics.

## Environment and files

Run commands from this repository root in PowerShell. Use Python 3.10–3.12;
the existing environment uses Python 3.12.7. Part 1 uses pandas, tables, and
scikit-learn from `requirements.txt`. Full-dataset review needs enough RAM to
load the original article table and temporary metadata. Part 2 dependencies
have not yet been exercised by these Part 1 checks.

The verified Part 1 versions were pandas 3.1.0, tables 3.11.1, and scikit-learn
1.9.1. The dependency file is an initial package list rather than a frozen lock
file; these version numbers describe the tested environment.

- `src/`: preprocessing, counting, corpus writing, reconstruction, and validation.
- `tests/`: synthetic tests; they do not load the real article dataset.
- `notes/`: validation notes and the preserved exploratory notebook.
- `report/`: the Part 1 draft and later final report sources/PDF.
- `runs/`: later required TREC retrieval runs; no retrieval runs exist yet.

Raw datasets, generated corpora, model files, environments, and bulk summaries
are stored outside Git. The example layout uses `../data/articles_mapped.h5`,
`../data/nyt-tc-2026/`, and `../data/corpora/`. All review input paths are CLI
arguments; no Jupyter variables or user-specific absolute paths are required.
The `.gitignore` excludes known data/model/cache paths and formats, but cannot
filter files by size; review new files before staging.

`src/preprocessing.py` is the canonical tokenizer. Earlier exploratory notebook
cells contain an older inline experiment; later cells import the reusable
function. The notebook is preserved and is not executed by these commands.

## Reproduce the Part 1 checks

Run the complete synthetic suite:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

Validate the shared 32 topic-query definitions without loading HDF5:

```powershell
.\.venv\Scripts\python.exe -B -m src.topic_queries `
  --official-dir "..\data\nyt-tc-2026"
```

For a complete review, load HDF5 **once**, reuse all existing independent
builders, validate collections/known-item queries/both qrels/topic definitions,
produce report statistics, and inspect the existing corpora. The optional
`--verify-corpus-text` flag checks every persisted article line against the
current tokenizer in source article order. It does not rewrite the corpora.

```powershell
$ReviewOutput = Join-Path "..\data" ("part1_review_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
.\.venv\Scripts\python.exe -B -m src.part1_review `
  --dataset "..\data\articles_mapped.h5" `
  --official-dir "..\data\nyt-tc-2026" `
  --corpus-dir "..\data\corpora" `
  --output-dir $ReviewOutput `
  --verify-corpus-text
```

This creates `part1_results.json` in the fresh external output directory.
Existing summaries are never overwritten. Output inside the repository or
official-file directory is rejected. `--hdf-key` defaults to `data`.
Exit code 0 means the reconstruction, expected counts, and corpus checks passed;
the JSON distinguishes original-data measurements, persisted corpus statistics,
independent reconstruction, and official-file reporting statistics.

Official data are read only **after independent reconstruction** for comparison.
Official topic distributions/query examples may then be used for reporting;
they never determine reconstructed documents, queries, or qrels. Future retrieval
experiments should use the official collection/query/qrels files, as required
by the assignment. The shared topic-query definitions apply to both periods.

The dedicated `src.document_collections`, `src.known_item_queries`,
`src.known_item_qrels`, and `src.topic_qrels` CLIs remain available for focused
diagnosis (`--help`). Prefer the consolidated review over running each command
separately, because the individual commands each reload HDF5.

## Training corpora

The two existing files are `corpus_P1.txt` (2005–2014) and `corpus_P2.txt`
(2015–2025), outside the repository. Each contains all cleaned articles for its
period, one space-separated tokenized article per UTF-8 line; no sampling or
stop-word removal is applied. Never copy them into the repository.

If corpora are absent, create them in a **new external destination**, using
configurable command arguments (do not run this merely to verify existing files):

```powershell
@'
import sys
import pandas as pd
from src.document_collections import clean_articles
from src.corpus_writer import write_training_corpora

original = pd.read_hdf(sys.argv[1], key="data")
cleaned, cleaning_report = clean_articles(original)
print(cleaning_report)
print(write_training_corpora(cleaned, output_dir=sys.argv[2]))
'@ | .\.venv\Scripts\python.exe -B - "..\data\articles_mapped.h5" "..\data\corpora_new"
```

The writer refuses existing corpus files. `.corpus_write_in_progress/` marks
an incomplete or active write; investigate it before using/recovering files.
A valid corpus pair has both final files and no marker. Interrupted writes are
handled as documented in `src/corpus_writer.py`.

## Report and remaining submission preparation

The draft records which supplied measurements were reproduced and defines
query-length and overlap conventions. Review it for correctness, add the
student name/email and required identification, refine the AI-use disclosure,
and format the eventual final PDF with numbered pages/tables/figures. A Parquet
cache is an optional reload-speed improvement suggested by the PDF; no cache
or additional Parquet engine is required by the current HDF5-based commands.
