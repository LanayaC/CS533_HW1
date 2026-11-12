
# CS533 Homework 1: Information Retrieval

Implementation of CS533 Homework 1, covering test collection construction, temporal word embeddings, retrieval models, and evaluation.

## Project Structure

| Directory | Contents |
| --- | --- |
| `src/` | Python implementations for Parts 1–4 |
| `tests/` | Unit tests and validation checks |
| `notes/` | Four Jupyter notebooks documenting experiments |
| `runs/` | 60 TREC-format run files for Part 4.2 |
| `report/` | Final PDF report |

Datasets, trained models, indexes, and intermediate results are kept outside the repository, as required by the assignment.

## Setup and Testing

**Tested environment:** Python 3.12.7, Windows 11.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the complete test suite:

```powershell
python -m unittest discover -s tests -q
```

**Last verified:** 272 tests passed in 27.787 seconds.

## Data and Model Locations

The experiments use the following external directory structure:

```text
../data/
├── articles_mapped.h5
├── nyt-tc-2026/
├── corpora/
├── models/
├── part3_indexes/
├── part3_results/
└── part4_results/
```

- `articles_mapped.h5`: article dataset.
- `nyt-tc-2026/`: official query and relevance files.
- `corpora/`: cleaned P1 and P2 training corpora.
- `models/`: trained Word2Vec models.
- `part3_indexes/`: retrieval indexes and document embeddings.
- `part3_results/`: saved retrieval rankings.
- `part4_results/`: evaluation tables and analyses.

The original dataset and official files must be obtained separately. Input and output paths are configurable through the command-line interfaces.

The pretrained Google News model is also excluded. Google News analysis was performed on Bilkent Leylak; local retrieval used a subset of embeddings matching the collection vocabulary.

## Part 1: Test Collection Construction

Part 1 implements article cleaning, document collection construction, known-item queries, topic queries, relevance judgments, and corpus preparation.

The main implementation files are in `src/`, including `preprocessing.py`, `document_collections.py`, `known_item_queries.py`, `topic_queries.py`, and their corresponding relevance-judgment modules.

Validate the reconstructed collections against the official files:

```powershell
python -m src.part1_review `
  --dataset ../data/articles_mapped.h5 `
  --official-dir ../data/nyt-tc-2026 `
  --corpus-dir ../data/corpora `
  --output-dir ../data/part1_review_new `
  --verify-corpus-text
```

The completed validation found no differences from the official collection and query specifications.

Individual construction scripts expose their arguments through `--help`, for example:

```powershell
python -m src.document_collections --help
python -m src.known_item_queries --help
python -m src.corpus_writer --help
```

See `notes/01_explore_dataset.ipynb` for dataset exploration and validation results.

## Part 2: Temporal Word Embeddings

Three Word2Vec models were trained:

| Model | Description |
| --- | --- |
| M1 | Trained on P1 |
| M2 | Continued from M1 using P2 |
| M2s | Trained from scratch on P2 |

The scratch-trained model is also called M3 in some Part 2 scripts.

Train M1:

```powershell
python -m src.train_m1 `
  --corpus ../data/corpora/corpus_P1.txt `
  --model ../data/models/new_M1.model
```

Continue training M2:

```powershell
python -m src.train_m2 `
  --source-model ../data/models/new_M1.model `
  --corpus ../data/corpora/corpus_P2.txt `
  --model ../data/models/new_M2.model
```

To train M2s, use `src.train_m1` with the P2 corpus and a different output path.

Additional analysis modules:

- `src.gn_comparison`: Google News comparisons.
- `src.semantic_drift`: Procrustes alignment and semantic drift.
- `src.temporal_analogies`: emerging vocabulary and analogies.

See `notes/02_word_embeddings.ipynb` for the saved analyses and visualizations.

## Part 3: Retrieval Models

Retrieval experiments are implemented in `src.retrieval`.

The implemented methods are TF-IDF, BM25, dense retrieval (M1, M2, M2s, GN), embedding-based query expansion, linear hybrid retrieval, and reciprocal rank fusion (RRF).

Example sparse retrieval run:

```powershell
python -m src.retrieval `
  --dataset ../data/articles_mapped.h5 `
  --official-dir ../data/nyt-tc-2026 `
  --index-dir ../data/part3_indexes_new `
  --output-dir ../data/part3_results_new `
  --combined-output
```

Other retrieval configurations use:

- `--dense-model` and `--embedding-model` for dense retrieval.
- `--expansion-model` for query expansion.
- `--hybrid-model` and `--input-results-dir` for hybrid retrieval.

Use `python -m src.retrieval --help` for the full set of arguments and `--smoke` for small-scale checks.

Use separate output directories when reproducing experiments to avoid overwriting existing results.

See `notes/03_retrieval_models.ipynb` for retrieval experiments and measurements.

## Part 4: Evaluation

Evaluation is implemented in `src.evaluation` using the official relevance judgments and saved retrieval CSVs.

| Experiment | Purpose |
| --- | --- |
| `bm25` | Metric validation |
| `past_future` | P1 parameter tuning and transfer to P2 |
| `main_results` | Part 4.2 evaluation tables |
| `temporal` | Period and publication-year comparisons |
| `per_query` | Query-level error and drift analyses |

Example:

```powershell
python -m src.evaluation `
  --official-dir ../data/nyt-tc-2026 `
  --results-dir ../data/part3_results `
  --output-dir ../data/part4_results_new `
  --experiment bm25
```

Change `--experiment` to run other analyses. Run `past_future` before `main_results`, using the same evaluation output directory, so the selected P1 parameters are available.

Evaluation uses the assignment's metric definitions and binary relevance judgments. P1-selected parameters are applied to P2 without retuning.

See `notes/04_evaluation.ipynb` for evaluation results and statistical analysis.

## TREC Run Files

The `runs/` directory contains **60 run files**, corresponding to the 15 methods evaluated on each of four test sets:

| Test set | Files |
| --- | ---: |
| KI1 / C1 | 15 |
| KI2 / C2 | 15 |
| Topics / C1 | 15 |
| Topics / C2 | 15 |
| **Total** | **60** |

Each file contains the top 100 documents per query in standard TREC format:

```text
qid Q0 docno rank score tag
```

The files were generated from the exact retrieval results referenced in the Part 4.2 evaluation tables.

To regenerate them from the external results:

```powershell
python -m src.export_trec_runs
```

The exporter validates query coverage, document IDs, ranking positions, scores, and duplicate documents. It refuses to overwrite existing files.

The generated TREC files are already included in `runs/`.

## Runtime and Reproducibility

**Local hardware:** Snapdragon X 12-core X1E80100 (3.40 GHz), 15.6 GiB RAM, Windows 11.

Measured runtimes from saved experiment logs:

| Operation | Runtime |
| --- | ---: |
| M1 training on P1 | 609.03 s |
| M2 continued training on P2 | 340.59 s |
| M2s training on P2 | 356.78 s |
| TF-IDF retrieval, C1 / C2 | 8.84 / 13.18 ms/query |
| BM25 retrieval, C1 / C2 | 8.14 / 11.68 ms/query |
| Dense M1, C1 indexing | 69.12 s |
| Dense M1, C1 retrieval | 4.61 ms/query |
| QE M1, C1 / C2 | 7.56 / 10.83 ms/query |
| Hybrid M1, C1 | 2.46 ms/query |

**Timing notes:**

- Word2Vec times cover vocabulary processing and training, excluding model saving and, where applicable, loading.
- Sparse and dense retrieval times exclude CSV writing and indexing.
- QE times exclude one-time neighbor preparation.
- Hybrid times measure fusion-stage work, not end-to-end retrieval.
- These are measured phases, not total pipeline execution times.

Word2Vec uses seed 42 with 12 workers. Multithreaded training may produce slightly different results across executions.

## Report

The final PDF report is submitted in `report/` and contains the findings, comparisons, limitations, and required AI-use disclosure for all four parts.
