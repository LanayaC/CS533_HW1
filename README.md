# CS533 Homework 1

Part 1 data preparation and Part 2.1-2.8 word-embedding experiments are complete.
Google News calculations were performed on Bilkent Leylak.
The [Part 2 report draft](report/part2_draft.md) currently covers 2.1-2.4;
rewritten sections 2.5-2.8 and the final report are still pending.

- [Part 1 notebook](notes/01_explore_dataset.ipynb): dataset exploration and validation checks.
- [Part 2 notebook](notes/02_word_embeddings.ipynb): saved model/results inspection and four PCA plots.
- [Notebook guide](notes/README.md): external result locations, GN lookup rules, and the Leylak command.
- [Part 1 report draft](report/part1_draft.md): preprocessing rules and independent validation results.
- `src/` contains reusable implementations; `tests/` contains synthetic checks.

## Environment and tests

Use Python 3.10-3.12 with the packages in `requirements.txt`.
Run commands from the repository root using the project environment:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

The verified local environment uses Python 3.12.7 and Gensim 4.4.0.
`src.preprocessing.preprocess_text` is shared by the data-preparation pipeline.
Datasets, corpora, models, indexes, environments, and generated results stay
outside Git. The local data root is `../data/`; input/output paths are configurable.

## Reproduce Part 1 validation

The consolidated review checks reconstructed collections, queries, qrels,
topic definitions, and existing corpora. It does not rewrite the inputs.
Choose a new output directory for each run:

```powershell
.\.venv\Scripts\python.exe -B -m src.part1_review `
  --dataset ../data/articles_mapped.h5 --official-dir ../data/nyt-tc-2026 `
  --corpus-dir ../data/corpora --output-dir ../data/part1_review_new `
  --verify-corpus-text
```

The completed review found zero differences from the official files and
verified all 970,720 corpus lines. Focused validation CLIs, including
`src.document_collections` and `src.known_item_queries`, provide `--help`.

## Reproduce Part 2 experiments

Training uses the existing `corpus_P1.txt` and `corpus_P2.txt` in `../data/corpora/`.
Each command below is a Python module entry point; run `python -m MODULE --help`
for its configurable arguments, using the project environment.

| Section | Module and inputs | Output outside Git |
| --- | --- | --- |
| 2.1 | `src.train_m1 --corpus P1_CORPUS --model NEW_M1` | M1 and `.training.json` |
| 2.2 | `src.train_m2 --source-model M1 --corpus P2_CORPUS --model NEW_M2` | Continued M2 and training report |
| 2.3 | `src.train_m1 --corpus P2_CORPUS --model NEW_M3` | Fresh M3 and training report |
| 2.4, 2.6 | `src.gn_comparison --m1 M1 --m2 M2 --gn GN_FILE --gn-format word2vec --output NEW_JSON` | GN neighbor comparisons, run on Leylak |
| 2.5 | `src.semantic_drift --m1 M1 --m2 M2 --m3 M3 --output-dir NEW_RESULTS` | Drift CSV/JSON and actual Procrustes rotation |
| 2.7 | Run the four PCA cells in the Part 2 notebook | Displays plots; keeps existing PNGs |
| 2.8 | `src.temporal_analogies --m1 M1 --m2 M2 --m3 M3 --alignment DRIFT_JSON --output NEW_JSON` | Emerging vocabulary and two temporal-analogue lists |

Replace uppercase placeholders with paths. Existing local models are in
`../data/models/`; completed measurements and plots are in `../data/part2_results/`.
Reuse the saved M3-to-M1 alignment for temporal analogues. GN is loaded offline
using `src.google_news`; it is not trained or downloaded by the analyzer.
Use fresh output paths because the programs refuse to overwrite existing artifacts.
Multithreaded Word2Vec training is not exactly reproducible despite seed 42.
