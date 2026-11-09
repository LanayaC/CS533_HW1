# Notebook guide

- [01_explore_dataset.ipynb](01_explore_dataset.ipynb): Part 1 exploration, cleaning, preprocessing, corpus preparation, and query inspection. Uses the shared `src.preprocessing.preprocess_text` function.
- [02_word_embeddings.ipynb](02_word_embeddings.ipynb): Part 2 inspection of saved models/results, Google News comparisons, four PCA drift plots, and temporal analogues.

Saved artifacts are outside the repository under `C:\Projects\IR\data`:

- Part 1 validation: [part1_results.json](../../data/part1_review/part1_results.json).
- Models and training measurements: [models/](../../data/models/) (`w2v_P1.model`, `w2v_P2_continued.model`, `w2v_P2_fresh.model`, and their `.training.json` reports).
- Drift measures and saved alignment: [semantic_drift.csv](../../data/part2_results/semantic_drift.csv) and [semantic_drift.json](../../data/part2_results/semantic_drift.json).
- Leylak Google News results: [gn_comparison.json](../../data/part2_results/gn_comparison.json).
- Emerging words and temporal analogues: [temporal_analogies.json](../../data/part2_results/temporal_analogies.json) and [results summary](../../data/part2_results/temporal_analogies_report.md).
- Four PCA figures: [plots/](../../data/part2_results/plots/).
- Report drafts: [Part 1](../report/part1_draft.md) and [Part 2](../report/part2_draft.md).

Run CLI commands from the repository root with the project environment; `python -m <module> --help` lists configurable paths. The [root README](../README.md) documents consolidated Part 1 validation.
Part 2 entry points are `src.train_m1` (M1 and fresh M3), `src.train_m2` (continued M2), `src.semantic_drift`, and `src.temporal_analogies`. Fresh M3 uses the P2 corpus and a separate destination.
The drift JSON preserves definitions, rankings, and the actual M3-to-M1 rotation; reuse it with `src.temporal_analogies --alignment`, applying `M3[w] @ R` without refitting.

GN uses `src.google_news.lookup_gn_key`: lowercase → capitalized → uppercase → component-capitalized underscore phrase; first match wins, otherwise `None`. This does not disambiguate senses.
`src.google_news.load_gn_vectors` loads an existing `.bin`/`.bin.gz` offline as binary KeyedVectors. On Leylak, use a Linux environment with the project dependencies (including Gensim 4.4.0); do not copy the Windows virtual environment.
The earlier Slurm planning allowance was one node/task, four CPUs, 32G memory, and 30 minutes, subject to cluster rules; it is not a measured requirement. Limit numerical-library threads to allocated CPUs and create any Slurm log directory before submission.

Example GN reproduction inside a Slurm allocation, using a fresh output name:

```bash
cd /home/aziza.carbonell/CS533_HW1
CS533_MODELS=/home/aziza.carbonell/cs533_data/models
CS533_RESULTS=/home/aziza.carbonell/cs533_data/part2_results
/home/aziza.carbonell/CS533_HW1/.venv/bin/python -B -m src.gn_comparison \
  --m1 "$CS533_MODELS/w2v_P1.model" \
  --m2 "$CS533_MODELS/w2v_P2_continued.model" \
  --gn "$CS533_MODELS/GoogleNews-vectors-negative300.bin.gz" \
  --gn-format word2vec \
  --output "$CS533_RESULTS/gn_comparison_new.json"
```

Training and analysis commands refuse existing outputs. Use fresh destinations for reproduction; investigate interrupted-run markers before recovery. Consult saved results rather than rerunning notebook cells merely to inspect completed work.
