# Part 2.1: M1 Word2Vec training

This script implements only M1 training on the existing P1 corpus. It does not
load HDF5, retokenize articles, download embeddings, continue training on P2,
or modify the notebook. `gensim` and scikit-learn are already listed in the
project requirements. The current environment has gensim 4.4.0 with compiled
`corpus_file` support. Optional psutil improves physical-core reporting; Windows
RAM information also has a standard-library native fallback.

## Preview (no training and no output creation)

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m src.train_m1 `
  --corpus "..\data\corpora\corpus_P1.txt" `
  --model "..\data\models\w2v_P1.model" `
  --dry-run
```

Both paths are configurable. Model artifacts must be outside the repository.
The long run is pending user approval; the preview does not instantiate a
training model or reserve/create the output directory.

## Intended full command (run only after approval)

```powershell
.\.venv\Scripts\python.exe -B -m src.train_m1 `
  --corpus "..\data\corpora\corpus_P1.txt" `
  --model "..\data\models\w2v_P1.model" `
  --words government economy baseball music technology
```

The script calls `Word2Vec(corpus_file=...)` with these fixed settings:

| Setting | Value |
| --- | ---: |
| vector_size | 300 |
| window | 5 |
| sg | 1 |
| negative | 5 |
| min_count | 10 |
| sample | 0.001 |
| epochs | 5 |
| seed | 42 |
| workers | `os.cpu_count()` |

The inspected machine has 12 physical cores and 12 logical CPUs, so the full
command uses 12 workers. Available RAM is recorded immediately before and
after training; it changes as other applications use memory. No RAM reservation
or fixed peak-memory claim is made. Gensim's own model-memory estimate is
reported separately from actual machine memory availability.

The P1 corpus was verified in Part 1 to contain 563,736 articles and 47,477,057
tokens. A new read-only frequency scan before training found 66,853 tokens
meeting min_count=10. This is a preflight retention count, not a trained-model
result. That scan recorded government (37,869 occurrences) and economy (10,255
occurrences). The five chosen example words are government, economy, baseball,
music, and technology. Actual model vocabulary/counts are recorded after
training rather than inferred from this scan.

## Output and report

The default example destination resolves to
`C:\Projects\IR\data\models\w2v_P1.model`. The script saves the complete native
gensim model, including training state, and a companion
`w2v_P1.model.training.json` report. The report contains:

- exact settings, gensim version, vocabulary size, corpus article/token counts;
- training time measured around construction, including vocabulary building
  and all five epochs, excluding neighbor calculation and persistence;
- CPU name, physical/logical core counts, RAM total and available snapshots;
- eight cosine neighbors each for government, economy, baseball, music, and
  technology, with query-word frequencies;
- the nondeterminism limitation of multithreaded training despite seed=42.

Neighbor reporting uses `model.wv.most_similar(word, topn=8)` directly and
preserves its returned words, scores, and order without filtering. Stop words,
numbers, and short tokens are retained if returned. Stop words also remain in
the training corpus. The five query words above are the defaults and can be
changed with `--words`, which requires exactly five different alphabetic,
non-stop words of length >= 2 present in the trained vocabulary.

Existing model files, sidecars, reports, and training markers cause refusal;
there is no overwrite flag. Saving uses a single native model file (including
recursive arrays), a staging directory, and exclusive hard-link publication.
Interrupted or failed runs cannot expose a partly written final model.
A forced termination may leave `.w2v_P1.model.training/`; investigate it before
recovering/retrying. A completed run has the model and report with no marker.
No generated model or training data belongs in Git.

## Tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

The new tests use mocks and one tiny 40-line synthetic training corpus in a
temporary directory. They do not train on the real P1 corpus. They check exact
settings, `corpus_file` use, five-word defaults and CLI arguments, dry-run
behavior, overwrite protection, unfiltered top-eight neighbors, native-model
reload, metadata, and failure cleanup.
