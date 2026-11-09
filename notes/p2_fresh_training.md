# Part 2.3: train a fresh P2 model

The professor requires the same Word2Vec settings as M1 and reporting of
vocabulary size and training time. The existing `src/train_m1.py` already
constructs `Word2Vec(corpus_file=..., **assignment_settings())` from scratch
with configurable corpus and model paths. Reuse it unchanged for P2; no new
training implementation is needed and no existing model is loaded.

Settings remain vector_size=300, window=5, sg=1, negative=5, min_count=10,
sample=0.001, epochs=5, seed=42, workers=`os.cpu_count()` (12 on this machine).
The fresh vocabulary and vectors come only from P2. M1 and the continued P2
model remain separate and are never loaded or overwritten by this command.
Existing model, report, sidecar, and staging-marker protection is inherited
from the existing trainer, as is the outside-repository output requirement.

## Preview

Run from `C:\Projects\IR\CS533_HW1` in PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m src.train_m1 `
  --corpus "C:\Projects\IR\data\corpora\corpus_P2.txt" `
  --model "C:\Projects\IR\data\models\w2v_P2_fresh.model" `
  --dry-run
```

The preview validates paths and prints settings without training or creating
output files or directories.

## Full command (only after approval)

```powershell
.\.venv\Scripts\python.exe -B -m src.train_m1 `
  --corpus "C:\Projects\IR\data\corpora\corpus_P2.txt" `
  --model "C:\Projects\IR\data\models\w2v_P2_fresh.model"
```

The native model will be saved as `w2v_P2_fresh.model`, accompanied by
`w2v_P2_fresh.model.training.json`. The required measurements are its
`vocabulary_size` and `training_seconds`. Timing includes vocabulary building
and all five epochs, excluding neighbor calculation and saving. The existing
report also retains CPU/RAM metadata and eight unfiltered neighbors for the
five configured sanity-check words. Multithreaded training is nondeterministic
despite seed=42. Full P2 fresh training has not started; its measurements are
pending that run.

## Tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

Reuse `tests/test_train_m1.py` for exact settings, corpus_file construction,
saving/reloading, report measurements, overwrite protection, dry runs, and
failure cleanup. Two added P2 tests verify the proposed CLI command on tiny
synthetic input: fresh training never loads an existing model, retains only
P2 vocabulary, preserves existing-model/report sentinels and the corpus,
reports the correct vocabulary/counts and timing definition, and refuses
existing M1, continued-model, and fresh-model destinations. The P2 dry run
checks the configurable paths without training or creating output.

Validation: all 92 tests passed, and the real-input P2 dry run passed with the
requested fresh-model output path and exact assignment settings (12 workers).
Only tiny synthetic corpora were trained during these checks.
