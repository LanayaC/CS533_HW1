# Part 2.2: continue M1 on P2

`src/train_m2.py` loads the saved native M1 into memory and follows the
professor's screenshot:

```python
model = Word2Vec.load(source_model_path)
model.build_vocab(corpus_file=corpus_P2_path, update=True)
model.train(corpus_file=corpus_P2_path, total_examples=model.corpus_count,
            total_words=model.corpus_total_words, epochs=model.epochs)
```

The screenshot includes `total_words=model.corpus_total_words`, which the
installed Gensim 4.4.0 requires alongside `corpus_file`. Both corpus counts
come from the P2 vocabulary update. All Word2Vec settings are inherited from
M1 without overrides: vector_size=300, window=5, sg=1, negative=5, min_count=10,
sample=0.001, epochs=5, seed=42, workers=12 for the existing saved M1.

M1 is loaded only and never saved over. The continued model and JSON report
must use a new destination outside the repository. Existing models, sidecars,
reports, or training markers cause refusal. Publication uses the existing M1
trainer's staging and exclusive hard-link approach. Caught failures roll back
this attempt's output; forced termination may leave a marker that must be
investigated. There is no automatic retry.

## Dry run

From `C:\Projects\IR\CS533_HW1` in PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m src.train_m2 `
  --source-model "C:\Projects\IR\data\models\w2v_P1.model" `
  --corpus "C:\Projects\IR\data\corpora\corpus_P2.txt" `
  --model "C:\Projects\IR\data\models\w2v_P2_continued.model" `
  --dry-run
```

The dry run checks paths, loads M1, and prints its vocabulary size and settings.
It does not update vocabulary, train, create output directories, or save files.

## Full command (requires approval)

```powershell
.\.venv\Scripts\python.exe -B -m src.train_m2 `
  --source-model "C:\Projects\IR\data\models\w2v_P1.model" `
  --corpus "C:\Projects\IR\data\corpora\corpus_P2.txt" `
  --model "C:\Projects\IR\data\models\w2v_P2_continued.model"
```

All three paths are configurable. The native model will be accompanied by
`w2v_P2_continued.model.training.json`, which reports original and updated
vocabulary sizes, their difference (new words added), and training time.
Training time includes the P2 vocabulary update plus all inherited epochs;
M1 loading and output saving are excluded. Corpus counts, inherited settings,
Gensim version, and CPU/RAM snapshots provide context. Multithreaded training
is nondeterministic despite the inherited seed. Full P2 training has not run.

## Synthetic tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

`tests/test_train_m2.py` verifies the exact call order and P2 counts, vocabulary
growth and min_count behavior, inherited settings, model reload, source and
corpus preservation, dry-run behavior, overwrite protection, failure cleanup,
and no retries. Real continuation in the tests uses only tiny temporary corpora.

Validation after implementation: all 90 synthetic tests passed. The real-input
dry run passed and confirmed M1 has 66,853 words and the settings listed above.
Updated vocabulary size, new words added, and full training time remain pending
the approved full P2 run.
