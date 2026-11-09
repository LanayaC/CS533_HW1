# Part 2.4: pretrained Google News vectors and one lookup rule

Use the existing pretrained Google News Word2Vec vectors (GN). Do not train or
continue training GN. `src/google_news.py` loads an existing local file only;
it contains no downloader, automatic fallback download, or training calls.

## Shared lookup API for all later GN comparisons

```python
from src.google_news import lookup_gn_key

key = lookup_gn_key(gn, "new_york")
if key is None:
    # Record an explicit missing result; do not fabricate a zero vector.
    pass
else:
    vector = gn[key]
```

The rule is fixed, with the first existing key winning:

1. The original lowercase token: `new_york`.
2. `token.capitalize()`: `New_york`.
3. `token.upper()`: `NEW_YORK`.
4. For underscore-separated phrases, capitalize each component: `New_York`.

Duplicate candidates are checked once. The fourth rule only applies to
underscore phrases. The function preserves underscores and does not join
separate corpus tokens into phrases, remove punctuation, average case variants,
or change GN's vocabulary. Inputs must be nonempty lowercase strings without
whitespace; invalid input raises an error, while an absent word returns `None`.
For example, `us` wins over `Us` and `US` if all exist. This deterministic
choice does not resolve the semantic ambiguity introduced by lowercasing.

Later GN similarity and neighbor comparisons must import `lookup_gn_key` and
resolve queries through it before indexing GN or calling `most_similar`.
Retain the original lowercase label and record its matched GN key. Keep GN
neighbor spellings as returned. No Part 2.5/2.6 measures are implemented here.

## Local development and synthetic checks

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

`tests/test_google_news.py` uses dictionaries and tiny three-dimensional
KeyedVectors. It checks lookup order, missing results, phrase casing, input
validation, unmodified vectors, binary/gzip loading, read-only memory mapping,
and CLI output. It never downloads GN or trains embeddings.

Validation: all 104 repository tests passed, including 12 new GN tests. The
CLI help check passed. No real Google News file was loaded during validation.

## Proposed Leylak setup (approval required before execution)

Continue developing and testing in local VS Code. Copy the source to a Linux
working directory on Leylak only when approved. Keep model files in approved
shared/project storage outside the repository, accessible to the compute node.
Do not copy the Windows `.venv`; create a Linux Python 3.10-3.12 environment.

Needed for this Part 2.4 job:

- `src/google_news.py` in the project's `src/` directory.
- A Linux environment with `gensim==4.4.0` and compatible NumPy/SciPy; Gensim's
  package dependencies also include smart_open. The local reference is Python
  3.12.7, NumPy 2.5.4, SciPy 1.18.1, and Gensim 4.4.0. Resolve compatible
  Linux packages and record their versions rather than copying Windows wheels.
- The full pretrained `GoogleNews-vectors-negative300.bin` or `.bin.gz` file.
  Reuse a verified shared copy if available; otherwise obtain it only after
  approval. Record provenance and checksum. Gensim's
  [model catalog](https://raw.githubusercontent.com/RaRe-Technologies/gensim-data/master/list.json)
  identifies GN as 3,000,000 entries of dimension 300 with an approximately
  1.74 GB compressed distribution.
- A Slurm batch script, an approved output/log directory, and confirmed Leylak
  partition/account, environment setup, storage paths, and resource limits.

No corpora, M1, continued model, or fresh model are needed just to load GN and
test lookups. Later comparisons will need the relevant saved models too.

### Loading the pretrained file

The loader uses the existing local binary without a vocabulary limit:

```python
from src.google_news import load_gn_vectors
gn = load_gn_vectors("/approved/project/models/GoogleNews-vectors-negative300.bin.gz")
```

Internally this calls `KeyedVectors.load_word2vec_format(path, binary=True)`.
It preserves the original case-sensitive keys and full float32 vectors.
Loading a gzip file does not require writing a decompressed file first.
GN is a KeyedVectors object, not a trainable Word2Vec model. See
[Gensim 4.4.0's KeyedVectors API source](https://github.com/piskvorky/gensim/blob/4.4.0/gensim/models/keyedvectors.py).

The vector matrix alone is 3,000,000 x 300 x 4 = 3.6 GB (about 3.35 GiB).
Vocabulary strings/dictionaries and later similarity operations need extra
memory. Propose an initial CPU job with one task, four CPUs, and 32 GiB RAM,
subject to Leylak's actual allocation rules; this is a planning allowance,
not a measured requirement. Use job accounting to measure peak RSS and adjust.
No GPU is needed for this loader. Allow roughly 15 GB of free project storage
if keeping compressed input and an optional native cache together.

### Optional native cache for repeated jobs

After an approved one-time load, an uncompressed native cache with a separate
vector array allows later read-only memory mapping. For example, inside an
approved compute job, create a new external cache directory exclusively:

```python
from pathlib import Path

cache_dir = Path("/approved/project/models/gn_cache")
cache_dir.mkdir(exist_ok=False)
gn.save(str(cache_dir / "google_news.kv"), separately=["vectors"])
```

Keep `google_news.kv` and every generated sidecar together. Later:

```python
gn = load_gn_vectors("/approved/project/models/gn_cache/google_news.kv",
                     model_format="keyedvectors")
```

This calls `KeyedVectors.load(path, mmap="r")`; it does not directly memory-map
the original word2vec `.bin` or gzip archive. Vocabulary metadata still uses
RAM. No cache has been created, and this conversion is proposed only.

### Slurm template (not submitted)

The following is a template to save as `gn_lookup.slurm` on Leylak after
approval. Replace project paths and confirm partition/account before use.
The cluster-specific values have not been verified; no connection was made.

```bash
#!/bin/bash
#SBATCH --job-name=cs533-gn-lookup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=gn-lookup-%j.out

set -euo pipefail
cd /approved/project/CS533_HW1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

/approved/project/env/bin/python -B -m src.google_news \
  --model /approved/project/models/GoogleNews-vectors-negative300.bin.gz \
  --model-format word2vec \
  --words government economy baseball music technology new_york
```

Submit only after approval, supplying the confirmed partition and account if
required. Slurm supports these resource directives and writes job output to
the chosen log; it does not transfer project/model files for you. See the
[official sbatch documentation](https://slurm.schedmd.com/sbatch.html).
The JSON output lists model dimensions and each token's actual matched key,
ordered candidate spellings, and `found`/`missing` status (`null` for a missing key).
Check 3,000,000 entries and 300 dimensions for the full GN distribution.

No large model download, cluster connection, Slurm submission, or native-cache
conversion has been performed. Each awaits explicit approval.
