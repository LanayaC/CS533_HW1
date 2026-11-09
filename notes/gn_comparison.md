# Part 2.6: the pretrained model's view

`src/gn_comparison.py` compares M1, continued M2, and pretrained GN. It reuses
the same 44 targets and group labels as Part 2.5, and the case-sensitive lookup
and offline loader in `src/google_news.py`. No fresh M3, training, downloading,
cross-model cosine, or alignment is used.

## Required comparisons

For every target, obtain GN's ten nearest neighbors among resolvable words in
M1's vocabulary. Compare that set with M1's ten nearest neighbors and continued
M2's ten nearest neighbors. Each overlap is the intersection size divided by
ten. Local neighborhoods search each local model's own full vocabulary without
filtering. Only GN's candidate vocabulary is restricted to M1.

The implementation resolves every original M1 label through `lookup_gn_key`:
lowercase, capitalized, uppercase, and component-wise capitalization for
underscore-separated phrases. It copies and normalizes only these GN candidate
vectors, and calculates exact cosine scores over that restricted matrix.
It never retrieves global GN neighbors and filters them afterward.

For each query, use the same lookup rule. Exclude the query's original label
and every candidate mapped to its GN key, so a casing alias cannot return the
query itself. Returned GN neighbors retain the original M1 label in `word`,
their actual GN spelling in `gn_key`, and the GN cosine score. Set overlap uses
`word` exclusively. Different M1 labels remain distinct candidate words even
when mapped to the same GN key; query-key aliases are all excluded. Equal
cosines are ordered alphabetically by the original M1 labels.

All 44 rows are retained. Missing queries are explicit per model and unavailable
scores are null. If fewer than ten valid candidates remain, flag
`insufficient_candidates` and leave the relevant overlap undefined; do not
change the denominator or add substitute words. Zero/nonfinite GN candidate
vectors are listed as undefined in candidate-pool metadata and excluded from
cosine search. A zero/nonfinite query is also reported as undefined.

The compact JSON includes:

- Ten neighbors in each available model, with original labels and scores;
  GN neighbors also record their resolved GN keys.
- `nn_gn_m1` and `nn_gn_m2`, intersection words, availability statuses,
  membership, and matched query keys.
- `closer_to`: M1, M2, tie, or unavailable, directly comparing the two overlaps.
- Candidate-pool coverage counts, model paths, vocabulary sizes, and versions.

No arbitrary disagreement threshold or additional correlation statistics are
introduced. The report should use the two overlaps and neighbor lists to
explain which period GN resembles and where it disagrees with both. Case
fallback can merge senses (for example `us`/`US`); consider that and GN's
verified training period. Do not infer a training date from a file timestamp.

## Synthetic validation

From the local repository root:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_g*.py"
```

All 29 GN-related tests passed: 17 Part 2.6 tests plus 12 existing Part 2.4
tests. The tests use tiny synthetic vectors/files only. They cover the lookup
order, restricted cosine search against a brute-force reference, exclusion of
GN-only and M2-only candidates, query/alias exclusion, phrase labels, exact
overlap, full local neighborhoods, all 44 rows, explicit missing/invalid cases,
gzip loading, unchanged source files, compact JSON, dry runs, no retries, and
protection of existing/concurrent results. CLI help also passed. The real GN
model has not been loaded and no full analysis or cluster operation has run.

## Exact Leylak analysis command (not executed)

The user supplied these paths. Make the new module and its existing imports
available under the Leylak project's `src/` directory before execution.
The Linux environment needs the existing project dependencies, including
Gensim, NumPy, SciPy, and scikit-learn (through reused modules).

Run the following **inside an approved Slurm allocation**, after full-analysis
approval:

```bash
cd /home/aziza.carbonell/CS533_HW1
/home/aziza.carbonell/CS533_HW1/.venv/bin/python -B -m src.gn_comparison \
  --m1 /home/aziza.carbonell/cs533_data/models/w2v_P1.model \
  --m2 /home/aziza.carbonell/cs533_data/models/w2v_P2_continued.model \
  --gn /home/aziza.carbonell/cs533_data/models/GoogleNews-vectors-negative300.bin.gz \
  --gn-format word2vec \
  --output /home/aziza.carbonell/cs533_data/part2_results/gn_comparison.json
```

Every model/output path is configurable. The analysis creates the results
directory if absent. Existing output files and private in-progress markers
cause refusal; it never overwrites model or result files. Errors propagate
without retry. Output is published exclusively from a private staging directory.

Appending `--dry-run` validates file paths and prints the plan without loading
models, performing numerical analysis, or creating files/directories. Even this
Leylak command has not been executed locally or remotely.

## Recommended Slurm request

Propose one node, one task, four CPUs, 32 GiB RAM, and 30 minutes. This is a
conservative initial allowance, not a measured requirement or a verified Leylak
allocation limit. GN's 300-dimensional float32 matrix for three million entries
alone occupies approximately 3.35 GiB; vocabulary metadata, M1/M2, and temporary
restricted candidate arrays need additional memory. No GPU is used. Inspect
peak RSS/runtime after the approved job and adjust future requests.

Use these standard Slurm resource directives, subject to Leylak's partition,
account, and scheduling rules (those values remain unverified):

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
```

In the batch script, limit numerical-library threads to the allocation:

```bash
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
```

Then execute the exact command above. If Slurm logs are directed into the
results directory, create that directory **before submission**, because Slurm
opens its log before the Python process can create directories. No script has
been submitted and no remote directories have been created. See the
[official sbatch documentation](https://slurm.schedmd.com/sbatch.html) for the
resource flags, accounting/partition options, and log handling.

## Inspection in Jupyter (example only; notebook unchanged)

```python
import json
import pandas as pd

path = "/home/aziza.carbonell/cs533_data/part2_results/gn_comparison.json"
with open(path, encoding="utf-8") as handle:
    results = json.load(handle)
table = pd.DataFrame(results["words"])
table[["word", "gn_key", "nn_gn_m1", "nn_gn_m2", "closer_to", "missing_models"]]
# Inspect a row's m1/m2/gn dictionaries for its neighbor lists and statuses.
```
