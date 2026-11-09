# Part 2.5: semantic drift from the three saved models

`src/semantic_drift.py` reads M1 (P1), M2 (continued on P2), and M3 (fresh P2).
M3 is the fresh model called M2s in the professor's screenshot. No model is
trained, vocabulary-updated, saved, or overwritten. Google News numerical
comparisons remain a separate Leylak task using the shared GN lookup rule.

## Target words

Use every Table 1 word and the five requested additions, exactly as spelled:

- Drift candidates (30): zoom, delta, booster, variant, lockdown, distancing,
  mask, quarantine, positive, spike, remote, essential, trump, pence, meta, ai,
  chair, stream, streaming, tweet, platform, podcast, viral, vaccine, tesla,
  uber, apple, cloud, corona, woke.
- Controls (9): river, piano, mountain, wine, coffee, bread, doctor, bridge,
  baseball.
- Additional (5): privacy, internet, phone, election, climate.

Every word gets an output row. Membership is checked by exact key in each
model. No case fallback or replacement words are used for these local models.
An unavailable measure is JSON null / blank CSV, with per-measure status and
per-model membership fields. Partial observations are retained: missing M2
prevents cos_cont but not comparisons between M1 and M3. Zero/nonfinite query
vectors and insufficient neighborhoods are reported as undefined. Invalid
alignment anchors stop analysis explicitly rather than changing the anchor set.

## Alignment and measures

Select the 20,000 most frequent M1 words **among the M1/M3 intersection**.
Sort by M1's saved `count` attribute, descending; ties use M1 vocabulary index.
Do not intersect after taking M1's top 20,000. Include stop words and evaluation
targets. Require the full 20,000 anchors; do not silently reduce this number.

Let X contain fresh M3 anchor row vectors and Y contain corresponding M1 rows.
Length-normalize every row of X and Y before fitting. Compute:

```text
U, singular_values, Vt = svd(X_normalized.T @ Y_normalized)
R = U @ Vt
aligned_M3_word = M3[word] @ R
```

R minimizes `||X_normalized @ R - Y_normalized||_F` subject to orthogonality.
No mean centering, learned scale, or determinant correction is used; orthogonal
reflections are allowed. Normalize copies, never model arrays. Save the anchor
list/counts, R, singular values, fit error, and orthogonality diagnostics in JSON.

- `cos_cont`: cosine(M1[word], continued M2[word]), without alignment.
- `cos_align`: cosine(M1[word], fresh M3[word] @ R).
- `nn_at_10`: size of the intersection of the two sets of ten nearest
  neighbors from M1 and fresh M3, divided by ten. Each neighborhood uses its
  model's entire vocabulary, with no filtering, shared-vocabulary restriction,
  or alignment. Keep the two lists and intersection in JSON for audit.

Cosines use vector lengths in their denominator; clipping only corrects
floating-point excursions beyond [-1, 1].

## Ranking, correlations, and examples

Rank all words with defined `cos_align` in ascending order: lower aligned
cosine means more drift. Use alphabetical ties. CSV/JSON word rows follow this
order, with unranked rows last. Choose the five lowest as most drifted, and
five highest as most stable (descending cosine, alphabetical ties), keeping
the two selections disjoint. This follows the user's request for five stable
words; the screenshot requires only two. Other measures remain available for
comparison and do not determine the ranking.

If fewer than ten words can be ranked, flag `insufficient_rankable_words`
and report the available selections instead of substituting other words.
For selected words, reuse the existing trainer's unfiltered eight-neighbor
helper for M1 and continued M2, with actual cosine scores. Explicitly mark a
selected word absent from M2 rather than changing the selection.

Calculate all three pairwise Spearman correlations on available pairs across
all target groups. Report rho, p-value, n, and the included words. Constant
inputs or fewer than two pairs yield explicit status/null values. A p-value
may also be null when undefined, such as with only two pairs.

Use privacy, internet, and phone as the fixed three raw-versus-aligned M1/M3
examples. Show both values, or explicit missing/undefined statuses; do not
replace a missing example. Raw M1/M3 cosine is only a diagnostic, never a
drift ranking measure.

Independently trained spaces have arbitrary axes: a rotation or reflection
can preserve every within-model similarity while changing raw cross-model
cosines. Thus raw M1/M3 cosine cannot distinguish coordinate orientation from
meaning change. Procrustes uses common anchors to establish comparable axes.
Continued M2 starts in M1's coordinate system, so its direct cosine is meaningful.
Raw M1/GN cosine has the same orientation problem; GN numerical examples will
be handled separately on Leylak.

## Commands and results

From `C:\Projects\IR\CS533_HW1` in PowerShell, validate paths/plan only:

```powershell
.\.venv\Scripts\python.exe -B -m src.semantic_drift `
  --m1 "C:\Projects\IR\data\models\w2v_P1.model" `
  --m2 "C:\Projects\IR\data\models\w2v_P2_continued.model" `
  --m3 "C:\Projects\IR\data\models\w2v_P2_fresh.model" `
  --output-dir "C:\Projects\IR\data\part2_results" `
  --dry-run
```

Remove `--dry-run` only after approval for the full analysis. All four paths
are configurable. The dry run does not load models or create output files.
Existing dependencies are sufficient: Gensim, NumPy, SciPy, and scikit-learn
(imported by the reused trainer helper).

The full run writes `semantic_drift.csv` (the ranked 44-word measurement table)
and `semantic_drift.json` (all measurements, membership, alignment, correlations,
rankings, selected neighbors, examples, interpretation, paths, and versions).
Results must be outside the repository. Existing result files or an analysis
marker cause refusal. Use exclusive staged publication, as in the trainers;
on a caught failure remove only this attempt's artifacts. A forced termination
can leave `.semantic_drift.in_progress`, which requires investigation. No
automatic retry is performed.

## Synthetic verification

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

The focused tests recover a known nonsymmetric rotation in the correct
M3-to-M1 direction, reject its transpose, verify length-normalization invariance,
allow reflections, and check anchor intersection/order, exact NN overlap,
partial missing words, tied Spearman ranks, undefined statistics, rankings,
eight-neighbor reports, strict JSON/CSV output, dry-run behavior, source-file
preservation, and protection against overwriting results. All model files used
by these tests are tiny temporary fixtures; the actual assignment models are
not loaded or analyzed.

Validation: all 22 focused tests passed. The real-input dry run passed with
the specified three model paths, 44 words, 20,000 anchors, and external results
directory. Full analysis and actual vocabulary/missing-word measurements
remain pending approval.
