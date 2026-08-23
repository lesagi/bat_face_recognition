# Peer-review response — phase summary and handoff

One page per phase: what was asked, what was done, what held up, what did not.
Written to be picked up cold in a new session. Every claim points at the artifact
that backs it.

**Status as of 2026-08-23:** Phases 0–1 complete. Phase 2 partially complete.
Phase 3 not started and its scope is an **open decision** (see the end).

---

## The four original review comments, and where they landed

| # | Asked for | Outcome |
|---|---|---|
| 1 | A measure like SNR showing the species' datasets do not differ | **Refuted the premise.** 11 of 12 quality metrics differ; nothing is equivalent under TOST. `docs/quality_parity.md` |
| 2 | 20 folds per (model, background), then species significance tests | **Done**, 360 runs. But the design cannot detect the effects in question — see 1F. |
| 3 | Kernel-smoothed saliency showing each species' focus | **Answered, then largely retracted.** The randomisation control fails. `docs/saliency_species.md` |
| 4 | A p-value per species | **Done**, plus three defects fixed. All 18 cells above chance (q ≤ 0.0032). |

---

## Phase 0 — snapshot

Annotated tag `shiloutte-controls-pre-kfold` on 932b414. Local only, not pushed.
Excludes two files untracked at the time.

## Phase A — are the two datasets comparable?

**Verified.** They are not.

- Effective resolution separates the species **completely**: median native head-box
  614 vs 312 px, Cliff's δ = **+1.00**, no overlap at all (mau min 441 > rou max 407).
  48.4% of rousettus crops are upscaled at 320 px vs 0.2% of mauritius.
- 11 of 12 quality metrics differ after Benjamini–Hochberg; **zero** are equivalent
  under TOST at |g| < 0.5. The one non-significant metric is under-powered, not equivalent.
- Ranked by standardised difference, resolution is largest (g = +3.72), then
  **colourfulness** (g = +2.12) — larger than any sharpness or noise measure.
- mauritius is 2.2–3.1× **noisier**, so part of its 3.78× "sharpness" advantage is
  noise counted as detail. The three sharpness measures disagree in exactly the
  pattern a resolution difference predicts (`tenengrad`, least noise-sensitive,
  is smallest and non-significant).
- **Intrinsic separability**: matching sample sizes changes nothing (ratios
  unmoved); matching *resolution* collapses the d′ gap and mildly reverses it,
  and halves the Fisher/silhouette gaps. So "rousettus individuals are less
  distinctive" is substantially a measurement artifact.

**Refuted (1C):** the colour difference is **not** pigmentation. White-balanced
hue is indistinguishable between species (g = −0.05, p = 0.98); both chromaticity
measures are non-significant and the nearest one points the *opposite* way from
colourfulness. Colour is a third confound (lighting), not biology.

Artifacts: `docs/quality_parity.md`, `outputs/quality/{parity_report,separability}.json`,
`outputs/quality/native_boxes.parquet`, `outputs/quality/battery.parquet`.

## Phase B — folds that vary honestly

**Verified.** The previous 90 "multi-seed" runs varied only the training RNG; the
data split was frozen, so every run held out the same bats. Now the split is
re-drawn from the seed in memory, and the *number* of held-out identities varies
too (`test_fraction` became a floor).

- `size_mode="exact"` is byte-identical to the old behaviour, so the 90 published
  runs stay reproducible. Pinned by a frozen-fixture test.
- Full provenance in MLflow (`split.*` params, assignment digest, source manifest
  hash) plus a `split_assignment.json` artifact per run.
- **40 distinct assignments** = 2 species × 20 folds, confirming all nine
  model/background cells within a fold share one partition — which is what makes
  the paired model comparisons valid.

## Phase C — the 360-run sweep

360/360 runs, 18 cells × 20 folds, no gaps.

- **All 18 cells above chance** after BH across all 18 (q ≤ 0.0032). Two cells
  needed the harmonic-mean combiner to get there (median per-fold p as high as
  0.170), which is the case the machinery exists for.
- **Species differ in only 2 of 9 cells**, both on the random background
  (AdaFace δ = +0.63 q = 0.006; ArcFace δ = +0.48 q = 0.044). Not a general
  species effect — and still confounded by resolution.
- **No model beats any other** in any of 18 comparisons (all p ≥ 0.24) under the
  Nadeau–Bengio correction. A naive paired t-test would have called several
  significant.
- **Refuted a working assumption:** more held-out bats does **not** stabilise the
  estimate. Rank correlation between test size and spread is *positive*
  (+0.60 mau, +0.40 rou); R² of the split shape on the score is 0.042 / 0.003.
  The instability is *which* bats are held out, not how many.

Artifacts: `outputs/kfold_results.parquet`, `kfold_split_sensitivity.json`,
`kfold_species_significance.json`. MLflow experiment `kfold20-20260805`.

## Phase D — statistics

Three defects found and fixed:

1. **ROC-AUC was never permuted.** The permutation test covered only thresholded
   metrics, so the report's central significance claim cited **F1's** p-values
   while attributing them to ROC-AUC. Now permuted and pre-declared primary.
2. **`min()` over five correlated metrics** in the summary script — best-of-five
   cherry-picking. Replaced by one metric fixed in advance.
3. **Floor censoring unreported.** With B = 1000 the smallest expressible p is
   1/1001; 290 of 360 runs sit exactly there, so those are upper bounds. Now
   marked `<` and the floor is printed.

New `bat_stats.model_comparison`: Nadeau–Bengio corrected t-test, TOST, Cliff's δ,
Hedges' g, hierarchical bootstrap, harmonic-mean p (validated against the
reference package's published example to six significant figures), BH.

## Phase E / 1E — saliency: answered, then largely retracted

**This is the most important reversal in the whole workstream.**

Two method defects were found in sequence:

1. **Grad-CAM discarded a third of the images, unevenly by species** (20% mau /
   46% rou). Cause: `class_logit` is a scale-invariant cosine while Grad-CAM
   assumes the target grows with activation magnitude, so pooled weights can come
   out net-negative and ReLU erases the map. Confirmed by the scale-*dependent*
   `emb_mag` target giving 0% / 0%. The first hypothesis (weak top-class cosine)
   was **refuted**: p = 0.84 / 0.92.
2. **The randomisation control was broken and passed vacuously.** It zeroed every
   1-D parameter including BatchNorm γ, making the network emit identically zero.
   "100% degenerate, nothing survives" demonstrated a dead model, not a
   learning-dependent explanation.

With Integrated Gradients (pixel resolution, zero attrition) and a **working**
control — a functioning untrained network at default init:

| | trained | untrained | learning-attributable |
|---|---|---|---|
| mauritius eye density | 1.669 | **1.771** | nothing; untrained is higher |
| rousettus eye density | 0.911 | **1.322** | training moves it *away* |
| mauritius peak on eye | 82.8% | **64.1%** | +18.7 pp |
| rousettus peak on eye | 14.6% | **47.9%** | −33.3 pp |
| species δ, eye density | +0.98 | **+0.94** | ≈ nothing |

**Refuted:** "the mauritius model learned to look at the eyes, 81% vs a 16%
chance rate". The correct baseline is 64%, not 16% — gradient attribution lands
on high-contrast structure and eyes are dark spots on a lighter face. And the
species difference survives randomisation almost unchanged, so it is a property
of the *images*, most plausibly the resolution confound again.

**Survives:** the trained-minus-untrained contrast *within* each species, which
has opposite signs — mauritius moves toward the eyes, rousettus away. That is
learning-dependent, but it is a different claim and needs its own test rather
than inheriting the old p-values.

## Phase 1A/1B — background leakage

**Verified, and it undermines the best-scoring arm.**

Each bat was filmed in exactly one video, so same-bat pairs always share a
background and different-bat pairs never do. Scoring the identical test pairs with
the identical metric but using a **background-colour descriptor instead of a
model**:

| species | `original` | `green` | `random` |
|---|---|---|---|
| mauritius | **0.740 ± 0.089** | 0.498 | 0.489 |
| rousettus | **0.792 ± 0.078** | 0.500 | 0.497 |

Both negative controls sit on chance across 20 folds, which is what makes the
`original` figures believable. The 24 px mask dilation is calibrated, not guessed
— at smaller values the build's 4 px feathering contaminates the green control.

Paired by fold, **no model significantly beats a scorer that never sees a face**;
the Siamese network ties on mauritius and *loses* on rousettus.

Identity-disjoint splitting does not protect against this: the cue is within-pair
background *agreement*, not memorised identity. **It is unfixable with this data**
— one video per bat means no same-bat pair can span two backgrounds. The headline
belongs on `green`.

Artifact: `docs/background_leakage.md`, `outputs/leakage/background_probe.json`.

## Phase 1F — the power ceiling

**Verified.** Minimum detectable model difference at 80% power is
**0.440 ROC-AUC**, against a largest observed gap of **0.207**. Even the most
favourable configuration (2 held-out bats) only reaches 0.280.

So "no model beats another" provably means *undetectable*, not *equal*. The
Nadeau–Bengio inflation is 12.3×, driven by `n_test/n_train` = 0.564 — therefore
**more folds cannot help**; only a smaller test fraction would, which trades
against stability.

This replaces the earlier "add more individuals" recommendation, which is not
actionable (there are no more bats).

Artifact: `outputs/kfold_power_ceiling.json`, `scripts/analyze_power_ceiling.py`.

## Phase 1G — housekeeping

`uv run pre-commit run --all-files` passes cleanly **for the first time** — it had
been failing since before this workstream (3 ruff errors plus black/isort on 5
files, all pre-existing). 550 tests pass.

## Phase 2 — dataset builds

- **Done:** `data/manifests/rousettus_green_bg_intersect_manifest.csv` — 1059
  images, all 12 identities, the intersection of all three backgrounds. Fixes the
  asymmetry where rousettus green had 1093 images against original/random's 1059
  (34 green-only frames with no `original_bg` file on disk, concentrated in larry
  and charlie). mauritius was already consistent at 520 everywhere.
- **Not done:** resolution-degraded mauritius (2A), band-restricted arm (2B).
  Both exist only to feed Phase 3 arms whose value is now in question.

---

## Phase 3 — DECIDED: run the full set (~342 runs)

The user's decision, 2026-08-23: **the full set, for completeness — no corners
cut.** Recorded here because two Phase-1 findings argue the other way and whoever
picks this up will notice the tension; the decision was made with them in view.

| arm | cells | runs | purpose |
|---|---|---|---|
| 3A mauritius resolution-degraded | ArcFace/AdaFace × green/original | 80 | Species comparison with resolution controlled, all 16 bats kept |
| 3B band-restricted, both species | ArcFace/AdaFace × green/original | 160 | Same question on *real* pixels rather than simulated degradation |
| 3C background-only training | ArcFace × original, both species | 40 | Causally tests the leak — the central Phase-1 finding |
| 3D rousettus green re-run | all 3 models × green | 60 | Clean background comparison on the intersection manifest |
| 3E dilated-backbone saliency | ArcFace × 2 species, seed 42 | 2 | 20×20 Grad-CAM cross-check against IG |

**Gated on Phase 2 first:** 3A needs 2A (resolution-degraded mauritius) and 3B
needs 2B (band-restricted manifests). Neither is built yet. 3D can start
immediately — its manifest is done.

### Interpret 3A and 3B against the power ceiling, not against p = 0.05

This is the caveat that makes the full set worth running rather than misleading.
Phase 1F establishes a minimum detectable difference of **0.44 ROC-AUC**; the
species differences at stake are around 0.1. So 3A and 3B **will** return
"not significant" whatever the truth is, and that must not be written up as
"resolution does not explain the species gap".

What they *can* deliver, and how to report them:

- **Effect sizes and their direction**, which need no significance test. If
  matching resolution shrinks the species gap in both arms, that is evidence
  regardless of p — and it is exactly the pattern the Phase-A separability result
  already predicts.
- **Agreement between two independent controls.** 3A simulates the degradation
  and keeps all 16 bats; 3B uses real in-band pixels but drops to 10/11 bats.
  They fail in different directions, so concordance is a much stronger claim than
  either alone. That is the real reason to run both.
- **A quantified upper bound.** "Matching resolution changed the gap by at most
  X" is a publishable statement even when X's confidence interval spans zero.

Report effect size + CI + the MDE together, every time. Do not report a bare
p-value from these arms.

### Operational notes for whoever runs Phase 3

- **Two concurrent jobs maximum.** Measured: 1→2 gained 18%; 3 jobs *lost* 35%.
  One trainer saturates ~40 of 112 cores and the bottleneck is not cores.
- `scripts/kfold_matrix.sh` supports `LOG_DIR`, `FOLDS` and resume-skip (it skips
  any (experiment, fold) already recorded clean `OK`). Give each shard its own
  `LOG_DIR` or they interleave appends into one summary.
- A run can exit 0 having produced **no metrics** (`_safe` swallows step
  failures); the runner flags these `OK_WITH_WARNINGS`. Check that and
  `aggregate_kfold.py`'s coverage report.
- Never edit a running bash script — bash re-reads by byte offset. Python is safe.
- `nvidia-smi` first; the box is shared with another user.

## Cross-cutting lesson

Five of the findings in this workstream were **method defects producing
confident-looking wrong answers**, not data findings: p-values attributed to the
wrong metric, a bootstrap that corrupted clustering measures, saliency maps
silently dropping a third of images unevenly, a control that passed because the
network was dead, and a figure overwritten by its own control. Each was caught by
an internal consistency check rather than by inspection — densities that must sum
to 1 not summing to 1, a confidence interval not containing its own point
estimate, a negative control not returning chance. **Build the consistency check
before trusting the number.**
