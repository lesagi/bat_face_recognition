# Where this project stands — read this first

Written to be picked up cold. Every claim points at the artifact that backs it.
Last updated 2026-08-29, after 1,344 training runs across four workstreams.

---

## The bottom line, in four sentences

Individual bats **are** identifiable well above chance, and that is solid — 18 of
18 cells, q ≤ 0.0032. But **no trained model in this project demonstrably beats
six numbers** summarising the colour of the face, so "a neural network recognises
bat faces" is not currently supportable. Two independent leaks — the background on
`original`, global colour on `green` — both trace to the same cause: **each bat was
filmed in exactly one video**, so anything constant within a clip is a free
identity cue that no identity-disjoint split can remove. The single most valuable
thing that could happen to this project is **a second recording session for a
handful of individuals**; almost every open question reduces to that.

---

## What is defensible

| claim | evidence |
|---|---|
| Individual bats identifiable above chance, both species, every condition | 18/18 cells, q ≤ 0.0032, `outputs/kfold_species_significance.json` |
| The two species' datasets are **not** comparable | 11 of 12 quality metrics differ; effective resolution separates them completely (Cliff's δ = +1.00, no overlap). `docs/quality_parity.md` |
| The `original` background is substantially solvable **without a face** | Model-free probe 0.740/0.792; a model trained on face-deleted images reaches 0.831/0.780 against a control at chance. `docs/background_leakage.md`, `docs/phase3_results.md` |
| Image quality does **not** explain the species difference | Survives three independent controls; largest reduction 0.043 ROC-AUC. `docs/phase3_results.md` |
| Bat-specific training moves attribution **off** eyes and nose, toward the periphery | 38 of 40 models, both species, both backgrounds, q ≤ 0.007. `docs/saliency_species.md` |
| Six global colour numbers reach 0.849 / 0.830 | `docs/occlusion_results.md`, `outputs/leakage/face_colour_probe.json` |

## What has been refuted — five claims, all by their own controls

| claim | what killed it |
|---|---|
| "The mauritius model learned to look at the eyes — 81% vs a 16% chance rate" | Wrong null. An untrained network scores **64%** on the same images. |
| "The trained-minus-untrained contrast survives, opposite directions per species" | Pre-registered, then tested on 40 models: does not replicate on a clean background. |
| "Training differentiates the species at the **nose**" | Reversed on 10 folds (δ −0.44, CI excluding zero). The exploratory reading was a **single-seed artifact**. |
| "The species difference is substantially a measurement artifact" | Phase 3: the gap survives blur, re-crop and band-restriction. This **reverses** `quality_parity.md`'s reading for recognition. |
| "The networks recognise faces" | No model beats a six-number colour summary; rousettus Siamese is significantly *worse*. |

## What is open, and what it would take

- **Is the colour cue biology or capture?** Individuals may genuinely differ in coat
  colour (a real answer to "what distinguishes bats"), or it may be a per-video
  signature. Grey-world white balancing does **not** remove it, so it is not simple
  illumination. **One video per bat makes these indistinguishable.** Needs a second
  session.
- **Which facial features carry identity?** The interpretability evidence is
  *silent*. Every measurement that looked like an answer turned out to be a method
  artifact, an image property, a single-seed artifact, or background-confounded.
- **Do the models beat the colour baseline?** Undetectable at this power. The design's
  minimum detectable difference is **0.440 ROC-AUC** against gaps around 0.1
  (`outputs/kfold_power_ceiling.json`). More folds cannot help — the driver is
  `n_test/n_train`, not fold count.

---

## Everything that was checked

| # | check | verdict | where |
|---|---|---|---|
| 1 | Species dataset comparability (12 metrics, TOST) | not comparable | `docs/quality_parity.md` |
| 2 | Seeded k-fold with varying held-out counts | built, `exact` mode frozen | `bat_data.splitter` |
| 3 | 360-run sweep, 18 cells × 20 folds | all above chance; no model beats another | `outputs/kfold_results.parquet` |
| 4 | Statistics defects (ROC-AUC never permuted, best-of-5, floor censoring) | 3 fixed | `bat_stats.model_comparison` |
| 5 | Background leakage, model-free | 0.740/0.792 on `original`; green/random at chance | `docs/background_leakage.md` |
| 6 | Power ceiling | MDE 0.440 vs largest gap 0.207 | `outputs/kfold_power_ceiling.json` |
| 7 | 3A — degraded mauritius (blur + re-crop) | gap survives | `docs/phase3_results.md` |
| 8 | 3B — band-restricted, real in-band pixels | gap survives | `docs/phase3_results.md` |
| 9 | **3C — trained on face-deleted images** | **0.831/0.780 vs control at chance** | `docs/phase3_results.md` |
| 10 | 3D — rousettus green intersection | asymmetry harmless, all CIs contain zero | `docs/phase3_results.md` |
| 11 | 3E — dilated backbone, 20×20 Grad-CAM | works; 10×10 → 20×20 verified | `configs/model/arcface_dilated.yaml` |
| 12 | Saliency randomisation control | **fails** — untrained scores 64% | `docs/saliency_species.md` |
| 13 | **Pre-registered eye/nose test, 40 models** | **both hypotheses fail** | `docs/saliency_preregistration.md` |
| 14 | ROI occlusion, area-matched | **positive control fails** → not interpretable | `docs/occlusion_results.md` |
| 15 | **Six-number colour baseline, paired per fold** | **no model beats it** | `docs/occlusion_results.md` |

---

## Method patterns now established — keep using them

1. **Pre-register before the data exists.** Two hypotheses were registered and both
   failed; a third (the nose) reversed. Without registration the nose would have
   been written up from the single seed it was found in. `git log` order is the
   audit trail — the pre-registration commit must remain an ancestor of the result.
2. **Every arm needs a negative or positive control**, and the control gates the
   result. This caught five defects, including two of mine in this session.
3. **Quote against the right null.** Not chance — the **six-number baseline**
   (`scripts/probe_face_colour.py`, ~1 minute). Area-share nulls and chance nulls
   have both produced false claims here.
4. **Effect size + CI + the MDE, never a bare p-value.** Below 0.440 ROC-AUC,
   "not significant" means undetectable, not equal.
5. **Interpretability needs many models.** Going from 1 model to 10 flipped the sign
   of the primary effect; per-fold contrasts ranged +0.046 to −0.586.

## Traps that silently produce wrong answers

- **Any saliency output from before 2026-08-25 is suspect.** `find_run_dir_for_edge`
  ignored background and picked by mtime, so it could load a *dilated* checkpoint
  into a *stock* ResNet — shapes match, so nothing raised. Fixed; pin with
  `--checkpoint SPECIES=PATH`. Regression tests in `tests/`.
- **SmoothGrad was unseeded** until 2026-08-25: ±6 pp run-to-run on the pointing
  game. Fixed (per-image seed); runs now agree to ~7 significant figures.
- **`aggregate_kfold.py` keys cells on `(species, model, bg, fold_id)`** with no arm
  dimension. Give every new arm a distinct `data.background` **label** — it is
  label-only (`runtime.py:249`, `:2069`); `load_manifest` never filters on it. The
  CSV column must stay the `green`/`original`/`random` Literal.
- **`kfold_matrix.sh` hardcodes its nine configs.** Use `scripts/phase3_matrix.sh`
  (same resume/warning logic, `CONFIGS` from env).
- **`quality_parity.md` measured `aligned_224`; the sweep trained on
  `not_augmented/`.** Byte-identical for mauritius, *different images* for
  rousettus. Correction is in that file; use the right-hand column.
- **Checkpoints are 376 MB** and `/home` has hit 100%. `scripts/saliency_seed_pipeline.sh`
  shows the train → analyse → delete pattern that caps peak usage.
- **`analyze_power_ceiling.py` int-casts `fold_id`** and crashes on fold-less runs.
  Keep single-seed arms in their own MLflow experiment.
- **Still unfixed:** `DataLoader(shuffle=True)` has no explicit generator, so runs are
  not bit-reproducible (`docs/PHASE_4_CHECKLIST.md`).

---

## What to do next, in priority order

1. **Collect a second recording session for even 3–5 individuals.** It is the only
   thing that separates biology from capture signature, and it is now implied by two
   independent results. Everything else is analysis of a constraint that only
   collection can break.
2. **Re-baseline the paper's claims against the six numbers**, not against chance.
   `scripts/probe_face_colour.py` gives the paired per-fold margin with a CI.
3. **Rewrite the species and saliency sections.** Both conclusions have flipped:
   the species gap is real and not a resolution artifact; the eye-attention account
   does not survive its controls.
4. Optional: re-run the saliency ROI analysis on `green` checkpoints at 320 px — the
   machinery is built (`--background green`), only the checkpoints need retaining.

## How to run things

```bash
uv sync --frozen
uv run pytest                                  # 559 passed, 1 skipped
uv run pre-commit run --all-files              # the only quality gate

# the baseline every claim should clear
uv run python scripts/probe_face_colour.py

# a new sweep arm (two concurrent jobs maximum; one trainer saturates ~40 of 112 cores)
CONFIGS="exp_a exp_b" MLFLOW_EXP=<name> LOG_DIR=outputs/<arm> \
  bash scripts/phase3_matrix.sh <species> <gpu>
uv run python scripts/aggregate_kfold.py --experiment-name <name> --output <parquet>
```

**Docs map:** `quality_parity.md` (datasets) · `background_leakage.md` +
`phase3_results.md` (the leak, and the controls) · `saliency_species.md` +
`saliency_preregistration.md` (attention) · `occlusion_results.md` +
`occlusion_preregistration.md` (dependence, and the six-number baseline) ·
`PHASE_SUMMARY.md` (the earlier workstream, phase by phase).
