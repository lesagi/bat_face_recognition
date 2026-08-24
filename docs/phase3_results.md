# Phase 3 results — controls for the background leak and the resolution confound

624 runs, 2026-08-24, all clean (620 k-fold runs across 31 cells at 20/20 folds,
zero failures and zero `OK_WITH_WARNINGS`, plus 4 saliency runs).
MLflow: `phase3-20260824` and `phase3e-saliency-20260824`.
Raw: `outputs/phase3_results.parquet`, `outputs/phase3_analysis.json`.

Two findings change how earlier results must be read. One of them **refutes a
prediction this workstream itself made**.

---

## 1. The background leak is causal, not just available

`docs/background_leakage.md` showed the cue *exists*: a colour-histogram scorer
with no model reaches ROC-AUC 0.740 / 0.792 on `original`. Arm 3C trains a real
ArcFace end to end on images whose faces have been **inpainted away** (green-screen
matte, dilated 24 px, Telea fill, applied to train/val/test alike) and evaluates
it on held-out bats.

| | faceless model | random-background control | full model |
|---|---|---|---|
| mauritius | **0.831 ± 0.173** | 0.516 ± 0.097 | 0.890 ± 0.095 |
| rousettus | **0.780 ± 0.145** | 0.555 ± 0.074 | 0.860 ± 0.105 |

Paired by fold against its own control (Nadeau-Bengio corrected):
mauritius **+0.314**, CI [+0.229, +0.393], Cliff's δ = +0.82;
rousettus **+0.225**, CI [+0.165, +0.286], δ = +0.78. Both CIs exclude zero.

**A model that never sees a face reaches 93% (mauritius) / 91% (rousettus) of the
full model's ROC-AUC on held-out individuals.** The negative control sits at
chance, so the pipeline is not manufacturing the effect.

Read this as an *upper bound*, not a point estimate. These are tight face crops:
the dilated mask covers a median 89% (mauritius) / 77% (rousettus) of the frame, so
inpainting extrapolates most of each image from the surviving background rim and
therefore **amplifies** the cue rather than merely preserving it. A model-free
descriptor on the inpainted crops reads 0.979 / 0.840 against 0.740 / 0.792 on the
masked rim. No face pixel survives either way, which is what makes the arm valid.

**Consequence:** the `original` arm is not repairable and should not carry a
headline. Any `original` number is substantially background matching.

## 2. Image quality does NOT explain the species difference

This is the reversal. `docs/quality_parity.md` found that matching resolution
collapses the intrinsic-separability gap and concluded that "rousettus individuals
are less distinctive" is *substantially a measurement artifact*. The recognition
experiment does not support that.

First, where the difference actually lives. Across the published 360-run sweep the
species differ on exactly one background:

| background | ArcFace gap | AdaFace gap |
|---|---|---|
| green | −0.000 | −0.027 |
| original | +0.079 | +0.039 |
| **random** | **+0.116** | **+0.136** |

On `green` there is no mauritius advantage to explain. `original` is the leaking
arm. So `random` is the only place a species effect can be tested — which is why
arms 3A and 3B were extended to it (they were originally specified as
green/original only, and would have controlled the confound everywhere except
where the effect is).

Three independent controls, gap on `random` with 95% bootstrap CIs:

| control | ArcFace | AdaFace | Δ gap vs baseline |
|---|---|---|---|
| baseline (uncontrolled) | +0.116 [+0.012, +0.228] | +0.136 [+0.030, +0.253] | — |
| 3A `blur` (quality-matched) | +0.083 [+0.018, +0.172] | +0.210 [+0.077, +0.278] | −0.033 / **+0.074** |
| 3A `recrop` (filmed further away) | +0.114 [+0.001, +0.183] | +0.136 [+0.054, +0.204] | **−0.002 / −0.000** |
| 3B band-restricted (real pixels) | +0.073 [−0.023, +0.173] | +0.120 [−0.075, +0.215] | −0.043 / −0.017 |

**The gap survives every control.** The largest reduction across three methods and
two models is **0.043 ROC-AUC**, and the physically grounded control — actually
re-cropping mauritius from frames downsampled to rousettus' capture scale — leaves
it unchanged to three decimal places. One control (`blur`, AdaFace) *increased* it.

The publishable statement is a bounded negative: **matching image quality changed
the random-background species gap by at most 0.043 ROC-AUC across three
independent controls; the species difference is not a resolution artifact.**

This is a bound, not a null hypothesis test. The design's minimum detectable
difference is 0.440 ROC-AUC (`outputs/kfold_power_ceiling.json`) against gaps
around 0.12, so no comparison here can be "significant" and none is reported that
way. What carries the conclusion is that three controls that fail in *different*
directions agree, and that the one with no free parameters moved nothing.

Why this does not contradict the separability result: that analysis measured
embedding-space geometry (d′, Fisher ratio, silhouette) on a frozen ImageNet
backbone. This measures trained open-set verification. Resolution matching flattens
the former and not the latter, so the two are answering different questions — and
the recognition claim is the one the paper makes.

### Degradation also did not hurt mauritius' own score
Twelve paired comparisons of degraded vs baseline mauritius (same 16 bats, same
fold seeds): eleven CIs span zero. Only `recrop`/`original`/AdaFace shows a
decrease (−0.051, CI [−0.093, −0.014]). So the degradations removed measured image
quality without removing recognisable identity information.

## 3. The rousettus manifest asymmetry was harmless

Arm 3D re-ran the rousettus green arm on the 1059-image intersection of all three
backgrounds (the published green build had 34 extra frames with no `original_bg`
counterpart). Paired by fold against the published green arm:

| model | difference | 95% CI |
|---|---|---|
| Siamese | −0.035 | [−0.085, +0.017] |
| ArcFace | −0.012 | [−0.093, +0.071] |
| AdaFace | +0.037 | [−0.023, +0.104] |

All CIs contain zero. The asymmetry was not driving the published background
comparison, though the intersection manifest is the correct one to use going
forward (`configs/data/manifest_rousettus_green_intersect.yaml`).

## 4. A 20×20 Grad-CAM is now available

Arm 3E adds `replace_stride_with_dilation=[False, False, True]` to the ResNet-50
backbone (`model.backbone_dilated`), giving output stride 16 instead of 32.
Verified on the trained checkpoints: the CAM grid is **20×20 with 16 px cells** at
320 px input, against 10×10 / 32 px for the stock backbone. Parameter count is
identical (23,508,032), so this is a resolution change, not a new architecture.
Four runs (ArcFace × 2 species × {original, green}), checkpoints retained.

This removes the obstacle `docs/saliency_species.md` records — that an eye ROI is
smaller than one CAM cell at any input edge. It does **not** revisit that
document's conclusions: the randomisation control there still fails, and these
four runs are a single seed each and are not bit-reproducible (the DataLoader has
no explicit generator, see `docs/PHASE_4_CHECKLIST.md`). Qualitative cross-check
against Integrated Gradients only; no p-values.

---

## Caveats that must travel with these numbers

- **Every species comparison here is below the design's MDE** (0.440 vs gaps
  ~0.12). Effect size + CI + concordance across controls is the result. A bare
  p-value from 3A or 3B is not reportable.
- **~1.45× noise survives both 2A variants.** Blur removes noise while mauritius
  is the *noisier* species, so neither degradation closed that gap
  (`outputs/phase2a_review/README.md`). Residual image-quality confound in 3A.
- **3B's absolute scores are not comparable to full data.** Band restriction
  leaves 10 identities instead of 16/12 and 2–4 test identities instead of 2–6, so
  it scores ~0.09 higher on both species. Only the within-band *gap* is
  interpretable.
- **3C is an upper bound** on the background cue, for the reason given in §1.
- **`docs/quality_parity.md` overstates the quality gap** for the data the models
  trained on — see the correction inserted in that file. The published battery
  measured `aligned_224`; the sweep trained on `not_augmented/`, which for
  rousettus are different, sharper images.

## Reproduce

```bash
# builds
uv run python scripts/build_background_only.py --all
uv run python scripts/build_band_restricted.py
uv run python scripts/build_resolution_matched.py --variant blur   --verify
uv run python scripts/build_resolution_matched.py --variant recrop --verify --device cpu
uv run python scripts/build_random_degraded.py

# sweeps (two concurrent jobs maximum; one per card)
CONFIGS="..." MLFLOW_EXP=phase3-20260824 LOG_DIR=outputs/<arm> \
  bash scripts/phase3_matrix.sh <species> <gpu>

# analysis
uv run python scripts/aggregate_kfold.py --experiment-name phase3-20260824 \
  --output outputs/phase3_results.parquet
uv run python scripts/analyze_phase3.py
```
