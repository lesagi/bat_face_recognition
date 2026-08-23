# Background leakage in the `original` arm

**The verification task on the original background can be solved to ROC-AUC
0.74–0.79 using background colour alone — no face pixels, no model.** That is
most of what the trained models achieve on that arm, and it means the
`original` results cannot be read as evidence about faces.

Produced by `scripts/probe_background_leakage.py` (model-free) and the `C` arm of
`scripts/occlusion_probe.py` (model-side). Raw output:
`outputs/leakage/background_probe.json`.

## Why the leak exists, and why identity-disjoint splitting does not stop it

Evaluation is open-set **verification**: every unordered pair of test images is
scored and labelled 1 if the two images show the same bat
(`bat_evaluation.verification.predictions_from_embedding`).

Each bat was filmed in **exactly one video**. So:

- a **same-bat** pair is always two frames of one video → near-identical background;
- a **different-bat** pair is always two frames of different videos → different backgrounds.

"Do these two photographs share a background?" is therefore an almost perfect
proxy for "are these the same bat?", and it can be answered without looking at an
animal at all.

Identity-disjoint splitting does not help. It stops the model memorising *which*
bat is which, but the cue here is within-pair background **agreement**, which
requires no knowledge of any particular individual. A model that has never seen
a test bat can still notice that two of its photographs were taken in the same
room.

## The probe

Same test split, same pair construction, same metric — the only change is what
produces the score:

| | score comes from |
|---|---|
| model evaluation | cosine between the two images' embeddings |
| this probe | cosine between the two images' **background colour descriptors** |

The face is masked out using the green-variant segmentation
(`mask_from_green`), dilated by 24 px, and the descriptor is a 4×4×4 RGB
histogram of the remaining pixels plus per-channel mean and standard deviation.

## Result — 20 folds, both species

| species | background | ROC-AUC from background alone |
|---|---|---|
| mauritius | **original** | **0.740 ± 0.089** |
| rousettus | **original** | **0.792 ± 0.078** |
| mauritius | green | 0.498 ± 0.008 |
| mauritius | random | 0.489 ± 0.020 |
| rousettus | green | 0.500 ± 0.012 |
| rousettus | random | 0.497 ± 0.015 |

`green` (flat fill) and `random` (a fresh background per image) cannot carry
bat identity, so they are **negative controls**. Both land on chance, which is
what licenses believing the `original` numbers.

### The calibration that makes the controls trustworthy

At the first dilation tried, green came back at 0.59 — *not* chance, which would
have invalidated the whole probe. Sweeping the dilation shows why, and that the
far-field cue is real:

| face dilation | mau green | rou green | mau original | rou original |
|---|---|---|---|---|
| 0 px | 0.762 | 0.711 | 0.832 | 0.760 |
| 6 px | 0.592 | 0.555 | 0.804 | 0.766 |
| 14 px | 0.545 | 0.538 | 0.758 | 0.779 |
| **24 px** | **0.501** | **0.504** | **0.719** | **0.782** |

The green control falls monotonically to chance as the mask boundary is
excluded — that elevation was the build's 4 px mask feathering bleeding face
colour outward. Meanwhile `original` barely moves, and rousettus' is essentially
flat (0.760 → 0.782), which is the signature of a genuine background cue rather
than boundary bleed. Any `original` number quoted below 24 px dilation is not
trustworthy.

## What it does to the sweep results

Paired by fold, against the same test split (Nadeau–Bengio corrected):

| species | model | model | background-only | margin | p |
|---|---|---|---|---|---|
| mauritius | AdaFace | 0.915 | 0.740 | +0.174 | 0.094 |
| mauritius | ArcFace | 0.890 | 0.740 | +0.149 | 0.156 |
| mauritius | **Siamese** | 0.751 | 0.740 | **+0.010** | 0.949 |
| rousettus | AdaFace | 0.881 | 0.792 | +0.089 | 0.315 |
| rousettus | ArcFace | 0.860 | 0.792 | +0.068 | 0.497 |
| rousettus | **Siamese** | 0.747 | 0.792 | **−0.045** | 0.677 |

**No model significantly beats a scorer that never sees a face.** The Siamese
network is indistinguishable from the baseline on mauritius and *loses to it* on
rousettus. ArcFace and AdaFace keep a margin of +0.07 to +0.17, which is the
honest measure of what the face contributes — far less than the raw 0.86–0.92
suggests.

Note that these p-values are also bounded by the design's power ceiling
(`docs/` → `analyze_power_ceiling.py`: MDE = 0.44), so "not significant" here
carries the usual caveat. The point stands on the effect sizes: a +0.01 margin
needs no significance test to be unimpressive.

## Model-side confirmation

`occlusion_probe.py` arm **C** removes the face (dilated mask, flat-filled with
the mean background colour) and evaluates a trained original-background
checkpoint:

| arm | mauritius | rousettus |
|---|---|---|
| clean | 0.759 | 0.959 |
| texture removed (shape only) | 0.967 | 0.910 |
| shape removed (texture only) | 0.921 | 0.789 |
| **face removed (background only)** | **0.902** | **0.781** |

For rousettus, removing the face costs only 0.18 and still leaves 0.781 —
landing almost exactly on the model-free probe's 0.792. Two independent methods
agreeing.

The mauritius column, where every ablation *improves* on clean, should not be
over-read: this arm uses the baked split with only **2 test identities**, where
fold-to-fold standard deviation is 0.10–0.21. It is a single noisy split, not a
measurement. (A per-fold version is impossible with the pruned k-fold
checkpoints; the background-only *training* arm is the rigorous test.)

## Consequences

1. **The `original` arm cannot be repaired with this data.** Each bat has one
   video, so there is no way to construct a same-bat pair from two different
   backgrounds. This is a property of the dataset, not of the split or the code.
2. **The headline should move to `green`**, where the probe confirms zero
   leakage (0.498 / 0.500) and both species still score well above chance
   (mauritius AdaFace 0.854, rousettus AdaFace 0.829). Those are the defensible
   numbers.
3. **"Background matters, in a consistent order" is not a finding about
   backgrounds helping recognition.** `original > green` is at least partly
   `original` containing a shortcut that `green` removes.
4. Any future dataset collection should film each individual in **more than one
   setting**, which is the only real fix.
