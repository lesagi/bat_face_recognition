# Periphery occlusion — the test cannot answer the question, and why that matters

Pre-registered in `docs/occlusion_preregistration.md` (with Amendments 1–3).
320 runs total (160 with a defective fill, 160 after the fix), all clean.
Raw: `outputs/occlusion/roi_occlusion_{results.parquet,analysis.json}`.

**Result: the positive control fails for mauritius, so H3 and H4 are not
interpretable — and the reason is more interesting than the hypothesis was.**

## What was asked

The pre-registered saliency test left one effect that replicated in 38 of 40
models: bat-specific training moves attribution *away* from the eye/nose discs and
*toward* the periphery. Attribution can be wrong about causal dependence, so the
claim was converted into an ablation — destroy the pixels and measure the loss.

Four arms on `green`, ArcFace, 20 folds, both species: `none`, `centre` (eye+nose
discs destroyed, ~24%), `matched` (an identical pixel count, on the animal,
furthest from the discs — the area control), `periphery` (everything except the
discs, ~76%).

## What happened

| arm | mauritius median ROC-AUC | rousettus |
|---|---|---|
| `none` | 0.805 | 0.816 |
| `centre` | 0.854 | 0.873 |
| `matched` | 0.894 | 0.788 |
| `periphery` | 0.825 | 0.752 |

Cost of each ablation, paired by fold (positive = it hurt):

| species | arm | drop | 95% CI |
|---|---|---|---|
| mauritius | centre | +0.008 | [−0.057, +0.043] |
| mauritius | matched | **−0.063** | [−0.107, −0.003] |
| mauritius | periphery | **−0.021** | [−0.051, +0.119] |
| rousettus | centre | +0.023 | [−0.049, +0.047] |
| rousettus | matched | +0.041 | [−0.015, +0.077] |
| rousettus | periphery | **+0.083** | [−0.027, +0.144] |

**The positive control required `periphery` to cost ≥ 0.05 in both species.**
rousettus passes (+0.083). **mauritius fails**: destroying 76% of every image
*improved* median ROC-AUC by 0.021. So nothing may be claimed from H3 or H4, as
declared in advance.

For the record, both were inconclusive anyway — neither equivalent under TOST nor
significantly different:

| species | drop(centre) | drop(matched) | difference | TOST | verdict |
|---|---|---|---|---|---|
| mauritius | +0.008 | −0.063 | +0.036 [+0.003, +0.086] | not equivalent (p=0.446) | inconclusive |
| rousettus | +0.023 | +0.041 | −0.049 [−0.166, +0.038] | not equivalent (p=0.289) | inconclusive |

## Why the control fails: the signal is not localised

Six numbers — the per-channel **mean and standard deviation of the face region**,
with no spatial information whatsoever — identify a bat at:

| species | ROC-AUC from 6 global colour numbers | trained model on the same images (`none`) |
|---|---|---|
| mauritius | **0.777** | 0.805 |
| rousettus | **0.792** | 0.816 |

Robust, not an artifact of one split: median **0.780** / **0.784** across 30 random
3-identity subsets (5th–95th percentile 0.63–0.95 and 0.67–0.90, so the spread is
wide and any single split can look much better or worse — the baked test split
alone gives 0.897 / 0.849, which is the high end).

**The trained models add very little over six global numbers.** And a global
statistic cannot be removed by destroying any particular region: mask 76% of the
crop and the surviving 24% carries almost the same mean and standard deviation.
The positive control could not have passed for a cue of this kind.

This reframes the question the arm was built to answer. "Does the model use the
eyes or the periphery" presupposes that the cue is localised somewhere. For
mauritius on `green`, largely it is not.

### It is not simple illumination

The obvious explanation is per-video exposure and white balance — each bat was
filmed in exactly one video, so global colour would encode *the video*. **Tested
and rejected:** grey-world white balancing leaves the cue intact (mauritius
0.777 → 0.778, rousettus 0.792 → 0.784). So it is not a simple illumination
offset.

What remains is not resolvable with this data, and the two options have opposite
meanings:

- **genuine individual colouration** — bats really do differ in coat colour and
  texture statistics, which would be a real biological answer to "what
  distinguishes individuals", and would explain why it is spatially distributed;
- **a per-video capture signature** surviving white balance — sensor response,
  compression, focus, or anything else constant within a clip.

One video per bat makes these indistinguishable, exactly as with the background
leak (`docs/background_leakage.md`). It is the same structural limitation
reappearing in the animal's own pixels rather than in the scene behind it.

## A second reason, and it is a flaw in the pre-registration

The 0.05 control threshold was declared without checking it was detectable. The
per-fold spread of ROC-AUC in this design is ~0.12, so with 20 folds the standard
error on a median drop is ~0.027 and a 0.05 shift sits at roughly 2 SE — the
mauritius `periphery` CI is [−0.051, +0.119], wide enough to contain both zero and
0.05. **The control was underpowered as specified**, so its failure is partly a
statement about the threshold and not only about the data. Both readings are on
the table and the design does not separate them.

## What can be said

- **For rousettus there is some localised dependence.** `periphery` costs +0.083
  and keeping only the discs costs +0.101 more than removing them, both in the
  direction of "the region outside the discs matters". Neither survives correction
  (q = 0.944), so this is a direction, not a result.
- **For mauritius no localised ablation degrades recognition** — not the discs,
  not a matched area, not 76% of the image. Combined with the six-number result,
  the most economical reading is that mauritius identity on `green` is carried by a
  global property of the face rather than by any region of it.
- **The periphery finding from the saliency analysis is neither confirmed nor
  refuted.** The ablation cannot test it, because the premise that the cue is
  localised does not hold for one of the two species.

## What this does to the project's headline

`docs/background_leakage.md` moved the headline to `green` because `original` is
substantially solvable from background colour. This result says `green` has a
related weakness of its own: ~0.78 of ~0.81 is reachable from six global colour
numbers of the face. That is not a reason to abandon `green` — the models do beat
the descriptor, and the descriptor may be measuring something biologically real —
but any claim of the form "the model recognises facial features" now needs to
clear the six-number baseline first, and none of the numbers in this project
currently do.

**Recommended next measurement, and it is cheap.** Score the existing green
checkpoints against the six-number descriptor per fold, paired, exactly as
`scripts/probe_background_leakage.py` does for the background. That gives a
per-fold margin of "model over global colour" with a confidence interval, and it
is the number that says whether the models have learned anything spatial at all.
