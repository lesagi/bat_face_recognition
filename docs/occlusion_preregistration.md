# Pre-registration — does recognition actually depend on the periphery?

**Written 2026-08-26, before the ablated datasets exist.** `git log` is the audit
trail. Nothing below may be revised after the results are seen.

## Why

The pre-registered saliency test (`docs/saliency_preregistration.md`,
`docs/saliency_species.md`) produced one effect that replicated: in 38 of 40
models, across both species and both backgrounds, bat-specific training moves
attribution **away** from the eye and nose discs and **toward** the periphery.

That is a claim about where a model looks. It is not yet a claim about what the
model needs. Attribution maps can be wrong about causal dependence — which is the
whole reason this project stopped trusting them unaccompanied. So the claim gets
converted into an ablation: destroy the pixels and measure what performance
actually loses.

## The problem this design has to solve: area

The eye and nose discs together cover about **24%** of the crop; the periphery is
the remaining **76%**. Comparing "remove the discs" against "remove the
periphery" therefore mostly measures how many pixels were destroyed, and would
find the periphery more important no matter what the model does.

So the primary comparison is **area-matched**: the eye+nose discs against three
discs of *identical radius and count* displaced into the periphery. Same area,
same shape, same number of regions, same total boundary length — only the
location differs.

## Arms

All on the **`green`** background, ArcFace, 20 folds, both species.

| arm | what is destroyed | approx. area |
|---|---|---|
| `occ_none` | nothing — the matched baseline | 0% |
| `occ_centre` | the eye and nose discs | ~24% |
| `occ_matched` | three discs of the same radius, rotated 180° about the crop centre into the periphery | ~24% |
| `occ_periphery` | everything except the eye and nose discs | ~76% |

`green` only. `original` is excluded by design: arm 3C showed a model trained on
`original` images with the face deleted still reaches ROC-AUC 0.831 / 0.780, so an
occlusion test there would partly measure the background, and any result would be
uninterpretable.

`occ_none` is rebuilt through the same pipeline rather than reusing the published
green arm, because images whose pose keypoints are too weak to place an ROI are
dropped **from every arm**, and the baseline has to sit on that same reduced image
set. Comparing against the published arm would confound the ablation with a
different set of images.

Destroyed regions are filled with the **mean colour of the clean crop** — a
per-image constant, so it carries no spatial information, and the identical rule
in every arm, so the arms differ only in *which* pixels went.

## Hypotheses

- **H3 (primary)** — removing the eye and nose discs costs **no more** ROC-AUC
  than removing an area-matched periphery region. That is what the saliency result
  predicts. Formally: the per-fold drop from `occ_none` is *equivalent* between
  `occ_centre` and `occ_matched`.
- **H4** — the eye and nose discs alone are **not sufficient**: `occ_periphery`
  (which keeps only those 24%) loses substantially more than `occ_centre`.

H3 is a claim of *no difference*, so an ordinary significance test cannot support
it — a non-significant result would only mean underpowered. It is therefore
tested by **equivalence (TOST)**, the same tool and the same margin this project
used for the quality-parity work: **|Hedges' g| < 0.5**, declared in advance,
meaning "differences below half a between-fold standard deviation do not matter".
`bat_stats.tost` already implements it.

Three outcomes are possible and all are reportable:

| TOST | difference test | reading |
|---|---|---|
| equivalent | not significant | **H3 supported** — the central features are not special |
| not equivalent | significant | **H3 refuted** — the discs matter more than matched periphery, and the saliency finding does not translate into dependence |
| not equivalent | not significant | **inconclusive** — underpowered, stated as such |

## Statistical test

- **Within species, paired by fold.** Same manifest identities and the same fold
  seed, so fold *N* of two arms holds out the same bats. Nadeau–Bengio corrected
  t-test on the per-fold differences (`bat_stats.corrected_resampled_ttest`), which
  is required because folds share training data.
- **Effect size and bootstrap CI reported with every p-value**, never a bare p.
- **The power ceiling applies.** `outputs/kfold_power_ceiling.json` puts the
  minimum detectable difference for this design at 0.440 ROC-AUC. Differences
  smaller than that cannot be resolved, which is exactly why H3 is framed as
  equivalence rather than as a failure to reject.
- **Correction**: Benjamini–Hochberg across the comparisons reported (3 ablation
  arms × 2 species = 6).

## Positive control, and why this one is not optional

`occ_periphery` destroys 76% of every image. **It must lose a lot.** If it does
not, the ablation is not removing usable information — the fill is being ignored,
or the model is reading something the masks do not touch — and in that case
nothing may be claimed from any arm, including H3. An equivalence result between
two ablations that both do nothing is meaningless.

Quantitatively: `occ_periphery` must drop the per-fold median ROC-AUC by at least
**0.05** below `occ_none` in both species. Declared now.

## Area check, declared now

`occ_centre` and `occ_matched` must destroy areas within **2 percentage points**
of each other, measured per image and reported as a distribution. If they do not,
the primary comparison is void and is not reported. The displaced discs must also
overlap the true eye/nose discs by no more than **10%** of their area, or they are
not testing the periphery; images failing that are dropped from every arm.

## Declared failure conditions

- Positive control fails → nothing is claimed.
- Area match or overlap bound fails → the primary comparison is void.
- **H3 refuted** (`occ_centre` loses significantly more than `occ_matched`) → the
  saliency finding does not translate into causal dependence, and
  `docs/saliency_species.md` must say the attribution result overstated the
  periphery. This is a real possible outcome and is written down now so it cannot
  be reframed later.
- If pose failures drop more than 15% of either species' images, the arm is
  reported as covering a biased subset rather than the dataset.

## Known limitations, stated in advance

- **This measures dependence, not localisation.** A flat-filled disc still tells
  the network something is missing; the model may compensate rather than fail.
  Equal treatment across arms controls the comparison but not the absolute drop.
- **`green` retains the crop silhouette**, already flagged in the shape/texture
  controls. The periphery arm includes that outline, so "periphery" here means
  fur, ears and head outline together, not fur alone.
- **The ROI radii (0.16 × edge) are inherited unchanged** from the saliency
  analysis and are not tuned here in either direction.
- **The displaced discs may land on non-face background** on green, since the
  periphery includes the flat screen. That makes `occ_matched` a conservative
  control for H3: if anything it destroys *less* useful signal than a
  face-restricted control would, which biases against H3 rather than for it.

## Analysis code, fixed in advance

`scripts/build_roi_occlusion.py` (dataset build, area and overlap checks) and
`scripts/analyze_roi_occlusion.py` (paired tests, TOST, positive control), both
committed before the datasets are built. Output:
`outputs/occlusion/roi_occlusion_analysis.json`.

---

## Amendment 1 (2026-08-26, before any dataset was built)

**What changed:** the *construction* of `occ_matched`. **What did not change:** the
hypotheses, the metric, the test, the equivalence margin, the positive control,
the area and overlap bounds, or the failure conditions.

The declared construction — three discs of identical radius displaced by a 180°
rotation about the crop centre — **is infeasible on this data**, and that was
established by measurement before anything was built. The aligner centres the
crop on the face, so the eye/nose keypoint centroid sits essentially at the crop
centre (measured: x = 0.504, y = 0.503 of the edge for mauritius; 0.546, 0.589
for rousettus). Rotating about the centre therefore barely moves the discs:

| displacement rule | mean overlap with the true discs | images meeting the declared bounds |
|---|---|---|
| 180° rotation (as declared) | **36.4%** | **3.0%** |
| vertical reflection | 42.3% | 0.6% |

Both violate the declared overlap bound of ≤10% on nearly every image, so the
comparison as specified could not have been run.

**Replacement construction.** `occ_matched` destroys the set of pixels that is
(a) the same *count* as the eye+nose discs, (b) on the animal wherever possible,
and (c) furthest from the eye+nose discs. Concretely: rank every pixel by
distance from the eye/nose ROI, with on-animal pixels (from the green matte)
ranked ahead of background, exclude the ROI itself, and take the top *N* where
*N* is exactly the ROI's pixel count.

Measured on the same sample:

| property | value |
|---|---|
| area difference from `occ_centre` | **0.00000** — identical by construction |
| overlap with the eye/nose discs | **0.00000** |
| fraction of the destroyed region on the animal | mean **98.3%**, min 72.3% |
| images meeting both declared bounds | **100%** |

This is a *better* control than the one declared, not merely a feasible one. It
is area-exact rather than approximately matched, and it destroys a comparable
amount of *animal*: the eye+nose ROI is itself 98% (mauritius) / 90% (rousettus)
on the animal, so both arms remove bat rather than one removing bat and the other
removing flat green screen. That also removes the concern recorded under "Known
limitations" that the control might land on background and thereby destroy
nothing.

The cost is that the region is a contiguous band rather than three discs, so
shape and boundary length are no longer matched — only area, count of destroyed
pixels, and on-animal fraction are. That is stated here rather than discovered
later.

Pose detection succeeds on **100%** of images in both species, so the
"drop pose failures from every arm" provision has no effect and the
image-set-bias failure condition does not trigger.

---

## Amendment 2 (2026-08-26, after the first run — the positive control failed)

**The first run is void by the criterion declared above, and the cause was a
defect in the fill.** Recorded here rather than quietly re-run.

### What happened

| arm | mauritius median ROC-AUC | rousettus |
|---|---|---|
| `none` | 0.805 | 0.816 |
| `centre` | 0.845 | 0.835 |
| `matched` | 0.864 | 0.829 |
| `periphery` | 0.825 | 0.834 |

Destroying **76%** of every image cost **+0.004** (mauritius) and **+0.016**
(rousettus) against the declared minimum of 0.05. Every ablation scored at or
*above* the clean baseline. Positive control failed, so H3 and H4 were reported as
not interpretable.

### Why

The fill was "the mean colour of the clean crop" — chosen because it is a
per-image constant and therefore carries no *spatial* information. That reasoning
was incomplete: a per-image constant still carries **colour** information, and
colour is exactly the leak channel this project has been chasing since arm 3C.

Measured directly, with no model — pairs scored only by the distance between the
two images' mean colours:

| species | ROC-AUC from mean colour alone |
|---|---|
| mauritius | **0.613** |
| rousettus | **0.734** |

Against trained models reaching 0.81–0.86 on these arms. So the `periphery` arm
painted 76% of every image with a three-number summary that identifies the bat.
The positive control could not have passed: the more of the image the ablation
destroyed, the more of it was replaced by an identity cue.

Every arm is affected, not just `periphery` — `centre` and `matched` each paint
24% of the image with it.

### The fix

Fill with a **single global constant** — neutral grey (128, 128, 128), identical
for every image and both species. A global constant carries no per-image
information of any kind, colour included. Nothing else changes: the same masks,
the same arms, the same image sets, the same hypotheses, metric, test, margin,
positive control threshold and failure conditions.

The first run's numbers are retained in
`outputs/occlusion/roi_occlusion_analysis_meancolour_fill.json` as the record of a
failed control, not as a result.

### What this episode is

The positive control did its job: it caught a defect in my own build that would
otherwise have produced a confident, wrong answer about H3 — and note the
direction it would have produced. `matched` scored *highest* of all four arms,
which read naively says "destroying an area of periphery improves recognition".
Without the control that would have been reported as evidence for the periphery
finding. This is the fifth defect in this workstream caught by an internal
consistency check rather than by inspection.

---

## Amendment 3 (2026-08-26, after the grey-fill run) — outcome

The re-run with a global constant fill removed the colour leak (fill-colour
ROC-AUC exactly 0.500, down from 0.613/0.734). **The positive control still fails
for mauritius**: `periphery` costs −0.021, i.e. destroying 76% of every image
*improved* median ROC-AUC. rousettus passes (+0.083). The threshold was declared
as "both species", so H3 and H4 are reported as not interpretable.

Diagnosis, in `docs/occlusion_results.md`: six numbers — the per-channel mean and
standard deviation of the face region, no spatial information — identify a bat at
0.777 (mauritius) / 0.792 (rousettus), against trained models at 0.805 / 0.816 on
the same images. A global statistic survives the destruction of any particular
region, so the control could not have passed for a cue of this kind. Grey-world
white balancing does **not** remove it (0.777 → 0.778), so it is not simple
per-video illumination, and one video per bat leaves genuine individual colouration
and a per-video capture signature indistinguishable.

**A flaw in this pre-registration, recorded rather than glossed.** The 0.05
control threshold was set without checking it was detectable. The per-fold spread
in this design is ~0.12, so with 20 folds a 0.05 median shift sits at roughly 2
standard errors, and the mauritius `periphery` interval [−0.051, +0.119] contains
both zero and the threshold. The control was underpowered as specified. Its
failure is therefore partly a statement about the threshold rather than only about
the data, and this design does not separate the two.

No further re-run is proposed. A third attempt would be fitting the experiment to
the answer, and the diagnosis says the premise of the arm — that the cue is
localised — does not hold for one of the two species.
