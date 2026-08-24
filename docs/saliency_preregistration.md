# Pre-registration — does training move attention onto mauritius eyes and noses?

**Written 2026-08-25, before the data it tests exists.** The green-background
checkpoints this document specifies have not been trained at the time of
writing; `git log` is the audit trail. Nothing below may be revised after the
results are seen.

## Why this document exists

The hypothesis under test — that mauritius individuals are distinguished by eye
and nose features — is a good one, and the existing original-background analysis
is consistent with it. But that analysis was **exploratory**, and one of its two
supporting numbers was found by reading `outputs/saliency/roi_stats_ig.json`
after the fact, looking for a signal. Declaring a finding after seeing it is the
same defect Phase D already corrected once in this project (a `min()` taken over
five correlated metrics, i.e. best-of-five cherry-picking).

So the exploratory result is treated as what generated the hypothesis, and the
hypothesis is tested on data that does not yet exist.

## The exploratory result that motivates this

Original background, ArcFace, IG at 320 px, 112 images / 28 identities, single
seed. Per-identity mean density (attribution share ÷ area share; 1.0 = the region
gets exactly the attention its size entitles it to).

| region | arm | mauritius | rousettus | species Cliff's δ | Hedges' g |
|---|---|---|---|---|---|
| eyes | untrained | 1.771 | 1.322 | +0.938 | 2.36 |
| eyes | trained | 1.669 | 0.911 | +0.979 | 3.57 |
| **nose** | **untrained** | **1.281** | **0.940** | **+0.604** | **1.17** |
| **nose** | **trained** | **1.143** | **0.731** | **+0.854** | **2.01** |

The relevant quantity is not the trained column — a randomisation control exists
precisely because saliency maps look persuasive for networks that have learned
nothing. It is the **change from untrained to trained**. On that reading:

- the **eyes'** species gap is essentially static (+0.938 → +0.979), i.e. mostly a
  property of the images rather than of learning;
- the **nose's** species gap **grows** (+0.604 → +0.854, g 1.17 → 2.01), i.e. it is
  the region where training actually differentiates the two species.

That inversion — the nose carrying the learning signal the eyes do not — is the
finding this document commits to testing. It is currently unreported;
`docs/saliency_species.md` does not mention the nose at all.

## Hypotheses

- **H1** — the trained-minus-untrained change in **eye** density is greater for
  mauritius than for rousettus.
- **H2** — the trained-minus-untrained change in **nose** density is greater for
  mauritius than for rousettus.

**H2 is primary.** Declared in advance and on the stated grounds above: the nose
is where the species gap grows under training while the eyes' does not. If only
one of the two survives correction, H2 is the one the conclusion rests on.

Both are **directional** (mauritius > rousettus), so one-sided tests are
appropriate; two-sided p-values will also be reported so nothing is hidden by the
choice.

## Design

| | |
|---|---|
| Background | **`green`** — flat screen behind every face |
| Model | ArcFace, stock ResNet-50 backbone, `input_edge_length=320` |
| Method | SmoothGrad-IG (`--method ig`), 12 noise draws × 21 path steps |
| Models per cell | **10** (folds 0–9), each with its own matched control |
| Images | `--max-per-identity 4`, matching the exploratory run exactly |
| Comparator | the same design re-run on `original`, so the arms differ only in background |

`green` is the required background because arm 3C established that a model
trained on `original` images with the face **deleted** still reaches ROC-AUC
0.831 (mauritius) / 0.780 (rousettus) — 93% and 91% of the full model. A model
scoring pairs partly on the scene may distribute attention quite differently from
one forced onto the face, so the exploratory result may not describe a
face-driven model at all. On `green` the background is identical across every
image and cannot carry identity.

## Metric

**Per-identity mean `*_density`.** Fixed in advance, single metric, no
alternatives.

Explicitly *not* the pointing game (which region contains the single hottest
pixel). Three reasons, all known before running: it is computed per image while
density is computed per identity; it reduces a whole map to one pixel; and two
runs of the identical command on the same data differ by 6 percentage points on
it (76.6% vs 82.8%), against a contrast of interest of ~19 points. The pointing
game will be reported as a secondary descriptive, never as evidence.

## Statistical test

1. **Within species**, per region: paired Wilcoxon signed-rank on per-identity
   (trained − untrained) differences. Paired because the same identities, images
   and ROIs are scored under both arms; pairing is done on `(identity, path)`,
   not on position, because pose detection runs separately per arm and could drop
   different images.
2. **Across species**, per region: Mann-Whitney U with Cliff's δ and Hedges' g on
   those per-identity contrasts. Unpaired — the two species are different animals.
3. **Across models**: the whole procedure is repeated for each of the 10 folds,
   and the reported interval is a bootstrap CI **across models**, not across
   identities. Single-model identity-level intervals understate the uncertainty
   that matters, which is whether the pattern replicates in a differently trained
   network.
4. **Correction**: Benjamini–Hochberg across **3 regions × 2 species = 6** tests.
5. Effect size and CI are reported alongside every p-value, and the p-value is
   never reported alone.

## Declared negative control

**`periphery`.** The three regions partition the frame exactly, so the
area-weighted mean of the three densities is 1.0 by construction: eyes and nose
cannot rise without periphery falling. The control therefore is *not* "periphery
shows nothing" — it is that periphery must move **opposite** to eyes and nose and
by a magnitude consistent with the area weighting. A periphery effect in the
*same* direction, or one materially larger than the partition arithmetic allows,
means the measurement is broken and both hypotheses are void regardless of their
own p-values.

## Declared failure conditions

Written now so they cannot be renegotiated later.

- If **H2 does not replicate on `green`** after correction, the nose result is
  reported as an original-background artifact, most plausibly background
  confounding, and the hypothesis is **not supported**. It will not be rescued by
  switching to the pointing game, to raw mass, to a different image budget, or by
  dropping the correction.
- If **neither H1 nor H2 replicates**, the whole eye/nose account is reported as
  unsupported, and `docs/saliency_species.md` says so.
- If the **periphery control fails**, nothing is claimed from this analysis at all.
- If the two arms' identity sets cannot be paired 1:1, the run is discarded rather
  than analysed unpaired.

## Known limitations, stated in advance

- **The control is ImageNet-pretrained, not random.** `build_model` honours
  `pretrained: imagenet`, so only the projection and head are freshly
  initialised. The contrast therefore isolates **bat-specific training**, not
  learning from scratch. This is arguably a stronger Adebayo-style control, but it
  is a different quantity from what `docs/saliency_species.md` currently describes
  and it is what will be reported.
- **The ROI radii (0.16 × edge) are a judgement call**, inherited unchanged from
  the exploratory analysis rather than calibrated. They are not tuned here, in
  either direction.
- **The power ceiling applies.** This design cannot resolve arbitrarily small
  effects, and a non-significant result means undetectable, not absent.
- **Resolution remains a candidate explanation** for any species difference in
  attention. Phase 3 showed resolution does not explain the *recognition* gap, but
  attention and performance are different measurements and that result does not
  transfer automatically.

## Analysis code, fixed in advance

`scripts/species_saliency_maps.py` (pinned checkpoints, `--background`,
serialised `per_image`) and `scripts/analyze_saliency_contrast.py`, both written
and committed before the green checkpoints are trained. Output:
`outputs/saliency/contrast_analysis.json`.
