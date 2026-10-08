# Why does `recrop` lose so much? — it probably doesn't

The paired-design runs showed `pairres` (per-pair re-crop) at median ROC-AUC 0.815
against the undegraded `paired` arm's 0.890, while `pairblur` — matched to the same
sharpness — scored 0.893. That looked like re-cropping destroying something blur
does not. Four checks, and the effect does not survive them.

## 1. Framing is identical, so it is not alignment drift

Re-cropping re-runs the aligner on the downsampled frame, so a shift in crop centre
or extent was the obvious suspect. Measured in the output crops:

| arm | interocular px | eye-line y | face area fraction |
|---|---|---|---|
| paired | 81.9 ± 8.7 | 85.3 ± 18.1 | 0.603 ± 0.073 |
| pairres | 81.9 ± 9.0 | 85.2 ± 18.1 | 0.603 ± 0.074 |
| pairblur | 81.7 ± 8.7 | 85.6 ± 18.0 | 0.606 ± 0.073 |
| pairboth | 82.0 ± 8.7 | 85.4 ± 18.0 | 0.606 ± 0.074 |

Indistinguishable. Running segmentation and pose at full resolution and scaling the
keypoints down — as both builders do — preserves the geometry exactly, which is what
that design choice was for.

## 2. The spectrum says `recrop` destroys *less*, not more

Radial power spectrum, energy per band, relative to rousettus = 1.00:

| arm | low (0–14) | mid (14–35) | high (35–70) | v. high (70–110) |
|---|---|---|---|---|
| paired | 0.79 | 1.39 | 1.95 | **8.10** |
| pairres | 0.79 | 1.31 | 1.43 | **3.60** |
| pairblur | 0.78 | 1.17 | 0.77 | **0.90** |
| pairboth | 0.78 | 1.15 | 0.73 | 0.77 |

`pairblur` lands closest to rousettus in every band; `pairres` keeps 3.6× the
very-high-frequency energy. So the arm that *retains more detail* scored lower. Any
explanation of the form "re-cropping removes information the model needs" is
inconsistent with this, and gradient energy — on which the two read 1.83 and 1.02 —
is not measuring the thing that matters here.

## 3. No dose-response

If re-cropping were causing the loss, folds whose held-out bats were shrunk most
should lose most. Shrink factors range 1.53–2.42×, so there is real variation to
work with.

**Spearman ρ = +0.309, p = 0.385 (n = 10 folds)** — the wrong sign and not
significant. The two folds that *gained* (+0.067, +0.055) had the two highest mean
shrink factors (2.10, 2.11).

## 4. The effect is within fold-to-fold noise

Paired per fold against `paired`:

| arm | median Δ | 95% CI | Wilcoxon p | folds worse |
|---|---|---|---|---|
| pairres | −0.080 | [−0.093, +0.013] | 0.037 | 8/10 |
| pairblur | +0.003 | [−0.081, +0.089] | 0.846 | 4/10 |
| pairboth | −0.020 | [−0.183, +0.041] | 0.322 | 5/10 |

8/10 folds in the same direction is suggestive, and uncorrected p = 0.037 would pass
a naive threshold. But across the three comparisons made here, Benjamini–Hochberg
puts it at 0.111. And the undegraded arm's own per-fold sd is **0.090** — larger than
the 0.080 effect being claimed.

## Conclusion

The honest reading is that `pairres` does not demonstrably lose anything. The
direction is consistent enough to be worth remembering, but there is no dose-response,
no mechanism visible in the spectrum, and the magnitude sits inside the design's
noise. With 12 identities and a recorded minimum detectable difference of 0.44
ROC-AUC, this design cannot resolve a 0.08 effect, and no number of folds changes
that — only more individuals would.

What *is* supported, and matters more: `pairblur` is matched to the rousettus
spectrum in every band and scores 0.893 against the undegraded 0.890. The paired
arm's recognition result is not a sharpness or resolution artifact.
