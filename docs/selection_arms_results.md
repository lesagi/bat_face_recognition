# Choosing mauritius bats to match rousettus — results after the completed cull

## What was tried

Each bat was filmed in exactly one video, so anything constant within a clip is a
free identity cue. `docs/occlusion_results.md` measures it: six global colour numbers
score 0.78 against trained models at 0.81. The idea tested here is to attack that by
**choosing** mauritius bats whose capture conditions resemble the rousettus set,
rather than degrading them afterwards.

Arms, all ArcFace / green / 224 px head crop, 10 folds each:

| arm | bats | how chosen |
|---|---|---|
| `paired` | 12 | Hungarian 1:1 match to each rousettus bat on face-vs-background contrast |
| `lit12` | 12 | 12 drawn at random from `lit` — the control for identity count |
| `lit` | 22 | every bat inside the rousettus contrast range (−7.5 … +31.9) |
| `all` | 39 | every curated bat, uncontrolled |
| `pairres` | 12 | `paired`, re-cropped to each partner's native resolution |
| `pairblur` | 12 | `paired`, blurred to each partner's gradient energy |
| `pairboth` | 12 | both |

Bats with fewer than 10 kept frames are excluded: a 3-frame bat contributes almost
no same-identity pairs and lands in a 2-identity test split often enough to add
variance for no information. That drops 3–4 bats depending on the arm.

## Results — 70 runs

| arm | bats | ROC-AUC | 95% CI | colour baseline | margin | folds won |
|---|---|---|---|---|---|---|
| `paired` | 12 | 0.873 | [0.746, 0.986] | 0.899 | −0.025 | 5/10 |
| `lit12` | 12 | 0.796 | [0.651, 0.964] | 0.905 | −0.095 | 3/10 |
| `lit` | 22 | 0.740 | [0.566, 0.909] | 0.870 | −0.045 | 5/10 |
| `all` | 39 | 0.785 | [0.739, 0.850] | 0.825 | −0.055 | 2/10 |
| `pairres` | 12 | 0.860 | [0.768, 0.907] | 0.914 | −0.069 | 3/10 |
| **`pairblur`** | 12 | **0.919** | [0.867, 0.958] | 0.924 | **+0.033** | 6/10 |
| `pairboth` | 12 | 0.843 | [0.782, 0.953] | 0.924 | −0.025 | 5/10 |

Margin is model minus the colour baseline, paired within each fold. **No arm's
margin excludes zero.** `pairblur` is the only one positive at all, and the only one
winning a majority of folds.

## The headline from the earlier run did not survive

On the partially-reviewed cull, `paired` showed colour leakage of 0.815 against
rousettus's 0.816 — a clean "the pairing brings mauritius to the rousettus level".
After the cull was finished by hand, the same arm measures **0.886**, and every
12-bat arm now sits at 0.90–0.92.

The earlier number was computed over a frame set in which several paired bats still
carried keeps inherited from the legacy cull. Completing the review replaced those
with hand-selected frames — frontal, full-face, round mask — and that selection is
*more* colour-separable, not less. Selecting good frames and selecting
colour-neutral frames pull in opposite directions.

So selection on lighting does not reduce the colour cue once the frames themselves
are cleanly chosen. It is not an available route to removing this confound.

## What does hold

**Recognition is not a sharpness artifact.** `pairblur` is matched to each partner's
gradient energy (ratio 1.02, 11/12 within 1.25×) and scores 0.919 — the best of
every arm, above the undegraded 0.873. Removing the resolution advantage does not
remove the result.

**More bats did not help.** `all` (39 bats) scores 0.785 and wins 2/10 folds; `lit`
(22) scores 0.740. Both are worse than the 12-bat `paired`. With identity-disjoint
folds, adding loosely-matched identities adds heterogeneity faster than it adds
signal.

## What is still blocked

Every margin CI spans zero, and this is not fixable by running more folds. The
design's minimum detectable difference is 0.44 ROC-AUC at 16 bats
(`scripts/analyze_power_ceiling.py`), driven by the test/train identity ratio. At 12
bats a 0.03–0.09 effect is far below the floor. Only more individuals change this.

The one encouraging number — `pairblur` at +0.033 over the colour baseline, 6/10
folds — is exactly the size this design cannot resolve.
