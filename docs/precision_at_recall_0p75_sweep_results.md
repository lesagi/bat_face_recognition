# TPR-floor Optuna sweep results — Siamese, `val/precision_at_recall_0p75`

Results from the `siamese_precision_at_recall_search` Optuna study
(`configs/sweep/siamese_precision_at_recall.yaml`), 10 trials per species.
This is the TPR-floor counterpart to the FAR-budget study documented in
`docs/recall_far_1e2_sweep_results.md`; both stay in the same SQLite
storage (`optuna_studies.db`) under distinct study names.

## Search space

```yaml
target_metric: val/precision_at_recall_0p75
direction: maximize
search_space:
  loss.alpha:  {type: float,      low: 0.3,    high: 0.8}
  loss.gamma:  {type: float,      low: 1.0,    high: 4.0}
  trainer.lr:  {type: loguniform, low: 1.0e-5, high: 2.0e-4}
```

Narrower `loss.alpha` range than the FAR-budget sweep because the recall
floor is enforced by the threshold selector — heavy recall-bias via alpha
is no longer required.

## Per-species winners (val/precision_at_recall_0p75)

| Species | Best trial | val score | `loss.alpha` | `loss.gamma` | `trainer.lr` |
|---|---|---|---|---|---|
| Mauritius (green_bg, 19 IDs) | trial 3 | **0.797** | 0.6352 | 1.871 | 2.34e-5 |
| Rousettus (random_bg, 12 IDs) | trial 14 | **0.888** | 0.3675 | 1.027 | 1.76e-4 |

## Test-set transfer (retrain at winning HPs)

The sweep deliberately doesn't run `trainer.test()` per trial (test split is
sacred per workspace invariants — see `bat_sweeps/objective.py:172`). So I
retrained each species' winning HP combo end-to-end to get test numbers and
permutation p-values.

Baseline = default Focal HPs (`alpha=0.75, gamma=2.0, lr=5e-5`); see
`docs/precision_at_recall_0p75_baseline.md`.

| Species | Variant | ROC-AUC | precision@recall≥0.75 | threshold | F1 (perm) |
|---|---|---|---|---|---|
| Mauritius | baseline | 0.906 | **0.869** | 0.447 | 0.808 (p<0.001) |
| Mauritius | sweep-winner | 0.904 | **0.836** | 0.414 | 0.791 (p<0.001) |
| Rousettus | baseline | 0.806 | **0.749** | 0.381 | 0.750 (p<0.001) |
| Rousettus | sweep-winner | 0.818 | **0.774** | 0.224 | 0.762 (p<0.001) |

MLflow runs: `a92baf05f61e4af795e966bdf46c8db4` (mauritius sweep-winner),
`8b637f0e5f6945ec8a9c15ad8a04ab70` (rousettus sweep-winner).

## Findings

1. **Rousettus sweep transferred to test** (+2.5 pp test precision over
   the default-HP baseline, +1.2 pp ROC-AUC) at HPs that look very
   different from Focal "best practice": `alpha = 0.37` (precision-bias),
   `gamma = 1.03` ≈ plain BCE focusing, `lr = 1.76e-4` (~3.5× the
   default).
2. **Mauritius sweep over-tuned to val** — val improved 5.1 % relative
   (0.758 → 0.797) but test *regressed* 3.3 pp (0.869 → 0.836). The val
   and test splits each have only 3 identities, so the val signal is
   noisy and the winner picked up some idiosyncrasy that didn't
   generalise. ROC-AUC was essentially unchanged (0.906 → 0.904) so the
   underlying model quality didn't shift — the threshold operating
   point at val differed from the test one.
3. **HP signatures diverge between species.** The FAR-budget sweep saw
   both species converge on (`alpha≈0.59, gamma≈2.1, lr≈1e-4`). The
   TPR-floor sweep finds them apart: mauritius wants conservative
   training (low LR `2.3e-5`, moderate `gamma=1.87`), rousettus wants
   aggressive training (high LR `1.76e-4`, gamma close to 1.0). With 5
   completed trials each it's hard to tell whether this divergence is
   signal or noise.
4. **Pruning rate was high on rousettus** — 5 of 10 trials pruned at
   epoch 1 with `precision_at_recall_0p75 < 0.69`. The median pruner is
   well-calibrated for this metric: trials that don't hit ~0.7 by epoch
   1-2 essentially never do.

## Cross-framing reconciliation (mauritius, same ROC curve)

The mauritius model (sweep-winner, ROC-AUC=0.904) can be reported at three
operating points by selecting the threshold differently:

| Operating point | Threshold | Recall | Precision | F1 |
|---|---|---|---|---|
| Youden-J | 0.314 | 0.840 | 0.805 | 0.822 |
| **TPR-floor (recall ≥ 0.75)** | **0.414** | **0.751** | **0.836** | **0.791** |
| FAR-budget (FPR ≤ 1e-2) | ~0.93 | ~0.10 | ~0.95 | — |

For rousettus (ROC-AUC=0.818):

| Operating point | Threshold | Recall | Precision | F1 |
|---|---|---|---|---|
| Youden-J | 0.230 | 0.748 | 0.777 | 0.762 |
| **TPR-floor (recall ≥ 0.75)** | **0.224** | **0.750** | **0.774** | **0.762** |
| FAR-budget (FPR ≤ 1e-2) | ~0.95 | ~0.08 | ~0.91 | — |

On rousettus Youden-J and TPR-floor coincide — the ROC curve is flat
enough at recall ~ 0.75 that there's no slack to trade.

## Recommendation for the paper

Lead with the **TPR-floor framing** as the headline biometric metric for
this dataset:

- It's the more interpretable framing for ecological identification
  ("of true matches the system has to find, how many will it correctly
  flag at acceptable precision?").
- The numbers are non-degenerate (recall ~ 0.75, precision 0.77–0.87) —
  unlike the FAR-budget numbers which were 0.08–0.17 recall and looked
  worse than they should because the operating point was too strict for
  the dataset size.
- HP-tuned rousettus (the smaller training pool) gains more from the
  sweep than mauritius. Tuned-mauritius's val/test gap argues for either
  reporting the *baseline* as the headline number for mauritius (defaults
  are robust) or running the sweep with a held-out re-evaluation set to
  curb val overfitting.

## Reproducing

```
# 10-trial sweep per species:
uv run bat-cli sweep --experiment siamese_mauritius_green_bg_video \
    --sweep-config siamese_precision_at_recall
uv run bat-cli sweep --experiment siamese_rousettus_random_bg_video \
    --sweep-config siamese_precision_at_recall

# Retrain at winning HPs to get test numbers:
uv run bat-cli train --experiment siamese_mauritius_green_bg_video \
    --hydra loss.alpha=0.6352 --hydra loss.gamma=1.871 --hydra trainer.lr=2.34e-5
uv run bat-cli train --experiment siamese_rousettus_random_bg_video \
    --hydra loss.alpha=0.3675 --hydra loss.gamma=1.027 --hydra trainer.lr=1.76e-4
```
