# TPR-floor baseline — Siamese, `precision_at_recall_0p75`

Single-shot baselines on the default hyperparameters
(`loss.alpha=0.75, loss.gamma=2.0, trainer.lr=5e-5`), using the new
default trainer monitor `val/precision_at_recall_0p75` and the test-eval
checkpoint `best_model_precision_at_recall_0p75.pt`. These numbers are the
"untuned" reference Step 7's sweep results will beat.

## Test-set headline (operating point: recall ≥ 0.75)

| Species | ROC-AUC | precision@recall≥0.75 | threshold@recall=0.75 | realised recall | F1 (perm) |
|---|---|---|---|---|---|
| mauritius (green_bg, 19 IDs) | **0.906** | **0.869** | 0.447 | 0.756 | 0.808 (p<0.001) |
| rousettus (random_bg, 12 IDs) | **0.806** | **0.749** | 0.381 | 0.752 | 0.750 (p<0.001) |

MLflow runs: `e00fed6ef3494535869f381717928072` (mauritius),
`343bc0c66c62410783b83469706ef0a2` (rousettus).

The permutation test was run at the recall-0.75 threshold (per
`evaluation.permutation_threshold: recall_0p75`). All four metrics in
both species have p<0.001 against an n=1000 null with null-std ~0.01 —
i.e., the result is genuine, not threshold artefact.

## Cross-framing comparison (same model, different operating point)

For mauritius — three views of the **same** ROC curve (ROC-AUC=0.906):

| Operating point | Threshold | Recall (TPR) | Precision | F1 |
|---|---|---|---|---|
| Youden-J (balanced) | 0.227 | 0.912 | 0.854 | 0.882 |
| **TPR-floor (recall ≥ 0.75)** | **0.447** | **0.756** | **0.869** | **0.808** |
| FAR-budget (FPR ≤ 1e-2) | 0.902 | 0.106 | (≈0.95) | — |

The same comparison for rousettus — ROC-AUC=0.806:

| Operating point | Threshold | Recall (TPR) | Precision | F1 |
|---|---|---|---|---|
| Youden-J (balanced) | 0.477 | 0.701 | 0.787 | 0.742 |
| **TPR-floor (recall ≥ 0.75)** | **0.381** | **0.752** | **0.749** | **0.750** |
| FAR-budget (FPR ≤ 1e-2) | 0.965 | 0.084 | (≈0.91) | — |

The TPR-floor framing lands close to Youden-J on **rousettus** (both around
recall≈0.75/precision≈0.75) because the underlying ROC curve is less
discriminative — there's not much room to trade. On **mauritius** the
ROC-AUC is high enough (0.906) that the TPR-floor framing genuinely gains
precision (0.869) over Youden-J (0.854) by accepting a small recall
reduction (0.912 → 0.756). The FAR-budget framing's near-perfect precision
costs ~85 percentage points of recall on mauritius and ~67 pp on rousettus
— useful only when the cost of a false match is catastrophic.

## Notes for Step 7

1. The default-HP baseline gives a real target to beat. The sweep should
   try to push **precision@recall≥0.75** above 0.87 on mauritius and
   above 0.75 on rousettus.
2. The recall-0.75 threshold is ~0.45 on mauritius and ~0.38 on rousettus
   — well within the score range, no precision/float32 trickery needed
   here (unlike the FAR-1e2 threshold which sat at ~0.9 for these runs).
3. `best_model_precision_at_recall_0p75.pt` lands on disk for both
   species, confirming the Step 3 plumbing works end-to-end (no silent
   "falling back to final model" warning in the run log).

## Reproducing

```
uv run bat-cli train --experiment siamese_mauritius_green_bg_video
uv run bat-cli train --experiment siamese_rousettus_random_bg_video
```

With:

- `configs/trainer/pair.yaml` monitoring `val/precision_at_recall_0p75`.
- `configs/evaluation/verification.yaml` defaulting `test_checkpoint:
  precision_at_recall_0p75` and `permutation_threshold: recall_0p75`.
- Default Focal loss (`alpha=0.75, gamma=2.0`) and default Adam LR (`5e-5`).
