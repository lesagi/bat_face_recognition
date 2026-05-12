# Refactor status — Phase 0–3 complete on `main`

Snapshot of where the PyTorch refactor stands. Phase 4 (migration +
cleanup) is the last piece; see `docs/PHASE_4_CHECKLIST.md` for the
remaining workstation-driven steps.

---

## What's on `main`

Tag `refactor-foundation-pytorch` was placed on the Phase 0 commit.
Phase 3 completed at `40eeb75` (`fix: validate cli promotion flags before
training`); later commits may add documentation or Phase 4 prep tooling.

13 workspace packages under `packages/`, all Python 3.11, all importing
only from `bat_core` upward through the dependency graph. No package
imports `app.*` or `legacy.*` (verified by `tools/check_tf_isolation.py`).

| Package | Role | Key public surface |
|---|---|---|
| `bat_core` | Types + Protocols, no I/O | `ImageRecord`, `Manifest`, `Embedding`, `Predictions`, `EvalReport`, `RunArtifacts`, `SaliencyImage`; `FaceModel`, `Loss`, `Trainer`, `InterpretabilityAdapter`, `Tracker` Protocols |
| `bat_data` | Manifest + dataset + 3-way splitter + miners | `build_manifest`, `manifest_to_csv` / `manifest_from_csv`, `IdentitySplitter`, `BatDataset`, `HardNegativeMiner`, `SemiHardMiner` |
| `bat_preprocessing` | YOLO seg/pose, alignment, transforms, video extraction | `PreprocessingPipeline`, `YOLOSegmenter`, `YOLOPose`, `FaceAligner`, `VideoExtractor`, prediction-cache primitives, background generator |
| `bat_models` | Siamese / ArcFace / AdaFace `nn.Module`s | `SiameseModel`, `ArcFaceModel`, `AdaFaceModel`, `resnet50_backbone`, `ArcFaceHead`, `AdaFaceHead`, `export_onnx` |
| `bat_losses` | Pair + embedding losses | `BCELoss`, `FocalLoss`, `TripletLoss`, `ArcFaceLoss`, `AdaFaceLoss`, `CosFaceLoss`, `SubCenterArcFaceLoss` (each declares `.family`) |
| `bat_training` | Loop-only trainers + EMA + AMP + checkpointing + Accelerate | `PairTrainer`, `EmbeddingTrainer`, `make_trainer`, `TrainerConfig`, `BestCheckpointTracker`, `OptunaPruningCallback` seam via tracker wrap |
| `bat_evaluation` | Verification + identification, no CSV output | `evaluate_predictions`, `optimize_youden_j`, `run_eval_protocol`, `split_gallery_probe` |
| `bat_interpretability` | Saliency / GradCAM / projection adapters | `SiameseSaliencyAdapter`, `GradCAMAdapter`, `EmbeddingProjectionAdapter`, `select_adapter`, `composite_from_saliency_images` (5 rows × 2 cols) |
| `bat_stats` | Permutation tests + visualizer | `run_inference_test`, `run_retrain_test`, `PermutationVisualizer`, `experiment_name`, `build_filename` (naming-bug fix) |
| `bat_tracking` | MLflow facade with sectioned metrics | `MLflowTracker`, `start_run`, `register_model`, `get_champion`, `would_promote`, `promote_to_champion`, `KEEP` / `DROP_PREFIXES` allowlist |
| `bat_reporting` | Unified post-training PDF + run-comparison HTML | `build_unified_pdf`, `compare_runs`, `ReportData`, 8-section reportlab pipeline |
| `bat_sweeps` | Optuna val-only HP search | `make_study`, `build_objective`, `run_sweep`, `OptunaPruningCallback`/`OptunaPruningTracker` |
| `bat_cli` | Single-entry interactive launcher | `bat-cli` script: `train` / `sweep` / `evaluate` / `permutation-test` / `compare` / `promote` / `build-manifest`; `runtime.py` orchestration helpers |

---

## Auto post-training pipeline (Phase 3)

`bat-cli train` runs the full sequence in one invocation. Each step is
`_safe`-wrapped: failure logs a warning, downstream steps still attempt.

1. `trainer.fit(train_loader, val_loader)` → `RunArtifacts`
2. Test eval → `EvalReport` via `bat_evaluation.protocols.run_eval_protocol`
3. Saliency / GradCAM / projection (cross-split sample: 2 identities each
   from train/val/test by default) → `<run>/explanations/`
4. Unified PDF report → `<run>/post_training_report.pdf`
5. MLflow artifact upload (PDF, explanations, permutation outputs)
6. Inference-mode permutation test → `<run>/permutation/`
7. Champion promotion: off by default; opt in with `--promote` (silent
   auto if `would_promote` returns `beats=True`) or `--prompt-promote`
   (asks once, only when `beats=True`)

Flags to skip individual steps: `--no-mlflow`, `--no-permutation`,
`--no-explanations`, `--explanations-per-split N`.

---

## What still differs from the original plan

These are intentional simplifications agreed during execution:

- **No CI**: single-programmer repo. Quality gates run via `pre-commit`
  locally, matching the locked plan.
- **No Dockerfile**: same reason; `uv sync --frozen` on the workstation
  is the entire setup, matching the locked plan.
- **Two YOLO `.pt` weights kept** under `models/preprocessing/`
  (segmentation + pose) because they drive the still-image extraction
  pipeline. Tracked legacy artifacts were pruned in Phase 0; ignored
  local `legacy/` leftovers, if present on a workstation, are not part of
  the PyTorch runtime.
- **`bat_stats` retrain-mode permutation trainer hooks** deliberately kept
  thin: `bat_training.permutation_adapter.create_pair_trainer_factory`
  is the DI seam (no upstream modification of `bat_stats`).

---

## Open follow-ups not blocking Phase 4

- `EmbeddingTrainer._validate` only emits `loss`/`accuracy`/`f1`/
  `precision`/`recall` per val epoch; the default sweep target `val/roc_auc`
  needs either a `bat_training` patch (compute it via
  `bat_evaluation.verification`) or a re-targeted `target_metric`.
- `bat_training` doesn't expose an explicit per-epoch callback list;
  `bat_sweeps.OptunaPruningCallback` is wired via the `Tracker` wrap
  workaround. A `TODO(phase-2.1)` marker in
  `bat_sweeps/pruning_callback.py` flags the swap point.
- `bat_reporting.compare_runs` reads metrics via vanilla `MlflowClient`;
  a "champion vs candidate" mode using `bat_tracking.get_champion` is
  flagged for follow-up.
- ONNX export prefers `torch.onnx.dynamo_export` with a legacy `torch.onnx.export`
  fallback. Phase 4 reproducibility check should confirm dynamo path
  succeeds end-to-end on a full ArcFace model.

---

## Test counts at last verification

| Package | Tests | Notes |
|---|---|---|
| `bat_core` | 15 | runs without torch |
| `bat_data` | 39 | torch-gated tests skip cleanly |
| `bat_preprocessing` | 30 (1 skipped) | ultralytics-gated smoke skipped without weights |
| `bat_models` | 18 | torch + torchvision + onnx-gated |
| `bat_losses` | 33 | torch-gated |
| `bat_evaluation` | 22 | numpy + sklearn + torch |
| `bat_tracking` | 23 | mocked MlflowClient — no server |
| `bat_interpretability` | 26 (2 skipped) | UMAP-gated skip |
| `bat_stats` | 40 | numpy + sklearn + matplotlib |
| `bat_training` | 34 | torch + accelerate-gated |
| `bat_reporting` | 19 | reportlab + jinja2 |
| `bat_sweeps` | 32 | optuna |
| `bat_cli` | 17 (post-stabilize) | CliRunner + mocked runtime |

Last full verification before this status snapshot: `371 passed, 1 skipped`.
Full workspace pytest is the quality gate referenced in
`docs/PHASE_4_CHECKLIST.md`.

---

## Open PR queue (transient — delete this section when all are merged)

These PRs were opened from the Claude Code UI and are awaiting merge.
A future session checking this file should `gh pr list` (or use the UI)
to see whether they've landed; once they're all on `main`, drop this
section in the next docs sweep.

| PR | Branch | What it lands |
|---|---|---|
| #23 | `feat/bat-training-callback-list` | First-class per-epoch `callbacks=[...]` list on both trainers; `bat_sweeps.OptunaPruningCallback` plugs in directly; `OptunaPruningTracker` kept as fallback. |
| #24 | `feat/compare-runs-champion-mode` | `bat_reporting.compare_to_champion(candidate_run_id, model_name, ...)` + `bat-cli compare-champion` subcommand. Raises `NoChampionError` when no Production-stage version is registered. |

Merged in the previous batch (kept here for one cycle for cross-reference):

| PR | What it landed |
|---|---|
| #20 | `chore/torch-cu121` — route `torch`/`torchvision` through the CUDA-12.1 index on Linux (workstation driver compatibility). |
| #21 | `refactor/remove-tf` — Phase 4 step 4: deletion of `app/`, `run.py`, `Makefile`, `scripts/`, `setup.py`. Legacy snapshot preserved at tag `refactor-foundation-pytorch`. |
| #22 | `feat/embedding-trainer-val-roc-auc` — `EmbeddingTrainer` emits `val/roc_auc` + TAR@FAR every val epoch (with the `cfg.val_verification` flag). Unblocks the default sweep target metric. |

User workflow: PRs are created from the Claude Code UI by the user
themselves; Claude pushes branches but does **not** call `gh pr create`
(or any other PR-creation tool). "Pull, merge, repeat" runs fast — most
PRs land within minutes of being opened.
