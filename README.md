# Bat Face Recognition

Open-set face recognition for individual fruit bats (rousettus + mauritius), built on PyTorch.

The project trains and evaluates three model families against a manifest-driven dataset:

- **Siamese** (pair similarity, BCE / Focal / Triplet)
- **ArcFace** (margin-based softmax, Deng et al. 2019)
- **AdaFace** (adaptive margin, Kim et al. 2022)

A single CLI (`bat-cli`) wires composition, training, evaluation, interpretability, the unified PDF report, MLflow artifact upload, an inference-mode permutation test, and optional champion promotion into one invocation.

## Quick start

```bash
# Sync the workspace (one-time; pulls all 13 packages + deps).
uv sync --frozen
uv run pre-commit install

# Build a manifest from your processed images.
uv run bat-cli build-manifest \
  --input-dir data/processed/rousettus \
  --output data/manifests/rousettus_manifest.csv \
  --species rousettus

# Run training end-to-end. With no flags, training runs silently (no
# MLflow promotion). Add --prompt-promote to be asked, or --promote
# to auto-promote when criterion beats the current champion.
uv run bat-cli train --experiment arcface_rousettus_random_bg_video

# Or launch the interactive picker.
uv run bat-cli
```

## Repo layout

```
.
├── packages/                # uv workspace — 13 single-purpose packages
│   ├── bat_core/            # Pydantic types + Protocols (no I/O)
│   ├── bat_data/            # Manifest, 3-way splitter, dataset, miners
│   ├── bat_preprocessing/   # YOLO seg/pose, alignment, video extraction
│   ├── bat_models/          # Siamese / ArcFace / AdaFace nn.Modules
│   ├── bat_losses/          # BCE / Focal / Triplet + ArcFace / AdaFace / CosFace / SubCenter
│   ├── bat_training/        # PairTrainer + EmbeddingTrainer (loop only)
│   ├── bat_evaluation/      # Verification + identification, no CSV
│   ├── bat_interpretability/ # SiameseSaliency / GradCAM / projection adapters
│   ├── bat_stats/           # Permutation tests + experiment-aware visualizer
│   ├── bat_tracking/        # MLflow facade with sectioned metrics
│   ├── bat_reporting/       # Unified post-training PDF + run-comparison HTML
│   ├── bat_sweeps/          # Optuna val-only HP search
│   └── bat_cli/             # Single-entry interactive launcher
├── configs/                 # Hydra config root (data/model/loss/trainer/...)
├── docs/
│   ├── PHASE_4_CHECKLIST.md # Workstation-driven steps to finish migration
│   └── REFACTOR_STATUS.md   # Per-package status + open follow-ups
├── tools/
│   └── check_tf_isolation.py # Static check before deleting legacy TF code
├── models/
│   └── preprocessing/       # YOLO seg + pose weights (kept; runtime deps)
├── pyproject.toml           # uv workspace root, shared dev deps
└── uv.lock
```

The legacy TF/Keras surfaces (`app/`, `run.py`, `Makefile`, `scripts/`, `setup.py`) were removed during Phase 4 step 4. The pre-refactor snapshot is preserved at the `refactor-foundation-pytorch` git tag; recover any individual file with `git show refactor-foundation-pytorch:<path>`.

## Common workflows

```bash
# Train, with the auto post-training pipeline.
bat-cli train --experiment arcface_rousettus_random_bg_video

# Skip the heavy steps for a fast turnaround run.
bat-cli train --experiment siamese_rousettus_random_bg_video \
  --no-explanations --no-permutation --no-mlflow

# Override any Hydra value inline.
bat-cli train --hydra trainer.epochs=5 trainer.lr=1e-4

# Optuna sweep on the val split (test stays sacred).
bat-cli sweep --experiment arcface_rousettus_random_bg_video --n-jobs 1

# Re-run just evaluation against a checkpoint.
bat-cli evaluate --experiment arcface_rousettus_random_bg_video \
  --checkpoint outputs/runs/<dir>/checkpoints/best_roc_auc.pt

# Inference-mode permutation test on a configured run.
bat-cli permutation-test --experiment arcface_rousettus_random_bg_video \
  --n-permutations 1000

# Side-by-side MLflow run comparison HTML.
bat-cli compare <run-id-a> <run-id-b> --output outputs/compare/report.html

# Promote a registered model if criterion beats incumbent.
bat-cli promote <run-id> --criterion test/roc_auc
```

`bat-cli` with no arguments launches an InquirerPy walkthrough that exercises every action above.

## Auto post-training pipeline

`bat-cli train` runs the full sequence in one invocation. Each step is wrapped so a failure logs a warning rather than aborting downstream work.

1. `trainer.fit(train_loader, val_loader)` → `bat_core.RunArtifacts`
2. Test eval (verification + identification) → `bat_core.EvalReport`
3. Saliency / GradCAM / t-SNE / UMAP — samples 2 identities each from train/val/test by default → `<run>/explanations/`
4. Unified PDF report (8 sections) → `<run>/post_training_report.pdf`
5. MLflow artifact upload (PDF, explanations, permutation outputs)
6. Inference-mode permutation test → `<run>/permutation/`
7. Champion promotion gate — off by default; `--promote` for silent auto, `--prompt-promote` to ask once

Step-skip flags: `--no-mlflow`, `--no-explanations`, `--no-permutation`, `--explanations-per-split N`.

## Hydra config

Top-level `configs/config.yaml` composes one entry from each group. Override on the CLI with `--hydra key=value`:

```bash
bat-cli train \
  --hydra model=arcface loss=arcface trainer.epochs=40 data=manifest_rousettus
```

Composed experiments under `configs/experiment/` bundle whole stacks — e.g. `arcface_rousettus_random_bg_video.yaml` pins `model=arcface`, `loss=arcface`, `trainer=embedding`, `data=manifest_rousettus`. Add new ones by copying an existing file.

## Tests

```bash
# Full workspace.
uv run pytest packages/

# A single package.
uv run pytest packages/bat_core/

# Linters + formatters (the only quality gate; no CI).
uv run pre-commit run --all-files
```

At last full verification: 371 passed, 1 skipped across all 13 packages.

## Refactor status

Phase 0–3 of the PyTorch refactor are merged on `main`. Phase 4 (data manifest build, parity vs legacy TF, ArcFace/AdaFace baselines, TF deletion, version tag) is the remaining work and runs on the workstation.

- `docs/REFACTOR_STATUS.md` — per-package public surface, open follow-ups, test counts.
- `docs/PHASE_4_CHECKLIST.md` — exact commands and acceptance criteria for the remaining steps.

The Phase 0 commit is tagged `refactor-foundation-pytorch`.

## License

MIT — see `LICENSE`.
