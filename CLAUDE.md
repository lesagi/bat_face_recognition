# CLAUDE.md

Guidance for Claude Code sessions working in this repository. Read once on start, then refer back as needed.

## Project overview

Open-set bat face recognition. Three model families (Siamese / ArcFace / AdaFace) over a manifest-driven dataset (rousettus + mauritius species). Single CLI (`bat-cli`) drives composition → training → evaluation → interpretability → unified PDF → MLflow → permutation test → optional champion promotion.

The repo went through a multi-phase TF→PyTorch refactor. Phase 0–3 are merged on `main`. Phase 4 (parity runs + TF code deletion) is partially user-driven.

## Workspace layout

13 single-purpose packages under `packages/`, all Python 3.11, all members of one `uv` workspace.

```
packages/
  bat_core/            # types + Protocols (no I/O — never imports torch at module load)
  bat_data/            # manifest, identity-disjoint splitter, dataset, miners
  bat_preprocessing/   # YOLO seg/pose, alignment, video extraction
  bat_models/          # Siamese / ArcFace / AdaFace nn.Modules
  bat_losses/          # BCE / Focal / Triplet (pair) + ArcFace/AdaFace/CosFace/SubCenter (embedding)
  bat_training/        # PairTrainer + EmbeddingTrainer (loop only — no eval/MLflow)
  bat_evaluation/      # verification + identification, dataclass-only output
  bat_interpretability/ # SiameseSaliency / GradCAM / EmbeddingProjection adapters
  bat_stats/           # permutation tests + experiment-aware visualizer
  bat_tracking/        # MLflow facade with sectioned metrics + HP allowlist
  bat_reporting/       # unified post-training PDF + run-comparison HTML
  bat_sweeps/          # Optuna val-only HP search
  bat_cli/             # single-entry interactive launcher
```

`configs/` holds the Hydra root + groups. `models/preprocessing/` holds the two YOLO weights still in active use (face_seg.pt, face_pose.pt). `docs/` holds `REFACTOR_STATUS.md` (per-package state) and `PHASE_4_CHECKLIST.md` (workstation steps).

## Key invariants

- **No upward dependencies.** A package may only import from `bat_core` or packages strictly lower in the dep graph (see the table in `docs/REFACTOR_STATUS.md`). `bat_core` imports nothing from the workspace.
- **`bat_core` stays import-light.** It type-annotates `torch.Tensor` via `TYPE_CHECKING`; runtime never requires torch at import time.
- **Trainers are loops, not god-classes.** Data wiring → `bat_data`. MLflow → `bat_tracking`. Eval → `bat_evaluation`. Loss math → `bat_losses`. The trainer keeps only optimizer / scheduler / AMP / EMA / checkpointing / callbacks.
- **Test split is sacred.** `bat_sweeps` only reads val metrics; the objective never calls `trainer.test`. Verified with a mock test.
- **Naming bug fix is load-bearing.** Every plot title / filename / axis label routes through `bat_stats.naming.experiment_name(cfg)` so the species/source/background/model/loss combination is always visible. Don't bypass this helper.
- **No CSV output from new code.** The legacy TF stack emitted CSVs everywhere; the PyTorch eval/report path returns dataclasses and renders directly to the PDF.
- **Hydra cfg → caller responsibility.** `bat_training` does not log Hydra cfg itself; the CLI does (avoids leaking Hydra into trainer internals).

## Common operations

```bash
# Sync workspace (the only setup step).
uv sync --frozen

# Run the full workspace test suite.
uv run pytest packages/

# Lint + type-check (only quality gate; no CI).
uv run pre-commit run --all-files

# Static check before deleting legacy TF surfaces.
uv run python tools/check_tf_isolation.py

# Train with the auto post-training pipeline.
uv run bat-cli train --experiment arcface_rousettus_random_bg_video

# Interactive walkthrough (every action has a picker).
uv run bat-cli
```

## CLI surface (`bat-cli`)

| Command | Purpose |
|---|---|
| `train` | Compose cfg → train → test eval → explanations → PDF → MLflow → permutation → optional promote |
| `sweep` | Optuna search on val split |
| `evaluate` | Re-run test eval against a checkpoint |
| `permutation-test` | Inference-mode permutation test on a configured run |
| `compare <a> <b>` | Side-by-side MLflow run comparison HTML |
| `promote <run>` | Promote registered model if criterion beats incumbent |
| `build-manifest` | Walk a directory tree → split + write manifest CSV |

Step-skip flags on `train`: `--no-mlflow`, `--no-explanations`, `--no-permutation`, `--explanations-per-split N`.
Promotion: `--promote` (silent auto-on-improve) and `--prompt-promote` (ask once if improves) are mutually exclusive; default is no promotion.

## Where to find things

- Plan archive: `docs/REFACTOR_STATUS.md` — per-package public surface, intentional plan deviations, open follow-ups, test counts.
- Outstanding work: `docs/PHASE_4_CHECKLIST.md` — exact commands for the remaining steps + acceptance criteria.
- Legacy TF code (`app/`, `run.py`, `Makefile`, `scripts/`, `setup.py`) was removed in Phase 4 step 4. The pre-refactor snapshot is preserved at the `refactor-foundation-pytorch` git tag — `git show refactor-foundation-pytorch:<path>` recovers any file.
- Hydra root: `configs/config.yaml` composes from `configs/{data,model,loss,trainer,evaluation,preprocessing,sweep,experiment}/`.

## Things to avoid

- Adding upward deps (e.g., `bat_data` importing `bat_training`). The dep table in `REFACTOR_STATUS.md` is the contract.
- Modifying `bat_core` mid-task. Workers historically left `TODO(bat_core):` markers and used the closest existing primitive instead; I (the user) batch core changes when convenient.
- Restoring CSV output from new code. The plan locked this; report renders to the unified PDF.
- Bypassing `bat_stats.naming.*` helpers in any plot/file output. The original naming bug is the reason the helpers exist.
- Skipping `pre-commit` because there's no CI. It is the only quality gate.

## Open follow-ups (non-blocking)

These are flagged in `docs/REFACTOR_STATUS.md` but live here too as a quick reference:

- `bat_reporting.compare_runs` could grow a "champion vs candidate" mode using `bat_tracking.get_champion`.
- ONNX export prefers `dynamo_export` with a legacy fallback; full-model dynamo verification is part of the Phase-4 reproducibility check.
