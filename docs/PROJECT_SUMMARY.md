# Project summary — bat face recognition

Single go-forward reference for the project. Replaces the now-deleted
working docs (`REFACTOR_STATUS.md`, `PHASE_4_CHECKLIST.md`,
`PHASE_4_STEP2_FIXES.md`, `RESEARCH_DIRECTIONS.md`); their content is
consolidated here and the per-step debugging narratives are preserved
in git history + commit messages (#27–#34) + the
`refactor-foundation-pytorch` tag for pre-refactor code archaeology.

This is a **university research project** on open-set bat face
recognition. The goal is publishable results, not production
deployment. Recommendations below are framed accordingly.

---

## 1. What was built

A complete TF/Keras → PyTorch refactor culminating in the
`v2.0.1-pytorch` release tag. Three model families end-to-end on
identity-disjoint splits with a single CLI, MLflow tracking, automatic
8-section PDF reports, permutation-test statistical analysis, and
bit-exact reproducibility under `trainer.deterministic=true`.

### Architecture — 13-package `uv` workspace

| Package | Role | Key public surface |
|---|---|---|
| `bat_core` | Types + Protocols, no I/O | `ImageRecord`, `Manifest`, `Embedding`, `Predictions`, `EvalReport`, `RunArtifacts`, `SaliencyImage`; `FaceModel`, `Loss`, `Trainer`, `InterpretabilityAdapter`, `Tracker` Protocols |
| `bat_data` | Manifest + dataset + 3-way splitter + miners | `build_manifest`, `manifest_to_csv` / `manifest_from_csv`, `IdentitySplitter`, `BatDataset`, `HardNegativeMiner`, `SemiHardMiner` |
| `bat_preprocessing` | YOLO seg/pose, alignment, transforms, video extraction | `PreprocessingPipeline`, `YOLOSegmenter`, `YOLOPose`, `FaceAligner`, `VideoExtractor`, prediction-cache primitives, background generator |
| `bat_models` | Siamese / ArcFace / AdaFace `nn.Module`s | `SiameseModel`, `ArcFaceModel`, `AdaFaceModel`, `resnet50_backbone`, `ArcFaceHead`, `AdaFaceHead`, `export_onnx` |
| `bat_losses` | Pair + embedding losses | `BCELoss`, `FocalLoss`, `TripletLoss`, `ArcFaceLoss`, `AdaFaceLoss`, `CosFaceLoss`, `SubCenterArcFaceLoss` (each declares `.family`) |
| `bat_training` | Loop-only trainers + EMA + AMP + checkpointing + Accelerate | `PairTrainer`, `EmbeddingTrainer`, `make_trainer`, `TrainerConfig`, `BestCheckpointTracker`, `OptunaPruningCallback` seam |
| `bat_evaluation` | Verification + identification, dataclass output (no CSVs) | `evaluate_predictions`, `optimize_youden_j`, `run_eval_protocol`, `split_gallery_probe` |
| `bat_interpretability` | Saliency / GradCAM / projection adapters | `SiameseSaliencyAdapter`, `GradCAMAdapter`, `EmbeddingProjectionAdapter`, `select_adapter`, `composite_from_saliency_images` |
| `bat_stats` | Permutation tests + experiment-aware visualizer | `run_inference_test`, `run_retrain_test`, `PermutationVisualizer`, `experiment_name`, `build_filename` |
| `bat_tracking` | MLflow facade + HP audit allowlist + champion registry | `MLflowTracker`, `start_run`, `register_model`, `get_champion`, `would_promote`, `promote_to_champion`, `KEEP` / `DROP_PREFIXES` |
| `bat_reporting` | Unified post-training PDF + run-comparison HTML | `build_unified_pdf`, `compare_runs`, `ReportData`, 8-section reportlab pipeline |
| `bat_sweeps` | Optuna val-only HP search | `make_study`, `build_objective`, `run_sweep`, `OptunaPruningCallback` |
| `bat_cli` | Single-entry interactive launcher | `bat-cli`: `train` / `sweep` / `evaluate` / `permutation-test` / `compare` / `compare-champion` / `promote` / `build-manifest` |

**Invariants worth preserving** (don't undo these):

- **No upward dependencies.** A package may only import from `bat_core`
  or packages strictly lower in the dep graph. `bat_core` imports
  nothing from the workspace and stays torch-light (TYPE_CHECKING only).
- **Trainers are loops.** Data wiring goes in `bat_data`, MLflow in
  `bat_tracking`, eval in `bat_evaluation`, loss math in `bat_losses`.
  The trainer keeps only optimizer / scheduler / AMP / EMA /
  checkpointing / callbacks.
- **Test split is sacred.** `bat_sweeps` reads only val metrics; the
  Optuna objective never calls `trainer.test`.
- **Naming bug fix is load-bearing.** Every plot title / filename / axis
  label routes through `bat_stats.naming.experiment_name(cfg)`. Don't
  bypass.
- **No CSV emission from new code.** The PyTorch eval/report path
  returns dataclasses and renders into the unified PDF.
- **Identity-disjoint splits.** Train / val / test identity sets must
  be disjoint. Never propose a random pair split for "fair" comparison
  with legacy TF numbers — that paradigm is what we're explicitly
  fixing.

### Auto post-training pipeline

`bat-cli train` runs the full sequence in one invocation. Each step is
`_safe`-wrapped: failure logs a warning, downstream steps still attempt.

1. `trainer.fit(train_loader, val_loader)` → `RunArtifacts`
2. Test eval → `EvalReport`
3. Saliency + GradCAM + t-SNE/UMAP projection (cross-split sample) →
   `<run>/explanations/`
4. Unified 8-section PDF → `<run>/post_training_report.pdf`
5. MLflow artifact upload (PDF, explanations, permutation outputs)
6. Inference-mode permutation test → `<run>/permutation/`
7. Champion promotion (opt in with `--promote` or `--prompt-promote`)

Skip flags: `--no-mlflow`, `--no-permutation`, `--no-explanations`,
`--explanations-per-split N`.

### Release tags

- `refactor-foundation-pytorch` — pre-refactor snapshot of the TF
  stack. Use `git show refactor-foundation-pytorch:<path>` for
  archaeology.
- `refactor-phase-{1,2,3}-complete` — per-phase milestones.
- `v2.0.0-pytorch` — refactor complete: Siamese + ArcFace + AdaFace
  with identity-disjoint splits, auto post-training pipeline, MLflow
  tracking.
- `v2.0.1-pytorch` — adds bit-exact reproducibility under
  `trainer.deterministic=true` and documents the mauritius
  cross-species data point.

---

## 2. Central research result so far

**Identity-disjoint generalization stalls at ≤ ~12 train identities
regardless of model family.**

| Model | rousettus `test/roc_auc` | rousettus `test/f1` | mauritius `test/roc_auc` |
|---|---|---|---|
| Siamese (4-conv) | **0.8047** | **0.7420** | (not run) |
| ArcFace (ResNet50, tuned) | 0.5414 | — | 0.5263 |
| AdaFace (ResNet50, tuned) | 0.5075 | — | (not run) |

Three notable findings:

1. **Siamese beats arc-margin** on this dataset size. Pair training
   directly optimizes pairwise similarity — the metric verification
   measures — so it doesn't depend on closed-set geometry
   generalizing. ArcFace/AdaFace optimize angular separation among
   *training* identities; on 6 of them that geometry doesn't transfer
   to 3 held-out identities. On larger face datasets (MS1M et al.)
   the ordering typically reverses.
2. **Mauritius hits the same wall** at 11 identities. Confirms the
   limitation is small-identity-count, not species-specific.
3. **Paper-grade reproducibility is now in place.** Two ArcFace runs
   with identical Hydra cfg + `trainer.deterministic=true` matched
   bit-exactly across all 23 logged metrics (delta = 0.000000).

The legacy TF Siamese hit `test/f1 = 0.9872` on a *random-pair* split
where test pairs shared identities with train. That number is **not
comparable** to ours. The identity-disjoint paradigm shift is part of
the contribution.

---

## 3. Recommended next steps (mandatory for research output)

These items directly contribute to publishable results. Listed in
suggested execution order.

### 3.1 Re-enable augmentation

Every experiment to date runs with `data.augmented=false`. The legacy
TF pipeline used per-image HorizontalFlip + RandomBrightnessContrast +
RandomGamma + RGBShift + VerticalFlip + AdvancedBlur (~60 augmented
copies per source image). Augmentation is the textbook small-data
remedy and the most likely single-knob improvement for Siamese and
arc-margin on a 12-identity dataset.

**Concrete next step**: port the augmenter from
`git show refactor-foundation-pytorch:app/data_augmentation/augmentor.py`,
generate `data/processed/rousettus/.../augmented/`, build a parallel
manifest, re-run Siamese + ArcFace + AdaFace, compare `test/*` vs the
no-aug baselines. Use `trainer.deterministic=true`.

### 3.2 Multi-seed runs + permutation tests

Single-run numbers don't have uncertainty estimates. For every
configuration you plan to report:

- Run 3–5 seeds with `trainer.deterministic=true` (the v2.0.1 fix
  makes these true independent samples, no infrastructure variance).
- Aggregate: mean ± std on `test/roc_auc`, `test/f1`, `test/top1`.
- Run `bat-cli permutation-test` for p-values on the best seed per
  configuration.

These two together turn "Siamese 0.80 vs ArcFace 0.54" into "Siamese
0.80 ± 0.03, ArcFace 0.54 ± 0.04, paired-difference p < 0.01 over
N permutations" — i.e., a defensible claim.

### 3.3 Combined-species manifest (23 identities)

12 rousettus + 11 mauritius. ~2× the current per-species identity
count — exactly the regime where arc-margin starts being able to use
its angular-separation objective.

**Concrete next step**: extend `bat-cli build-manifest` to accept
multiple `--input-dir` flags (or call it twice + concatenate), keep
`species` as metadata not label, train ArcFace v2 + Siamese, compare.

A positive result here is a real contribution. A negative result is
still publishable ("identity-disjoint open-set face recognition needs
N ≥ … training identities").

### 3.4 Triplet loss with hard negative mining

The original refactor plan deferred this ("baseline only — not the
recommended path"). On small datasets the literature consensus is the
opposite: arc-margin shines on large datasets, triplet + hard mining
shines on small-N — exactly our regime.

Infrastructure that already exists:
- `bat_losses.TripletLoss`
- `bat_data.HardNegativeMiner`, `bat_data.SemiHardMiner`

What's missing: wiring the miners into the `PairTrainer` train loop
(currently only handles BCE/Focal).

If you only do one method-side item from this list, do this one.

### 3.5 Cross-species transfer experiment

Train on rousettus, evaluate on the mauritius test split (and vice
versa). True open-set generalization across species.

**Concrete next step**: `bat-cli evaluate` can already point at a
different manifest via `--hydra data.manifest_path=...`. Train ArcFace
v2 on rousettus, call `bat-cli evaluate` against the mauritius test
split, repeat in reverse. The *gap* between within-species and
cross-species is the publishable quantity.

---

## 4. Optional steps (not mandatory for university research)

These are real follow-ups but skip them unless thesis scope explicitly
demands them. None of them are necessary for getting publishable
identity-disjoint face-recognition results.

### 4.1 Research-side optionals (would extend the paper)

- **Face-domain pre-training**: replace the ImageNet-pretrained
  ResNet50 backbone with an ArcFace-MS1M checkpoint, or
  self-supervised pre-training (DINO/MAE/SimCLR) on the union of bat
  datasets. Bigger lift, potentially big jump in generalization.
- **Few-shot / metric-learning methods**: Prototypical Networks,
  Matching Networks, Relation Networks. Designed for our regime; not
  implemented today. Each would be a new `bat_models` + `bat_training`
  family.
- **Optuna HP sweep** via the existing `bat_sweeps`. Once you've
  picked a model family for the paper, run a val-only sweep on (lr,
  weight_decay, margin/scale or focal alpha/gamma, embedding_dim).
  Defensible HPs > hand-tuned guesses, but probably not the
  difference between a thesis passing and failing.

### 4.2 Infrastructure-side optionals (would NOT extend the paper)

These were explicitly deferred by the original refactor plan and
remain deferred. They don't move the science:

- Serving / inference API.
- ONNX dynamo full-model verification (export currently has a
  `torch.onnx.export` fallback that works; verifying the dynamo path
  end-to-end is a nice-to-have, not a research output).
- SLURM templates (single-workstation use; not needed).
- WandB integration as a `bat_tracking` backend (MLflow works fine).
- CI on GitHub Actions (single-programmer; pre-commit is the only
  quality gate and that's sufficient).
- Dockerfile (`uv sync --frozen` is the entire setup).

---

## 5. Where to find things

- `configs/config.yaml` — Hydra root; composes from
  `configs/{data,model,loss,trainer,evaluation,preprocessing,sweep,experiment}/`.
- `data/manifests/` — committed CSV manifests + `.hash` sidecars
  (currently gitignored locally; commit deliberately for the paper).
- `outputs/runs/` and `mlruns/` — per-run artifacts. Gitignored.
- `models/preprocessing/` — YOLO seg + pose weights for the still-
  image extraction pipeline. Tracked.
- `refactor-foundation-pytorch` tag — pre-refactor TF code for
  archaeology. `git show <tag>:<path>` recovers any file.

## 6. Common operations

```bash
# Setup
uv sync --frozen

# Quality gate (pre-commit is the only one; no CI)
uv run pre-commit run --all-files
uv run pytest packages/

# Build a manifest
uv run bat-cli build-manifest \
  --input-dir data/processed/<species>/.../random_bg \
  --output data/manifests/<species>_manifest.csv \
  --species <species> --val-fraction 0.25 --test-fraction 0.25 --seed 7

# Train (with auto post-training pipeline)
uv run bat-cli train --experiment <experiment_yaml>

# For a reproducible run pinnable to 4 decimal places
uv run bat-cli train --experiment ... --hydra trainer.deterministic=true

# Interactive walkthrough (every action has a picker)
uv run bat-cli
```
