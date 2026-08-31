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
- **Pose runs on the mask-crop, not the full frame.** `models/preprocessing/face_pose.pt` was trained on tight square face crops (`legacy/face_annotation_eyes_nose/`, 43 labeled faces filling the frame), so full-frame inference wrecks keypoints (border/nose misdetections → tilted alignment). Segmentation runs on the full frame; pose runs on the mask-centered square crop with keypoints mapped back to frame coords. See `scripts/build_variants_from_frames.py:pose_on_mask_crop`.
- **No CSV output from new code.** The legacy TF stack emitted CSVs everywhere; the PyTorch eval/report path returns dataclasses and renders directly to the PDF. New dataset/analysis artifacts use Parquet (per-image tables) or JSON (summaries) — see `outputs/quality/`.
- **Runtime re-split is opt-in, and `exact` sizing is frozen.** Manifests carry a baked `split` column; `bat_cli.runtime.resplit_manifest` re-partitions in memory only when `data.resplit` or `data.fold_id` is set. `IdentitySplitter(size_mode="exact")` must stay bit-identical — it reproduces the 90 published multi-seed runs (mauritius 12/2/2, rousettus 6/3/3). `size_mode="seeded"` treats `val_fraction`/`test_fraction` as **floors** and draws the actual identity counts from the seed, using a separate RNG stream so the identity-shuffle stream (and therefore `exact` mode) is untouched.
- **A cull produces a new arm; it never mutates a published one.** `rebuild_dataset.py --arm <name>` writes `<arm>/`, `<arm>_320/` and `data/manifests/<species>_<arm>_*`. Regenerating an *existing* manifest re-splits it, changes `manifest_hash`, and severs every published MLflow run citing it. A directory walk also cannot reproduce the manifests that are not directory walks — `*_band_*` are 10-identity band-restricted subsets, `*_intersect_*` is an intersection with another background, `occl_*` are occlusion arms. The derived arms (`blur`, `recrop`, `bgonly`, `occ_roi`, `occl`, `silhouette`) each read a published manifest by name and belong to a concluded experiment, so they are rebuilt individually and on purpose.
- **`data/curated/<species>/decisions.json` is the filter.** Do not curate by adding or deleting files under `keep/`, `unreviewed/`, or `dropped/` and leaving it at that — run `curate_frames.py --authority dirs --execute` so the JSON absorbs the change. Nothing is ever deleted, so any decision is reversible.
- **`manifest_hash` in MLflow always means the on-disk hash.** `Manifest.from_records` folds `split` into the digest, so a re-split rehashes; log `bundle.source_manifest_hash`, never `bundle.manifest.manifest_hash`, or runs stop being comparable to the published ones.
- **New MLflow params need an HP-audit entry.** `bat_tracking.hp_audit.filter_params` silently drops anything outside `KEEP`/`KEEP_PREFIXES` (the `split.` prefix is allowed).
- **Hydra cfg → caller responsibility.** `bat_training` does not log Hydra cfg itself; the CLI does (avoids leaking Hydra into trainer internals).

## Dataset generation (two stages)

Curation and transformation are separate, because every transformed variant is
downstream of one set of culled full-resolution frames. Filtering per variant
directory does not scale — there are ten of them.

```
raw video ──extract, keep everything scoreable──▶ data/curated/<species>/
                                                     decisions.json   ← edit this
                                                     keep/ unreviewed/ dropped/
                                                     thumbs/ review/*.html
                                                          │
                              rebuild_dataset.py ─────────┘
                                                          ▼
                            <arm>/ + <arm>_320/  ──▶  data/manifests/<species>_<arm>_*.csv
```

**Stage 1 — the cull is data, not directory state.** `decisions.json` holds one
verdict per frame (`keep` / `drop` / `undecided`) and is the single source of
truth. Two review surfaces feed it, and either can win; every run ends by
relocating files so the tree matches the JSON, so the two cannot drift.

```bash
# extract the reviewable superset (every frame with a usable det+seg+pose)
uv run python scripts/build_frontal_dataset.py --videos <list.json|stems> \
    --keep-all --output data/curated/<species>/_staging --device cuda:0
uv run python scripts/curate_frames.py --species <sp> --init --execute

# review: open the sheets, click K/D/U, Export…  (saves to localStorage as you go)
uv run python scripts/curate_frames.py --species <sp> --sheets
uv run python scripts/curate_frames.py --species <sp> \
    --authority html --import <export.json> --execute
# …or drag rejects into dropped/ in a file manager, then
uv run python scripts/curate_frames.py --species <sp> --authority dirs --execute
```

**Stage 2 — one command rebuilds everything downstream.**

```bash
uv run python scripts/rebuild_dataset.py --species <sp> --arm <name> --execute
uv run python scripts/rebuild_dataset.py --list-derived   # what is NOT auto-rebuilt
```

Notes that are easy to get wrong:

- **Three states, not two.** Re-extraction surfaces frames no human ever ruled
  on. `drop` would falsely imply rejection; `keep` would silently change the
  dataset. `undecided` is the honest answer and is excluded from builds.
- Frames are keyed `<identity>/f<frame:06d>` — on the frame number, not the
  filename, which carries a mask-area suffix that shifts if segmentation is
  re-run. The key survives re-extraction; the filename does not.
- Dry-run is the default for both scripts. Nothing moves without `--execute`.
- `build_frontal_dataset.py` without `--keep-all` keeps its original destructive
  behaviour, so the historical builds stay reproducible.

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
| `compare-champion <candidate> --model-name <name>` | Same report, left column auto-resolved via `bat_tracking.get_champion(name)` |
| `promote <run>` | Promote registered model if criterion beats incumbent |
| `build-manifest` | Walk a directory tree → split + write manifest CSV |

Step-skip flags on `train`: `--no-mlflow`, `--no-explanations`, `--no-permutation`, `--explanations-per-split N`.
Promotion: `--promote` (silent auto-on-improve) and `--prompt-promote` (ask once if improves) are mutually exclusive; default is no promotion.
K-fold: `--fold N` re-splits the manifest in memory (see "Runtime re-split" below).

## Where to find things

- **Start here: `docs/STATUS.md`** — what is defensible, what has been refuted (five
  claims so far), what is open, every check that was run with its verdict, the traps
  that silently produce wrong answers, and what to do next. Written to be picked up
  cold.

- **Curating which frames the model sees**: `data/curated/<species>/decisions.json`
  plus the contact sheets at `data/curated/<species>/review/index.html`. See
  "Dataset generation" above. Current state (mauritius, 2026-09-01): a 32-identity
  superset of 6,114 frames — all 22 usable 31-August recordings plus 10 other-day
  bats (seed 831, `outputs/quality/curation_target_set.json`). 1,179 frames are
  `keep`, seeded from the prior human cull and verified to reproduce it exactly
  (100% of 1,179, zero unmatched). The 8 newly-processed 31-August bats are
  `undecided` and need a first review pass before an arm built from this cull
  covers all 32. `20230831_161613` is deliberately excluded — a 1.5 s false start
  restarted 7 s later as `20230831_161620`, so treating it as its own identity
  would put one bat in both train and test.

- **Phase 3 controls (start here for any species or background claim)**:
  `docs/phase3_results.md` — 624 runs. Two results override earlier docs. (1) A
  model trained on images with the face *inpainted away* scores 0.831/0.780 on
  held-out bats against a control at chance, so the `original` arm is
  substantially background matching and cannot carry a headline. (2) Matching
  image quality does **not** explain the species difference: it survives three
  independent controls, largest reduction 0.043 ROC-AUC. This reverses
  `quality_parity.md`'s "substantially a measurement artifact" reading for
  recognition (that result was frozen-backbone embedding geometry, not trained
  verification).
- **The six-number colour baseline — check any "the model recognises faces" claim against this first**: `docs/occlusion_results.md`. Per-channel mean+std inside the green face matte (6 numbers, no spatial content) scores 0.849 (mauritius) / 0.830 (rousettus) median over the same 20 folds the models use. Paired per fold, **no model's margin excludes zero on the positive side** (best +0.042, CI spans zero), and rousettus Siamese is significantly *worse* (−0.149, CI [−0.304, −0.081], winning 2/20). Same shape as the background leak and the same cause — one video per bat, so anything constant within a clip is a free identity cue. Also why the ROI-occlusion arm's positive control failed: a global statistic cannot be ablated by destroying any region. Run `scripts/probe_face_colour.py`.
- Species dataset comparability: `docs/quality_parity.md` — the image-quality battery, the equivalence (TOST) tests, and the intrinsic-separability result. Short version: the two species' datasets differ on 11 of 12 quality metrics, effective resolution separates them completely (1.97×, Cliff's δ = 1.00), and matching resolution removes most of the apparent separability gap. Any species comparison needs the matched arm.
- **Background leakage — read before quoting any `original`-background result**: `docs/background_leakage.md`. Each bat has one video, so same-bat pairs always share a background and the verification task is solvable to ROC-AUC 0.74–0.79 from background colour alone, with no model. `green`/`random` are clean (0.50). No model significantly beats that baseline on `original`. The arm is unfixable with this data (one video per bat); the headline belongs on `green`.
- Per-species saliency: `docs/saliency_species.md` — group maps, pose-derived eye/nose ROIs, pointing game. Three gotchas. (1) Grad-CAM's grid is `input_edge/32`, so at the tuned 112px it is **4×4** — one cell larger than the eye ROI — and says nothing; ROI claims need ≥320px. (2) Grad-CAM with the `class_logit` target silently returns all-zero maps for 20% (mauritius) / 46% (rousettus) of images, because the target is a scale-invariant cosine while Grad-CAM assumes the target grows with activation magnitude; that differential attrition biased the species comparison. Use **Integrated Gradients** (`--method ig`, pixel resolution, zero attrition) or the `emb_mag` target. (3) **The randomisation control FAILS** — an untrained network of the same architecture peaks on an eye in 64% of mauritius images and reproduces the species difference at Cliff's δ = +0.94 vs the trained +0.98. So eye-attention and the species gap are properties of the *images* (very likely the resolution confound), not of learning. Do not quote eye-attention against the 16% area-share null. (4) **The "opposite directions per species" follow-up was pre-registered and then FAILED** (`docs/saliency_preregistration.md`, 40 models, 2 species × {green, original} × 10 folds): the eye version replicates on `original` (δ +0.42) but not on `green` (δ +0.18, CI spans zero), so it is background-confounded — arm 3C shows `original` is 93% solvable with the face deleted; and the nose version **reversed** (δ −0.44 on original, significant), i.e. the single-seed exploratory reading was an artifact. What *does* replicate, in 38 of 40 models on both backgrounds, is that bat-specific training moves attribution **away** from eyes and nose and **toward** the periphery. (5) Two mechanical traps, both now fixed but relevant to any older output: `find_run_dir_for_edge` used to ignore background and pick by mtime, so it would load a dilated checkpoint into a stock ResNet without raising (shapes match); and the SmoothGrad noise was unseeded, giving ±6 pp run-to-run on the pointing game. Pin with `--checkpoint SPECIES=PATH`.
- Power ceiling: `scripts/analyze_power_ceiling.py` — the k-fold design's minimum detectable model difference is **0.44 ROC-AUC** against a largest observed gap of 0.207, so "not significant" means "undetectable". Driven by `n_test/n_train`, not fold count: more folds cannot help.
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

- ONNX export prefers `dynamo_export` with a legacy fallback; full-model dynamo verification is part of the Phase-4 reproducibility check.
