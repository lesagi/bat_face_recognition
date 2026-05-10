# Phase 4 — Migration & cleanup checklist

Phase 0–3 are complete on `main`. Phase 4 is sequenced in five steps.
Steps 1–3 run on the workstation (need data + GPU). Step 4 is a mechanical
deletion once the package tests and isolation check are clean. Step 5 waits
until the parity and baseline acceptance checks pass.

Reference: the approved refactor plan in the original session and
`docs/REFACTOR_STATUS.md` for the per-package merge state.

## Status (live)

| Step | State | Notes |
|---|---|---|
| Pre-flight | ✅ done | `uv sync --frozen` clean (cu121 wheels), 371/372 tests pass, `tools/check_tf_isolation.py` clean |
| 1. Build manifest | ✅ done | `--val-fraction 0.25 --test-fraction 0.25 --seed 7` → 6 train / 3 val / 3 test identities; 1059 rows |
| 2. Siamese parity | ⏳ pending | needs workstation GPU; cu121 wheels in place |
| 3. ArcFace + AdaFace baselines | ⏳ pending | acceptance: `test/roc_auc > 0.9` + 8 PDF sections + experiment-aware filenames |
| 4. Delete TF code | ✅ done in this branch | `app/`, `run.py`, `Makefile`, `scripts/`, `setup.py` removed; pre-refactor snapshot at tag `refactor-foundation-pytorch` |
| 5. Tag `v2.0.0-pytorch` | ⏳ pending | run after steps 2–3 are done |

---

## Pre-flight (one-time)

```bash
# Sync workspace and verify the lock matches.
uv sync --frozen

# Lint + type gate (single-programmer, no CI).
uv run pre-commit run --all-files

# Workspace test suite (per-package, all green at last verification).
uv run pytest packages/

# Confirm no PyTorch package depends on legacy TF surfaces.
uv run python tools/check_tf_isolation.py
```

If any of those fail, fix before continuing.

---

## Step 1 — Build the canonical manifest

Convert `data/processed/` to a 3-way identity-disjoint manifest CSV.

```bash
uv run bat-cli build-manifest \
  --input-dir data/processed/rousettus \
  --output data/manifests/rousettus_manifest.csv \
  --species rousettus \
  --val-fraction 0.15 \
  --test-fraction 0.15 \
  --seed 42
```

Verify:
- `data/manifests/rousettus_manifest.csv` exists.
- `data/manifests/rousettus_manifest.csv.hash` exists (manifest hash sidecar).
- Counts printed in the CLI summary (train / val / test) look reasonable.

Commit the CSV + hash file under `data/manifests/`.

Repeat for `mauritius` if/when needed.

---

## Step 2 — Siamese parity vs legacy TF

Train the new PyTorch Siamese on the same manifest the legacy TF run used,
then compare best test-F1 against the most recent TF-Siamese MLflow run
already in `mlruns/`.

```bash
uv run bat-cli train \
  --experiment siamese_rousettus_random_bg_video \
  --run-name siamese-parity-pytorch \
  --no-permutation
```

Acceptance: PyTorch best test-F1 within ±2 % of legacy TF best test-F1.
Caveat: the new pipeline uses a 3-way identity-disjoint split, so the
comparison is approximate. Document the delta in the run's MLflow notes.

If the delta is larger than ±2 %, investigate **before** moving on:
- Confirm `weight_decay=1e-4` was applied (`SiameseModel.recommended_weight_decay`).
- Confirm Siamese conv/pool dims still match
  `refactor-foundation-pytorch:app/siamese_core/network.py`.
- Confirm the manifest split seed reproduces the same identity partitioning.

---

## Step 3 — ArcFace + AdaFace baselines

```bash
uv run bat-cli train --experiment arcface_rousettus_random_bg_video
uv run bat-cli train --experiment arcface_rousettus_random_bg_video \
  --hydra model=adaface loss=adaface
```

Acceptance per run:
- `test/roc_auc > 0.9`.
- `outputs/runs/.../post_training_report.pdf` renders all eight sections
  (title, training curves, confusion, ROC, identification table, saliency
  composite, t-SNE/UMAP, permutation summary).
- `outputs/runs/.../explanations/` contains saliency PNGs whose filenames
  embed `species_source_background_model_loss`.
- `outputs/runs/.../permutation/` contains the permutation null-distribution
  PDFs/JSON.

If a section is empty or a filename is missing the experiment slug, raise
an issue **before** Step 4.

---

## Step 4 — Delete TF code

Reversible via git but a big diff. The legacy source remains available at
tag `refactor-foundation-pytorch` for parity archaeology.

```bash
git checkout -B refactor/remove-tf origin/main

# Verify isolation one more time (paranoia).
python tools/check_tf_isolation.py

# Delete the TF surfaces. git history retains them at refactor-foundation-pytorch.
git rm -r app/ scripts/
git rm run.py Makefile setup.py

# The Hydra `outputs/` and `mlruns/` directories are user-machine artifacts;
# leave them as-is. They are already in .gitignore.

git commit -m "refactor: delete TF code (Phase 4 step 4)

Replaced wholesale by the PyTorch packages under packages/. Git history
retains the original code at tag refactor-foundation-pytorch."

git push -u origin refactor/remove-tf
```

Then merge as a normal PR.

---

## Step 5 — Tag the milestone

```bash
git checkout main
git pull --ff-only

# Tag the post-deletion main.
git tag -a v2.0.0-pytorch -m "PyTorch refactor complete: Siamese + ArcFace + AdaFace"
git push origin v2.0.0-pytorch
```

Optional, for archeology:

```bash
# These commits already on main today — useful breadcrumbs.
git tag -a refactor-phase-1-complete dc84d78 -m "All Phase-1 packages merged + stabilize"
git tag -a refactor-phase-2-complete e49f235 -m "All Phase-2 integration packages merged + stabilize"
git tag -a refactor-phase-3-complete 40eeb75 -m "bat_cli + auto-pipeline + promotion gate complete"
git push origin refactor-phase-1-complete refactor-phase-2-complete refactor-phase-3-complete
```

(Adjust SHAs if `git log --oneline` shows different ones at merge time.)

---

## Reproducibility check (after Step 5)

Per the approved plan, the same Hydra cfg + same manifest hash + same seed
must reproduce `test/roc_auc` to 4 decimal places.

```bash
uv run bat-cli train --experiment arcface_rousettus_random_bg_video --run-name repro-1
uv run bat-cli train --experiment arcface_rousettus_random_bg_video --run-name repro-2
# Compare the two MLflow runs' test/roc_auc.
uv run bat-cli compare <run-id-1> <run-id-2> --output outputs/compare/repro.html
```

Document any drift in the v2.0.0-pytorch release notes.

---

## Out-of-scope (deferred per plan)

- Serving / inference API.
- SLURM templates (single-workstation use).
- WandB integration (MLflow stays; can add later as a `bat_tracking` backend).
- Triplet/contrastive online hard mining (baseline only — not the
  recommended path).
