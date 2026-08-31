# `data/` layout

Five top-level directories, each meaning exactly one thing.

Almost nothing here is tracked by git — **deletions are permanent**. The
exceptions are the layout docs, the 57 manifests + `.hash` sidecars, and the
YOLO labels / `data.yaml` / `annotations.xml` under `annotations/` (~9 MB of
text). Every image, video and frame is ignored: they are write-once binaries
that need a **backup**, not version control.

| dir | holds | size |
|---|---|---|
| `raw/` | **source media only** — nothing derived | 5.4 G |
| `interim/` | **derived intermediates** — extracted frames, contact sheets | 1019 M |
| `processed/` | **the two canonical training trees**, and nothing else | 997 M |
| `annotations/` | **YOLO training data** for the preprocessing models | 598 M |
| `work/` | **curation workspaces** — not datasets | 1.9 G |
| `manifests/` | 57 manifest CSVs + `.hash` sidecars + generated `INDEX.md` | 7.8 M |

## Source video

`raw/mauritius/video/` holds all 91 mauritius `.mp4`, so every mauritius frame
under `interim/` is re-derivable here.

**Rousettus source video is deliberately not stored in this repo** — it is held
off-machine, and kept out to save space (`/home` runs near full). What is here
is `interim/rousettus/video/frames/`: 1093 already-extracted 1080×1920 frames
from 12 clips, one clip per identity, session 2025-09-09
(`VID_20250909_143734` … `VID_20250909_144641`).

This is by design, not an omission — the two species have asymmetric front
ends. `scripts/build_variants_from_frames.py` documents rousettus as starting
from *"a flat dir `<frames-root>/*.jpg`"*, while mauritius starts from `.mp4`
via `scripts/build_frontal_dataset.py --videos-dir`. Nothing in the pipeline
reads rousettus video.

Consequence worth knowing: **`interim/rousettus/video/frames/` is the only copy
on this machine.** Losing it means going back to the off-machine originals and
re-extracting, whereas mauritius frames can be rebuilt from `raw/` at any time.

One clip per bat is also why the `original` background arm is unusable for
rousettus — same-bat pairs always share a background. See
`docs/background_leakage.md`.

## The rule that matters

**Manifest `path` columns are literal strings, and `Manifest.from_records`
folds them into `manifest_hash`.** Move any file under
`processed/<species>/video/not_augmented/` and you rewrite 57 manifests, change
57 hashes, and break equality with the `manifest_hash` recorded in every past
MLflow run. `manifest_from_csv` verifies the `.hash` sidecar and raises
`InvalidManifestError` on mismatch, so this fails loudly rather than silently —
but it still costs a migration note (see `docs/manifest_hash_migration.md` for
what that looked like last time).

If you must, use `scripts/migrate_data_layout.py` as the template: it rewrites
paths *and* records the old→new hash mapping. `scripts/reorganise_data_layout.py`
(the 2026-08-30 reorganisation that produced this layout) deliberately moved
nothing any manifest points at, so it needed no such note.

## `processed/` — the canonical trees

```
processed/<species>/video/not_augmented/<variant>/<arm>/<identity>/<image>.jpg
```

Only `mauritius` and `rousettus`, only `video`, only `not_augmented`. The
species/source/augmented levels are kept because the manifest schema carries
those fields and past layouts used them.

**Variants** (`mauritius` has all 10, `rousettus` has 7 — no `blur`, `pool40`
or `recrop`):

| variant | what it is |
|---|---|
| `base` | aligned face crops at native resolution — the default |
| `base_224`, `base_320` | the same crops resampled to a fixed 224 / 320 px edge |
| `bgonly` | face **inpainted away**, background only — Phase 3 arm 3C |
| `blur` | quality-degraded, for the resolution-confound controls |
| `recrop` | resolution-matched recrop (`scripts/build_resolution_recrop.py`) |
| `occl` | occlusion arms `A` / `B` / `C` |
| `occ_roi` | ROI occlusion: `centre` / `matched` / `none` / `periphery` |
| `pool40` | the expanded 40-identity mauritius pool |
| `silhouette` | shape-only control — **no arm level**, goes straight to identity |

**Arms** are `green_bg` / `original_bg` / `random_bg`, plus `face_ellipse` on
some variants.

Two things inside the canonical tree that are **not** arms and that no manifest
references — left in place, `_`-prefixed where possible:

- `mauritius/…/pool40/_review/` (214 M) — review montages from the pool-40 curation.
- `…/base/face_ellipse/`, `…/base_224/face_ellipse/`, `…/base_320/face_ellipse/`,
  `…/pool40/face_ellipse/` — an ellipse-masked arm nothing currently trains on.

### Background caveats before you quote any number

- `original` is **background-confounded**. Each bat has one video, so same-bat
  pairs always share a background; verification is solvable to ROC-AUC 0.74–0.79
  from background colour alone. See `docs/background_leakage.md`.
- Even `green` has a colour leak: 6 numbers (per-channel mean+std inside the
  matte) score 0.849 / 0.830. See `docs/occlusion_results.md`.
- The species comparison needs the resolution-matched arm — see
  `docs/quality_parity.md` and `docs/phase3_results.md`.

## `manifests/`

See **[`manifests/INDEX.md`](manifests/INDEX.md)** — generated by
`scripts/index_data.py`, one row per manifest with its actual variant/arm
directory, image and identity counts, split sizes, and owning config.

Two naming traps the index exists to defuse:

- `mauritius_manifest.csv` / `rousettus_manifest.csv` are the **random**-background
  base arm. The name omits the arm entirely.
- `*_green_bg` is `<arm>_bg`, but `*_green_aligned_224` is `<arm>_<variant>` —
  the two orders are flipped.

8 manifests have no `configs/data/*.yaml` and are flagged `ORPHAN`; they are read
directly by the occlusion/silhouette analysis scripts, not deleted.

## `annotations/`

Training data for the YOLO preprocessing models in `models/preprocessing/`.
See [`annotations/README.md`](annotations/README.md).

## `work/`

- `mauritius_frontal/` — the curation pipeline that produced the 40-identity
  pool: `_experiment_16/` (16 identities), `_manual_review/` (24), `_pool40/`
  (40 **relative** symlinks unifying both). Produced by
  `scripts/build_frontal_dataset.py`, consumed by
  `scripts/build_variants_from_frames.py`.

## Deleted on 2026-08-30 (~2 G)

| was | why |
|---|---|
| `interim/rousettus/video/augmented/` | augmented output filed under `interim/`; regenerable |
| `processed/mauritius/archived/` | pre-migration flat dump; re-downloadable |
| `processed/siamese_input_aligned_picsum/mauritius_images+labels/` | 98 of 99 images and all 99 labels byte-identical to `annotations/face_seg/mauritius/general/`; the 99th differed only by re-encoding |
| `processed/rousettus/still/` | 224×224 crops — all 26 stems re-derivable from `raw/rousettus/still/` |
| 24 × `.DS_Store` | |

## Disk

`data/` is not the problem. As of 2026-08-30, `/home` is at **99 % — 96 G free
of 7.0 T**:

```
outputs/  141 G
models/   131 G
legacy/    97 G
data/      9.9 G   <- this directory
mlruns/     3.8 G
```

Anything that frees real space has to come out of the first three.
