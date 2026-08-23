"""Can the verification task be solved from the background alone?

The concern, and it is a design flaw rather than a bug. Evaluation is open-set
*verification*: every unordered pair of test images is scored, labelled 1 if the
two images show the same bat. But **each bat was filmed in exactly one video**,
so every same-bat pair shares a background and no different-bat pair does. A
scorer can therefore separate the pairs perfectly using the background and never
look at a face.

Identity-disjoint splitting does not protect against this. It prevents the model
from memorising *which* bat is which, but the cue here is within-pair background
*agreement*, which needs no memory of any particular animal.

This probe answers "is the cue available?" with **no model at all**: it replaces
the embedding-cosine score with a background-only descriptor similarity and
re-runs the identical pair construction and metric. Whatever ROC-AUC comes back
is obtainable without a single face pixel.

Negative controls are built in. ``green`` paints a flat colour behind every face
and ``random`` draws a fresh background per image, so neither can carry
bat-identity information — both must come back near 0.50. If they do not, the
probe is measuring something other than what it claims (face leakage through an
imperfect mask, say) and its ``original`` number cannot be trusted either.

Pair construction mirrors ``bat_evaluation.verification.predictions_from_embedding``
exactly: all unordered pairs (i, j) with i < j over the test split, label 1 when
identities match.

Usage:
    uv run python scripts/probe_background_leakage.py
    uv run python scripts/probe_background_leakage.py --folds 0 1 2 --species rousettus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from occlusion_probe import mask_from_green  # noqa: E402

from bat_cli.runtime import (  # noqa: E402
    compose_config,
    find_project_root,
    load_manifest,
    resplit_manifest,
)

SPECIES = ("mauritius", "rousettus")

# The background variants, and the experiment whose data config points at each.
# The random arm resolves to `manifest_{sp}` (see configs/experiment/*_random_*).
BACKGROUNDS = ("original", "green", "random")
EXPERIMENT = "arcface_{sp}_{bg}_bg_video_tuned"

DEFAULT_OUT = Path("outputs/leakage/background_probe.json")

# Colour-histogram bins per channel. 4^3 = 64 bins is coarse enough that two
# frames of one video land in the same bins despite compression noise, and fine
# enough to separate genuinely different scenes.
BINS = 4

# Grow the face mask before inverting it, so a slightly loose segmentation cannot
# leak face pixels into the "background" descriptor — that would manufacture
# exactly the effect this probe is testing for.
#
# 24 px is calibrated, not guessed. Sweeping it (3 folds, both species) shows the
# green negative control falling monotonically to chance while `original` barely
# moves, which is the signature of a real far-field background cue rather than
# boundary bleed:
#
#   dilate   mau green   rou green   |   mau original   rou original
#        0       0.762       0.711   |          0.832          0.760
#        6       0.592       0.555   |          0.804          0.766
#       14       0.545       0.538   |          0.758          0.779
#       24       0.501       0.504   |          0.719          0.782
#
# Below 24 the controls are contaminated by the build's 4 px mask feathering, so
# any `original` number quoted at a smaller dilation is not trustworthy.
FACE_DILATE_PX = 24


def background_descriptor(
    image_path: Path, green_path: Path, dilate_px: int = FACE_DILATE_PX
) -> np.ndarray | None:
    """Colour descriptor over background pixels only, or None if unusable."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    face = mask_from_green(green_path, img.shape[:2])
    if face is None:
        return None
    # Grow the face region so its boundary is excluded from the background.
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
        face = cv2.dilate(face.astype(np.uint8), kernel)
    bg = face == 0
    if bg.sum() < 128:
        return None

    pixels = img[bg].astype(np.float32)
    hist, _ = np.histogramdd(
        pixels, bins=(BINS, BINS, BINS), range=((0, 256), (0, 256), (0, 256))
    )
    hist = hist.ravel()
    total = hist.sum()
    if total <= 0:
        return None
    hist = hist / total

    # Histogram captures the palette; mean/std captures overall exposure and
    # contrast, which differ between rooms even when palettes overlap.
    stats = np.concatenate([pixels.mean(axis=0) / 255.0, pixels.std(axis=0) / 255.0])
    return np.concatenate([hist, stats]).astype(np.float64)


def probe_fold(
    species: str, background: str, fold: int, root: Path, dilate_px: int = FACE_DILATE_PX
) -> dict[str, Any] | None:
    """Score every test pair by background similarity and return its ROC-AUC."""
    cfg = compose_config(
        experiment=EXPERIMENT.format(sp=species, bg=background),
        overrides=(
            f"data.fold_id={fold}",
            "data.resplit=true",
            "data.split_size_mode=seeded",
        ),
    )
    manifest = load_manifest(cfg, root=root)
    manifest, info = resplit_manifest(manifest, cfg)
    if info is None:
        return None

    records = manifest.filter_split("test")
    descriptors: list[np.ndarray] = []
    identities: list[str] = []
    n_skipped = 0
    for record in records:
        path = Path(record.path)
        green = Path(str(path).replace(f"{background}_bg", "green_bg"))
        if not green.exists():
            n_skipped += 1
            continue
        desc = background_descriptor(path, green, dilate_px)
        if desc is None:
            n_skipped += 1
            continue
        descriptors.append(desc)
        identities.append(record.identity)

    if len(descriptors) < 4:
        return None

    feats = np.stack(descriptors)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True).clip(1e-12)
    sim = feats @ feats.T

    # Identical pair construction to predictions_from_embedding.
    i, j = np.triu_indices(len(identities), k=1)
    ids = np.asarray(identities, dtype=object)
    labels = (ids[i] == ids[j]).astype(int)
    scores = sim[i, j]

    if labels.sum() == 0 or labels.sum() == labels.size:
        return None

    return {
        "species": species,
        "background": background,
        "fold": fold,
        "n_images": len(descriptors),
        "n_identities": int(len(set(identities))),
        "n_pairs": int(labels.size),
        "n_genuine": int(labels.sum()),
        "n_skipped": n_skipped,
        "face_dilate_px": dilate_px,
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", choices=(*SPECIES, "all"), default="all")
    ap.add_argument("--background", choices=(*BACKGROUNDS, "all"), default="all")
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(20)))
    ap.add_argument("--face-dilate-px", type=int, default=FACE_DILATE_PX,
                    help="Grow the face mask by this many px before inverting it. "
                         "Raising it is the sensitivity check: if the green control "
                         "falls to ~0.50 while original stays high, the leak is real.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    root = find_project_root()
    species_list = SPECIES if args.species == "all" else (args.species,)
    bg_list = BACKGROUNDS if args.background == "all" else (args.background,)

    print("=== background-leakage probe (no model) ===")
    print("Scores test pairs by background-colour similarity alone.")
    print("green and random are NEGATIVE CONTROLS and must return ~0.50.\n")

    results: list[dict[str, Any]] = []
    for species in species_list:
        for background in bg_list:
            aucs = []
            for fold in args.folds:
                try:
                    entry = probe_fold(species, background, fold, root, args.face_dilate_px)
                except Exception as exc:  # noqa: BLE001 - one fold must not kill the sweep
                    print(f"  !! {species}/{background} fold {fold}: {exc}")
                    continue
                if entry is None:
                    continue
                results.append(entry)
                aucs.append(entry["roc_auc"])
            if not aucs:
                print(f"  {species:<10} {background:<9} no usable folds")
                continue
            arr = np.asarray(aucs)
            flag = ""
            if background in ("green", "random") and abs(arr.mean() - 0.5) > 0.10:
                flag = "   <-- CONTROL FAILED, probe is suspect"
            elif background == "original" and arr.mean() > 0.70:
                flag = "   <-- LEAK"
            print(
                f"  {species:<10} {background:<9} n_folds={len(arr):2d}  "
                f"ROC-AUC {arr.mean():.3f} ± {arr.std(ddof=1) if arr.size > 1 else 0:.3f}"
                f"  [{arr.min():.3f}, {arr.max():.3f}]{flag}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "description": "ROC-AUC obtainable from background colour alone, "
                "using the same pairs and metric as model evaluation",
                "bins_per_channel": BINS,
                "face_dilate_px": args.face_dilate_px,
                "negative_controls": ["green", "random"],
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
