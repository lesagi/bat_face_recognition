"""Silhouette-only control variant: derive a pure face-shape image (white mask on
black, zero internal texture) from each green-background crop, and write a matching
manifest with identical identities/splits.

Purpose (thesis internal-validity control): the `green` variant removes the
background but keeps the tight segmentation *silhouette*, which is itself an
identity-correlated cue. Training on the silhouette alone measures how much of the
recognition signal is carried by shape vs. internal facial appearance. If a model
trained on silhouettes lands near chance, shape alone is insufficient (texture
matters); if it matches `green`, shape is a strong channel.

  conda activate frec && uv run python scripts/generate_silhouette.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SPECIES = ("mauritius", "rousettus")


def silhouette_from_green(img: np.ndarray) -> np.ndarray:
    """Green-screen matte -> solid white face shape on black (BGR uint8).

    Background is the (0,255,0) fill (feathered at the edge); everything else is
    the segmented face. Clean specks/holes so the shape is a single solid blob.
    """
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    bg = (g > 150) & (r < 120) & (b < 120)
    face = (~bg).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    face = cv2.morphologyEx(face, cv2.MORPH_OPEN, k)   # drop specks
    face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, k)  # fill pinholes
    # keep only the largest connected component (the face)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(face, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        face = np.where(lab == largest, 255, 0).astype(np.uint8)
    return cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", nargs="*", default=list(SPECIES))
    args = ap.parse_args()

    for sp in args.species:
        src_csv = Path(f"data/manifests/{sp}_green_bg_manifest.csv")
        if not src_csv.exists():
            print(f"[{sp}] SKIP: {src_csv} not found")
            continue
        df = pd.read_csv(src_csv)
        out_root = Path(f"data/processed/{sp}/video/not_augmented/silhouette")
        out_root.mkdir(parents=True, exist_ok=True)

        new_paths, n_ok = [], 0
        for p in df["path"]:
            src = Path(p)
            dst = out_root / src.name
            img = cv2.imread(str(src))
            if img is None:
                new_paths.append(str(dst))
                continue
            cv2.imwrite(str(dst), silhouette_from_green(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
            new_paths.append(str(dst))
            n_ok += 1

        out = df.copy()
        out["path"] = new_paths
        # Keep the ImageRecord-valid literal ('green'); silhouette is derived from
        # green. The "silhouette" identity is carried at run time by manifest_path +
        # `--hydra data.background=silhouette` (drives output naming / MLflow param),
        # which is NOT validated against the ImageRecord background literal.
        out["background"] = "green"
        man = Path(f"data/manifests/{sp}_silhouette_bg_manifest.csv")
        out.to_csv(man, index=False)
        print(
            f"[{sp}] wrote {n_ok}/{len(df)} silhouettes -> {out_root}/  |  manifest {man}  "
            f"(splits: {out['split'].value_counts().to_dict()})"
        )


if __name__ == "__main__":
    main()
