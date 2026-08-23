"""Occlusion probe (causal control): does a model trained on real faces rely on the
crop *silhouette* or on *internal texture*?

From the original test split it builds two ablated test sets (using the green-derived
segmentation mask), keeping train/val untouched, and evaluates a trained checkpoint on
each:

  A  texture_removed  -> interior flat-filled with the mean face colour, silhouette kept
                        (the model sees SHAPE only)
  B  shape_removed    -> a FIXED circular aperture keeps interior texture, outline removed
                        (the model sees TEXTURE only, no variable silhouette)
  C  face_removed     -> the face is flat-filled with the mean background colour, scene kept
                        (the model sees the BACKGROUND only)

Comparing the ROC-AUC drop of A vs B on the same model shows which cue it relies on.
Absolute numbers are conservative (both arms are somewhat out-of-distribution); the
A-vs-B contrast is the informative part.

  conda activate frec && uv run python scripts/occlusion_probe.py --species mauritius \
      --checkpoint outputs/controls/occlusion_mauritius_arcface/best_model_roc_auc.pt
"""

from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np
import pandas as pd

APERTURE_FRAC = 0.34  # fixed aperture radius as a fraction of min(H,W)

# Face dilation for arm C, matched to scripts/probe_background_leakage.py so the
# two measurements describe the same "background" region. Calibrated there: at
# smaller values the build's 4 px mask feathering bleeds face colour outward and
# the green negative control stops returning chance.
FACE_DILATE_PX = 24


def mask_from_green(green_path: pathlib.Path, shape: tuple[int, int]) -> np.ndarray | None:
    g = cv2.imread(str(green_path))
    if g is None:
        return None
    if g.shape[:2] != shape:
        g = cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    b, gr, r = g[:, :, 0].astype(int), g[:, :, 1].astype(int), g[:, :, 2].astype(int)
    m = (~((gr > 150) & (r < 120) & (b < 120))).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def make_A(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Texture removed, silhouette kept: flat-fill the face interior."""
    out = img.copy()
    face = mask.astype(bool)
    if face.sum() == 0:
        return out
    out[face] = img[face].mean(axis=0).astype(img.dtype)
    return out


def make_B(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shape removed, texture kept: fixed circular aperture over the face centre."""
    h, w = img.shape[:2]
    ys, xs = np.where(mask > 0)
    cy, cx = (int(ys.mean()), int(xs.mean())) if len(xs) else (h // 2, w // 2)
    ap = np.zeros((h, w), np.uint8)
    cv2.circle(ap, (cx, cy), int(APERTURE_FRAC * min(h, w)), 1, -1)
    out = np.zeros_like(img)
    out[ap > 0] = img[ap > 0]
    return out


def make_C(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Face removed, background kept: does the model score pairs on the scene?

    The inverse of A and B. Every bat was filmed in one video, so a same-bat pair
    always shares a background while a different-bat pair never does — the
    verification task is partly solvable from the scene alone. A model-free probe
    (``scripts/probe_background_leakage.py``) shows that cue is worth ROC-AUC
    0.74-0.79 on the original background. This arm asks the complementary
    question: does the *trained model* actually exploit it?

    The face is dilated by ``FACE_DILATE_PX`` and flat-filled with the mean
    background colour. Dilation matches the probe's definition of "background",
    so the two measurements describe the same region; the flat fill keeps the
    removed area from re-introducing face texture at its rim.

    Caveat, stated because it bounds the interpretation: a filled region still
    has the *outline* of the face, so this arm is "background plus residual
    silhouette", not background alone. The model-free probe is the cleaner
    measure of the pure background cue; read the two together.
    """
    out = img.copy()
    face = mask.astype(np.uint8)
    if FACE_DILATE_PX > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (FACE_DILATE_PX * 2 + 1,) * 2
        )
        face = cv2.dilate(face, kernel)
    face_bool = face > 0
    background = ~face_bool
    if background.sum() < 128 or face_bool.sum() == 0:
        return out
    out[face_bool] = img[background].mean(axis=0).astype(img.dtype)
    return out


def build_variant(species: str, arm: str, fn) -> str:
    # Ablated test images are seed-independent; generate once and reuse.
    man = f"data/manifests/{species}_occl_{arm}_manifest.csv"
    if pathlib.Path(man).exists():
        return man
    src = pd.read_csv(f"data/manifests/{species}_original_bg_manifest.csv")
    outdir = pathlib.Path(f"data/processed/{species}/video/not_augmented/occl_{arm}")
    outdir.mkdir(parents=True, exist_ok=True)
    paths, n = [], 0
    for _, row in src.iterrows():
        p = pathlib.Path(row["path"])
        if row["split"] != "test":  # only the test split is ablated
            paths.append(row["path"])
            continue
        img = cv2.imread(str(p))
        mask = mask_from_green(pathlib.Path(str(p).replace("original_bg", "green_bg")), img.shape[:2]) if img is not None else None
        if img is None or mask is None:
            paths.append(row["path"])
            continue
        dst = outdir / p.name
        cv2.imwrite(str(dst), fn(img, mask), [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(str(dst))
        n += 1
    out = src.copy()
    out["path"] = paths
    man = f"data/manifests/{species}_occl_{arm}_manifest.csv"
    out.to_csv(man, index=False)
    print(f"[{species}/{arm}] ablated {n} test images -> {outdir}/  |  {man}")
    return man


def evaluate(species: str, checkpoint: str, manifest: str) -> float:
    from bat_cli.runtime import compose_config, run_evaluation

    cfg = compose_config(
        experiment=f"arcface_{species}_original_bg_video_tuned",
        overrides=[f"data.manifest_path={manifest}"],
    )
    rep = run_evaluation(cfg, checkpoint=pathlib.Path(checkpoint))
    return float(rep.verification.roc_auc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=42, help="Seed of the evaluated model (for the CSV).")
    ap.add_argument("--gen-only", action="store_true", help="Build ablated sets without evaluating.")
    args = ap.parse_args()

    clean = f"data/manifests/{args.species}_original_bg_manifest.csv"
    man_a = build_variant(args.species, "A", make_A)
    man_b = build_variant(args.species, "B", make_B)
    man_c = build_variant(args.species, "C", make_C)
    if args.gen_only:
        return

    rows = [
        ("clean", clean),
        ("texture_removed (shape only)", man_a),
        ("shape_removed (texture only)", man_b),
        ("face_removed (background only)", man_c),
    ]
    results = [(name, evaluate(args.species, args.checkpoint, man)) for name, man in rows]
    base = results[0][1]
    print(f"\n=== occlusion probe: {args.species} (ArcFace original, seed {args.seed}) ===")
    for name, v in results:
        print(f"  {name:32s} ROC-AUC={v:.3f}   delta_vs_clean={v - base:+.3f}")

    out = pathlib.Path("outputs/controls/occlusion_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a") as f:
        if write_header:
            f.write("species,seed,arm,roc_auc,delta_vs_clean\n")
        for name, v in results:
            f.write(f"{args.species},{args.seed},{name},{v:.4f},{v - base:.4f}\n")
    print(f"appended {out}")


if __name__ == "__main__":
    main()
