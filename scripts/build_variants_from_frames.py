"""Build the 4 recognition variants from hand-picked clean frontal frames.

Input: the culled frames under ``<exp-dir>/frames/<bat>/*.jpg`` (full
orientation-corrected frames that survived Stage-1 geometry + the user's
manual review). For each frame: segmentation + pose → eye-anchored align
(mask-centred, straightened) → original / green / random / circle variants,
written with the rousettus naming ``m--{id}--{id}.{frame}.jpg`` into
``data/processed/mauritius/video/not_augmented/<variant>/``.

The face_ellipse variant is a perfect circle; green/random cuts use a
smoothed + feathered seg mask (no hard jagged corners).

Reuses the same package functions as ``VideoExtractor`` so results match the
pipeline. Frames are already orientation-corrected (no re-rotate).

Sample run for review (3 frames per bat + per-bat montages):

    python scripts/build_variants_from_frames.py --sample 3 \
        --out-root <preview-dir> --montage-dir <preview-dir>/_montages
"""

from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from bat_preprocessing import (
    BackgroundGenerator,
    FaceAligner,
    YOLOPoseEstimator,
    YOLOSegmenter,
    create_blur_image,
    get_random_cropped_image,
    replace_background,
)

VARIANTS = ("original_bg", "green_bg", "random_bg", "face_ellipse")


def random_pool(edge: int, n: int) -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    for _ in range(n):
        try:
            pool.append(get_random_cropped_image(edge, edge, fallback_to_generated=False))
        except Exception:
            break
    return pool


def sample_evenly(items: list[Path], n: int) -> list[Path]:
    if n <= 0 or n >= len(items):
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def pose_on_mask_crop(pose, frame, seg_mask, margin: float = 0.12):
    """Run pose on a mask-centred square crop, keypoints mapped back to frame.

    The pose model was trained on tight square face crops (face fills the
    canvas), not full frames — cropping to the seg mask puts inference back
    in that distribution. Returns a frame-coordinate PosePrediction or None.
    """
    h, w = frame.shape[:2]
    m = seg_mask
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return None
    cx = (float(xs.min()) + float(xs.max())) / 2.0
    cy = (float(ys.min()) + float(ys.max())) / 2.0
    side = max(int(xs.max()) - int(xs.min()), int(ys.max()) - int(ys.min()))
    side = int(round(side * (1.0 + 2.0 * margin)))
    side = min(max(side, 32), h, w)
    x0 = max(0, min(int(round(cx - side / 2.0)), w - side))
    y0 = max(0, min(int(round(cy - side / 2.0)), h - side))
    pr = pose.predict(frame[y0 : y0 + side, x0 : x0 + side])
    if pr is None:
        return None
    kpts = pr.keypoints.copy()
    kpts[:, 0] += x0
    kpts[:, 1] += y0
    return replace(pr, keypoints=kpts, original_image_shape=(h, w))


def draw_pose(image: np.ndarray, kpts: np.ndarray) -> np.ndarray:
    """Paint the pose keypoints + a horizontal line through the eye midpoint.

    Levelled eyes sit exactly on the white line; the dots show where the pose
    model actually put each landmark (0=left eye red, 1=right eye blue,
    2=nose yellow).
    """
    vis = image.copy()
    colors = ((0, 0, 255), (255, 0, 0), (0, 255, 255))
    if kpts is None or len(kpts) < 2:
        return vis
    eye_y = int(round(float(kpts[:2, 1].mean())))
    cv2.line(vis, (0, eye_y), (vis.shape[1], eye_y), (255, 255, 255), 1)
    for i, (x, y, c) in enumerate(kpts[:3]):
        color = colors[i % 3]
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, color, -1)
        cv2.putText(
            vis,
            f"{i}:{c:.2f}",
            (int(x) + 6, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def montage_row(images: list[np.ndarray], gap: int = 4) -> np.ndarray:
    tiles = [np.pad(im, ((gap, gap), (gap, gap), (0, 0)), constant_values=255) for im in images]
    return np.hstack(tiles)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="data/processed/mauritius_frontal/_experiment_16")
    ap.add_argument("--out-root", default="data/processed/mauritius/video/not_augmented")
    ap.add_argument("--montage-dir", default="", help="If set, write a per-bat variant montage.")
    ap.add_argument("--sample", type=int, default=0, help="Frames per bat (0 = all).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seg-weights", default="models/preprocessing/face_seg_mauritius_v2.pt")
    ap.add_argument("--pose-weights", default="models/preprocessing/face_pose.pt")
    ap.add_argument("--edge", type=int, default=224)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--smooth", type=int, default=5, help="Mask contour-rounding radius (px).")
    ap.add_argument("--feather", type=int, default=4, help="Mask edge-feather radius (px).")
    ap.add_argument(
        "--nose-roll-weight",
        type=float,
        default=0.0,
        help="0 = eyes perfectly level, 1 = nose directly below the eye midpoint.",
    )
    ap.add_argument(
        "--pose-on-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run pose on the mask-centred square crop (matches its training data).",
    )
    ap.add_argument("--pose-crop-margin", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frames_root = Path(args.exp_dir) / "frames"
    out = Path(args.out_root)
    for v in VARIANTS:
        (out / v).mkdir(parents=True, exist_ok=True)
    montage_dir = Path(args.montage_dir) if args.montage_dir else None
    if montage_dir:
        montage_dir.mkdir(parents=True, exist_ok=True)

    seg = YOLOSegmenter({"weights": args.seg_weights, "device": args.device})
    pose = YOLOPoseEstimator({"weights": args.pose_weights, "device": args.device})
    aligner = FaceAligner(
        edge_length=args.edge,
        margin_ratio=args.margin,
        mode="eye_anchored",
        min_keypoint_confidence=0.35,
        require_confident_eyes=False,  # frames already vetted
        nose_roll_weight=args.nose_roll_weight,
    )
    rng = np.random.RandomState(args.seed)
    pool = random_pool(args.edge, 48)
    print(
        f"random_bg pool: {len(pool)} natural images" + ("" if pool else " (using blur fallback)")
    )

    bats = sorted(d for d in frames_root.iterdir() if d.is_dir())
    counts = dict.fromkeys(VARIANTS, 0)
    per_bat = {}
    for d in bats:
        stem = d.name
        frames = sample_evenly(sorted(d.glob("*.jpg")), args.sample)
        rows = []
        w = 0
        for fp in frames:
            fr = cv2.imread(str(fp))
            if fr is None:
                continue
            s = seg.predict(fr)
            if s is None:
                continue
            if args.pose_on_crop:
                pr = pose_on_mask_crop(pose, fr, s.mask, args.pose_crop_margin)
                if pr is None:
                    pr = pose.predict(fr)  # fall back to full-frame pose
            else:
                pr = pose.predict(fr)
            af = aligner.align_eye_anchored(fr, s, pr)
            if af is None:
                continue
            edge = af.image.shape[0]
            tok = re.search(r"__f(\d+)__", fp.name)
            frame_id = int(tok.group(1)) if tok else w
            name = f"m--{stem}--{stem}.{frame_id}.jpg"
            variants = {"original_bg": af.image}
            variants["green_bg"] = replace_background(
                af.image,
                af.mask,
                BackgroundGenerator.solid_color,
                smooth=args.smooth,
                feather=args.feather,
                color=(0, 255, 0),
            )
            bg = pool[int(rng.randint(len(pool)))] if pool else create_blur_image(edge, edge)
            variants["random_bg"] = replace_background(
                af.image, af.mask, bg, smooth=args.smooth, feather=args.feather
            )
            circle = aligner.elliptical_face_mask(af.mask, af.keypoints, edge)
            variants["face_ellipse"] = replace_background(
                af.image,
                circle,
                BackgroundGenerator.solid_color,
                feather=args.feather,
                color=(0, 0, 0),
            )
            for v, img in variants.items():
                if cv2.imwrite(str(out / v / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    counts[v] += 1
            if montage_dir:
                labeled = draw_pose(af.image, af.keypoints)
                rows.append(montage_row([labeled] + [variants[v] for v in VARIANTS]))
            w += 1
        per_bat[stem] = w
        if montage_dir and rows:
            cv2.imwrite(str(montage_dir / f"{stem}.jpg"), np.vstack(rows))
        print(f"  {stem}: {w} frames → variants")
    print("\nper-variant totals:", counts)
    print("identities:", len(per_bat), "| total per variant:", counts["original_bg"])


if __name__ == "__main__":
    main()
