"""Build the 4 recognition variants from hand-picked clean frontal frames.

Species-agnostic. For each culled full-resolution frame: segmentation + pose →
eye-anchored align (mask-centred, straightened) → original / green / random /
circle variants, written into ``<out-root>/<variant>/``.

Two source layouts are supported:

* **mauritius** — per-bat subdirs ``<exp-dir>/frames/<bat>/*.jpg`` (frame names
  ``<vid>__f<frame>__a<area>.jpg``); output names are constructed as
  ``m--<bat>--<bat>.<frame>.jpg``. Seg = ``face_seg_mauritius_v2.pt``.
* **rousettus** — a flat dir ``<frames-root>/*.jpg`` whose names are already in
  recognition form ``r--<bat>--<vid>.<frame>.jpg``; the input filename is
  preserved on output. Seg = the legacy rousettus checkpoint.

Both species use the same eye-anchored alignment (pose on the mask-centred crop,
which is the distribution ``face_pose.pt`` was trained on) and the same
smoothed + feathered seg mask for the green/random cuts. ``--edge`` sets the
output crop size; the aligner re-samples the full-resolution source in a single
warp, so 224 and 320 are genuine re-samplings of the source pixels (not upscales
of a fixed intermediate). The per-run upscale fraction (crops whose source head
bbox is smaller than ``--edge``) is logged so we can tell whether a given
``--edge`` carries real detail.

Reuses the same package functions as ``VideoExtractor`` so results match the
pipeline. Frames are already orientation-corrected (no re-rotate).

Sample run for review (3 frames per bat + per-bat montages):

    python scripts/build_variants_from_frames.py --species rousettus --sample 3 \
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

SPECIES_DEFAULTS = {
    "mauritius": {
        "prefix": "m",
        "frames_root": "data/processed/mauritius_frontal/_experiment_16/frames",
        "seg_weights": "models/preprocessing/face_seg_mauritius_v2.pt",
        "layout": "subdirs",
    },
    "rousettus": {
        "prefix": "r",
        "frames_root": "data/raw/rousettus/video/frontal_video_frames/rousettus-09_09_2025",
        "seg_weights": (
            "legacy/rousesttus_segmentation_still/training_results/runs/"
            "segment/bat_face_seg/weights/best.pt"
        ),
        "layout": "flat",
    },
}


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


def collect_groups(
    frames_root: Path, prefix: str, layout: str, sample: int, name_style: str = "legacy"
) -> dict[str, list[tuple[Path, str]]]:
    """Group source frames by identity → list of (frame_path, output_name).

    ``subdirs`` (mauritius): identity = subdir name; frame id parsed from
    ``__f<n>__``.
    ``flat`` (rousettus): identity = the ``class`` token of the recognition-form
    filename ``<prefix>--<class>--<rest>``; the input filename is preserved.

    ``name_style`` controls the mauritius output name:

    ``legacy``   ``<prefix>--<id>--<id>.<frame>.jpg`` — what the existing dataset
                 uses. The identity appears twice, because for mauritius the
                 identity *is* the video stem and the id token was built from it.
    ``compact``  ``<prefix>--<id>--f<frame:06d>.jpg`` — same three-token grammar,
                 identity written once. Still globally unique because the class
                 token is in the name, which matters: several joins in this
                 project key on basename alone.

    Default is ``legacy`` so re-running this script cannot silently rename the
    frozen datasets.
    """
    groups: dict[str, list[tuple[Path, str]]] = {}
    if layout == "subdirs":
        for d in sorted(p for p in frames_root.iterdir() if p.is_dir()):
            stem = d.name
            items: list[tuple[Path, str]] = []
            for fp in sample_evenly(sorted(d.glob("*.jpg")), sample):
                tok = re.search(r"__f(\d+)__", fp.name)
                frame_id = int(tok.group(1)) if tok else len(items)
                name = (
                    f"{prefix}--{stem}--f{frame_id:06d}.jpg"
                    if name_style == "compact"
                    else f"{prefix}--{stem}--{stem}.{frame_id}.jpg"
                )
                items.append((fp, name))
            if items:
                groups[stem] = items
    elif layout == "flat":
        by_id: dict[str, list[Path]] = {}
        for fp in sorted(frames_root.glob("*.jpg")):
            parts = fp.name.split("--")
            if len(parts) < 3 or parts[0] != prefix:
                continue
            by_id.setdefault(parts[1], []).append(fp)
        for identity, frames in by_id.items():
            items = [(fp, fp.name) for fp in sample_evenly(sorted(frames), sample)]
            if items:
                groups[identity] = items
    else:
        raise ValueError(f"unknown layout: {layout}")
    return groups


def mask_side_px(mask: np.ndarray, frame_shape: tuple[int, int], margin: float) -> float | None:
    """Approximate the aligner's source-crop ``side`` (px) from the seg mask.

    Mirrors ``FaceAligner`` ``side = max_dim * (1 + 2*margin)`` on the mask bbox
    (measured at frame resolution). Used only to flag upscales; the aligner uses
    the rotated bbox, so this is a close proxy, not exact.
    """
    h, w = frame_shape
    m = mask
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return None
    max_dim = max(int(xs.max()) - int(xs.min()), int(ys.max()) - int(ys.min())) + 1
    return max_dim * (1.0 + 2.0 * margin)


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
    ap.add_argument("--species", choices=("mauritius", "rousettus"), default="mauritius")
    ap.add_argument("--frames-root", default="", help="Override the species default frames root.")
    ap.add_argument("--out-root", default="", help="Default: data/processed/<species>/video/not_augmented/<edge>")
    ap.add_argument("--montage-dir", default="", help="If set, write a per-bat variant montage.")
    ap.add_argument("--sample", type=int, default=0, help="Frames per bat (0 = all).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seg-weights", default="", help="Override the species default seg checkpoint.")
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
    ap.add_argument(
        "--eye-align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate to level the eyes. --no-eye-align falls back to a mask-centred square crop.",
    )
    ap.add_argument("--pose-crop-margin", type=float, default=0.12)
    ap.add_argument(
        "--name-style",
        choices=("legacy", "compact"),
        default="legacy",
        help="compact writes <prefix>--<id>--f<frame>.jpg instead of repeating the "
        "identity twice. Default legacy so existing datasets are untouched.",
    )
    ap.add_argument(
        "--out-layout",
        choices=("flat", "by-identity"),
        default="flat",
        help="by-identity writes <variant>/<identity>/<file> instead of one flat "
        "directory per variant, so the frames can be reviewed a bat at a time. "
        "build_manifest walks recursively, so either layout loads.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    defaults = SPECIES_DEFAULTS[args.species]
    frames_root = Path(args.frames_root or defaults["frames_root"])
    seg_weights = args.seg_weights or defaults["seg_weights"]
    out = Path(
        args.out_root
        or f"data/processed/{args.species}/video/not_augmented/{args.edge}"
    )
    for v in VARIANTS:
        (out / v).mkdir(parents=True, exist_ok=True)
    montage_dir = Path(args.montage_dir) if args.montage_dir else None
    if montage_dir:
        montage_dir.mkdir(parents=True, exist_ok=True)

    groups = collect_groups(
        frames_root, defaults["prefix"], defaults["layout"], args.sample, args.name_style
    )
    if not groups:
        raise SystemExit(f"no frames found under {frames_root} (layout={defaults['layout']})")

    seg = YOLOSegmenter({"weights": seg_weights, "device": args.device})
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
        f"[{args.species} @ {args.edge}px] seg={Path(seg_weights).name} eye_align={args.eye_align} "
        f"| random_bg pool: {len(pool)} natural images" + ("" if pool else " (blur fallback)")
    )

    counts = dict.fromkeys(VARIANTS, 0)
    per_bat = {}
    n_aligned = 0
    n_upscale = 0
    for stem, items in sorted(groups.items()):
        rows = []
        w = 0
        for fp, name in items:
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
            # --no-eye-align → pose=None ⇒ mask-centroid crop, angle 0 (un-rotated).
            af = aligner.align_eye_anchored(fr, s, pr if args.eye_align else None)
            if af is None:
                continue
            side = mask_side_px(s.mask, fr.shape[:2], args.margin)
            if side is not None:
                n_aligned += 1
                if side < args.edge:
                    n_upscale += 1
            edge = af.image.shape[0]
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
                dest_dir = out / v / stem if args.out_layout == "by-identity" else out / v
                dest_dir.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(dest_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
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
    if n_aligned:
        pct = 100.0 * n_upscale / n_aligned
        print(
            f"upscale check @ {args.edge}px: {n_upscale}/{n_aligned} crops "
            f"({pct:.1f}%) have source head bbox < {args.edge}px (interpolated up)."
        )


if __name__ == "__main__":
    main()
