"""Visualise every stage of the mauritius 3-model extraction pipeline.

For each of N videos (same seeded selection as ``bat-cli extract-videos``),
sample a few frames and write a labelled "step strip" per frame showing:

    original (+detection box) | chip | chip+pose | chip+seg | aligned
    | green_bg | face_ellipse

so we can see exactly which stage misbehaves. Reuses the real pipeline code
(``VideoExtractor`` internals + ``FaceAligner``) so the viz matches production.

Usage (from repo root, in the frec env):
    python scripts/debug_extract_steps.py --n-videos 50 --frames-per-video 6 --device cuda:1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from bat_preprocessing import (
    BackgroundGenerator,
    VideoExtractionConfig,
    VideoExtractor,
    YOLODetector,
    YOLOPoseEstimator,
    YOLOSegmenter,
    replace_background,
)

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
TILE = 320
PAD = 6
LABEL_H = 22


def _tile(img: np.ndarray, label: str, note: str = "") -> np.ndarray:
    """Letterbox ``img`` into a TILE×TILE canvas with a label bar on top."""
    canvas = np.full((TILE, TILE, 3), 40, np.uint8)
    if img is not None and img.size:
        h, w = img.shape[:2]
        s = min(TILE / w, TILE / h)
        rw, rh = max(1, int(w * s)), max(1, int(h * s))
        resized = cv2.resize(img, (rw, rh))
        y0, x0 = (TILE - rh) // 2, (TILE - rw) // 2
        canvas[y0 : y0 + rh, x0 : x0 + rw] = resized
    bar = np.full((LABEL_H, TILE, 3), 0, np.uint8)
    cv2.putText(bar, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.vstack([bar, canvas])
    if note:
        cv2.putText(
            out, note, (4, LABEL_H + TILE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 255, 255), 1, cv2.LINE_AA,
        )
    return np.pad(out, ((PAD, PAD), (PAD, PAD), (0, 0)), constant_values=255)


def _missing(label: str, why: str) -> np.ndarray:
    out = _tile(np.full((TILE, TILE, 3), 30, np.uint8), label, "")
    cv2.putText(
        out, why, (PAD + 8, PAD + LABEL_H + TILE // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (0, 0, 255), 2, cv2.LINE_AA,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="data/raw/mauritius/m_videos")
    ap.add_argument("--output", default="data/processed/mauritius_aligned/_debug")
    ap.add_argument("--n-videos", type=int, default=50)
    ap.add_argument("--frames-per-video", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seg-weights", default="legacy/face_segmentation_mauritius/chosen_model/best.pt")
    ap.add_argument("--detector-weights", default="legacy/face_detection/chosen_model/best.pt")
    ap.add_argument(
        "--pose-weights",
        default="legacy/face_annotation_eyes_nose/runs/pose/augmented_train/weights/best.pt",
    )
    args = ap.parse_args()

    videos = sorted(p for p in Path(args.videos_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)
    random.Random(args.seed).shuffle(videos)
    videos = videos[: args.n_videos]
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading 3 models on {args.device} …")
    detector = YOLODetector({"weights": args.detector_weights, "device": args.device})
    segmenter = YOLOSegmenter({"weights": args.seg_weights, "device": args.device})
    pose = YOLOPoseEstimator({"weights": args.pose_weights, "device": args.device})
    cfg = VideoExtractionConfig(
        output_dir=out_root, identity="dbg", align_mode="eye_anchored", edge_length=224,
        margin_ratio=0.03, min_keypoint_confidence=0.35,
        require_confident_eyes=True, random_bg_style="blur",
    )
    ext = VideoExtractor(cfg, segmenter=segmenter, pose_estimator=pose, detector=detector)

    for vi, video in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0.0)
        rot_code = ext._orientation_code(cap)  # honour portrait rotation metadata
        vdir = out_root / video.stem
        vdir.mkdir(parents=True, exist_ok=True)

        # Scan the clip for detection-positive frames (the interesting ones),
        # then keep a spread of frames_per_video of them. Also a couple of
        # detection-NEGATIVE frames for context. Report the detection rate.
        stride = max(1, total // max(1, args.frames_per_video * 30))
        hits: list[tuple[int, np.ndarray]] = []
        misses: list[tuple[int, np.ndarray]] = []
        scanned = 0
        hit_count = 0
        fidx = 0
        while fidx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            this_idx = fidx
            fidx += stride
            if not ok or frame is None:
                continue
            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)
            scanned += 1
            try:
                has_det = ext._detector.predict(frame) is not None
            except Exception:
                has_det = False
            if has_det:
                hit_count += 1
                if len(hits) < args.frames_per_video * 8:
                    hits.append((this_idx, frame.copy()))
            elif len(misses) < 2:
                misses.append((this_idx, frame.copy()))

        chosen = hits
        if len(hits) > args.frames_per_video:
            pick = np.linspace(0, len(hits) - 1, args.frames_per_video).round().astype(int)
            chosen = [hits[i] for i in sorted({int(p) for p in pick})]
        chosen = chosen + misses  # a couple of no-detection frames for context

        for fno, frame in chosen:
            strip = np.hstack(_build_step_tiles(ext, frame))
            cv2.imwrite(str(vdir / f"frame_{fno:06d}_steps.jpg"), strip)
        det_rate = (hit_count / scanned * 100.0) if scanned else 0.0
        print(
            f"  [{vi}/{len(videos)}] {video.stem}: scanned~{scanned} det-rate~{det_rate:.0f}% "
            f"→ wrote {len(chosen)} strips"
        )
        cap.release()
    print(f"Done → {out_root}")


def _build_step_tiles(ext: VideoExtractor, frame: np.ndarray) -> list[np.ndarray]:
    """Full-frame pipeline: detection (gate) → seg(full) → pose(full) → align.

    Seg/pose are computed on the full frame (where the seg model is clean); the
    seg/pose tiles are *displayed* cropped to the detection box for visibility.
    """
    tiles: list[np.ndarray] = []
    h, w = frame.shape[:2]

    det = None
    try:
        det = ext._detector.predict(frame)
    except Exception as exc:  # noqa: BLE001
        print("    detection error:", exc)

    # 1) original + detection box
    vis = frame.copy()
    vcrop = (0, 0, w, h)
    if det is not None:
        x1, y1, x2, y2 = det.pixel_box(h, w)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 4)
        # display-only crop window around the box for the seg/pose tiles
        mx, my = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
        vcrop = (max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my))
    tiles.append(_tile(vis, "1 original+det", "" if det else "NO DETECTION"))
    if det is None:
        tiles += [_missing(f"{i} —", "gated out") for i in range(2, 7)]
        return tiles

    def _vc(img: np.ndarray) -> np.ndarray:
        a, b, c, d = vcrop
        return img[b:d, a:c]

    # 2) segmentation on the FULL frame (overlay, shown cropped to the box)
    seg_p = None
    try:
        seg_p = ext.segmenter.predict(frame)
    except Exception:
        seg_p = None
    if seg_p is None:
        tiles.append(_missing("2 seg(full)", "NO MASK"))
    else:
        m = seg_p.mask
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = frame.copy()
        red = np.zeros_like(frame)
        red[..., 2] = 255
        sel = m > 0
        overlay[sel] = (0.55 * overlay[sel] + 0.45 * red[sel]).astype(np.uint8)
        tiles.append(_tile(_vc(overlay), "2 seg(full)", f"area={float(sel.mean()) * 100:.0f}%"))

    # 3) pose on the FULL frame (keypoints, shown cropped to the box)
    pose_p = None
    try:
        pose_p = ext.pose_estimator.predict(frame)
    except Exception:
        pose_p = None
    pvis = frame.copy()
    note = "NO POSE"
    if pose_p is not None:
        kp = pose_p.keypoints
        names, colors = ["LE", "RE", "NO"], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in range(min(3, len(kp))):
            x, y, c = float(kp[i][0]), float(kp[i][1]), float(kp[i][2])
            cv2.circle(pvis, (int(x), int(y)), 8, colors[i], -1)
            cv2.putText(
                pvis, f"{names[i]}{c:.2f}", (int(x) + 6, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2, cv2.LINE_AA,
            )
        if len(kp) >= 2:
            dy, dx = float(kp[1][1] - kp[0][1]), float(kp[1][0] - kp[0][0])
            note = f"eye-angle={np.degrees(np.arctan2(dy, dx)):.0f}deg"
    tiles.append(_tile(_vc(pvis), "3 pose(full)", note))

    # 4/5/6) aligned + variants
    af = None
    if seg_p is not None:
        try:
            af = ext._aligner.align_eye_anchored(frame, seg_p, pose_p)
        except Exception:
            af = None
    if af is None:
        tiles += [_missing("4 aligned", "—"), _missing("5 green_bg", "—"), _missing("6 ellipse", "—")]
        return tiles
    tiles.append(_tile(af.image, "4 aligned", ""))
    green = replace_background(af.image, af.mask, BackgroundGenerator.solid_color, color=(0, 255, 0))
    tiles.append(_tile(green, "5 green_bg", ""))
    emask = ext._aligner.elliptical_face_mask(af.mask, af.keypoints, af.image.shape[0])
    ell = replace_background(af.image, emask, BackgroundGenerator.solid_color, color=(0, 0, 0))
    tiles.append(_tile(ell, "6 ellipse", ""))
    return tiles


if __name__ == "__main__":
    main()
