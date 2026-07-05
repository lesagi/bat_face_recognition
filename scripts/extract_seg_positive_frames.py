"""Show where the segmentation model identifies a face, per bat (for review).

Runs the seg model over orientation-corrected frames of the given videos, keeps
the frames where it produces a mask, and writes per bat:

  <out>/<stem>/<stem>__f{frame}__a{area}.jpg   clean upright frames (seg-positive)
  <out>/<stem>/_seg_overlay_sheet.jpg          grid with the mask overlaid (review)

You review the overlay sheet to judge seg coverage/quality; the clean frames are
then ready to feed the pipeline.

    python scripts/extract_seg_positive_frames.py --videos 20230831_163856,20230805_190452 --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from bat_preprocessing import YOLOSegmenter
from bat_preprocessing.video_extractor import VideoExtractor


def _overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cc = np.zeros_like(img)
    cc[mask > 0] = (0, 255, 0)
    return cv2.addWeighted(img, 0.6, cc, 0.4, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="data/raw/mauritius/m_videos")
    ap.add_argument("--videos", required=True, help="Comma-separated video stems (the bats).")
    ap.add_argument("--output", default="data/processed/mauritius_seg_review")
    ap.add_argument("--seg-weights", default="models/preprocessing/face_seg_mauritius_v2.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-candidates", type=int, default=160, help="Frames sampled per video.")
    ap.add_argument("--min-area", type=float, default=2.0, help="Min mask %% of frame to keep.")
    ap.add_argument("--max-area", type=float, default=70.0)
    ap.add_argument("--sheet-cols", type=int, default=10)
    args = ap.parse_args()

    stems = [s.strip() for s in args.videos.split(",") if s.strip()]
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    seg = YOLOSegmenter({"weights": args.seg_weights, "device": args.device, "confidence_threshold": args.conf})

    for stem in stems:
        vp = Path(args.videos_dir) / f"{stem}.mp4"
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"!! cannot open {vp}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0.0)
        rot = VideoExtractor._orientation_code(cap)
        stride = max(1, total // max(1, args.max_candidates))
        vdir = out_root / stem
        vdir.mkdir(parents=True, exist_ok=True)

        thumbs: list[np.ndarray] = []
        kept = scanned = 0
        i = 0
        while i < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            idx = i
            i += stride
            if not ok or fr is None:
                continue
            if rot is not None:
                fr = cv2.rotate(fr, rot)
            scanned += 1
            s = seg.predict(fr)
            if s is None:
                continue
            mask = s.mask
            if mask.shape[:2] != fr.shape[:2]:
                mask = cv2.resize(mask, (fr.shape[1], fr.shape[0]), interpolation=cv2.INTER_NEAREST)
            area = float((mask > 0).mean()) * 100.0
            if not (args.min_area <= area <= args.max_area):
                continue
            kept += 1
            name = f"{stem}__f{idx:06d}__a{area:04.1f}.jpg"
            cv2.imwrite(str(vdir / name), fr)
            # individual mask overlay (resized) for per-frame review by a subagent
            ov = _overlay(fr, mask)
            ovr = cv2.resize(ov, (512, int(512 * fr.shape[0] / fr.shape[1])))
            (vdir / "_overlay").mkdir(exist_ok=True)
            cv2.imwrite(str(vdir / "_overlay" / name), ovr)
            t = _overlay(fr, mask)
            t = cv2.resize(t, (200, int(200 * fr.shape[0] / fr.shape[1])))
            cv2.putText(t, f"f{idx} {area:.0f}%", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            thumbs.append(np.pad(t, ((3, 3), (3, 3), (0, 0)), constant_values=255))
        cap.release()

        if thumbs:
            cols = args.sheet_cols
            rows = (len(thumbs) + cols - 1) // cols
            while len(thumbs) < rows * cols:
                thumbs.append(np.full_like(thumbs[0], 255))
            sheet = np.vstack([np.hstack(thumbs[r * cols : (r + 1) * cols]) for r in range(rows)])
            cv2.imwrite(str(vdir / "_seg_overlay_sheet.jpg"), sheet)
        print(f"{stem}: scanned {scanned} → seg-positive {kept}")
    print(f"done → {out_root}")


if __name__ == "__main__":
    main()
