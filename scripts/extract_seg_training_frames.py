"""Cut candidate frames from the mauritius videos for seg-model retraining.

Runs the face DETECTION model over each video (orientation-corrected — these
clips are portrait stored landscape with a 90° flag), keeps the most confident
detections spread across the clip, and writes per-bat:

  <out>/<video_stem>/<video_stem>__f{frame}__c{conf}.jpg   full upright frames
  <out>/<video_stem>/_contactsheet.jpg                      labelled grid to pick from

You then pick 1-2 frontal frames per bat to annotate + retrain segmentation.

Parallelise across GPUs with --shard-index / --shard-count (one process per GPU).

    python scripts/extract_seg_training_frames.py --device cuda:0 --shard-index 0 --shard-count 2 &
    python scripts/extract_seg_training_frames.py --device cuda:1 --shard-index 1 --shard-count 2 &
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from bat_preprocessing import YOLODetector
from bat_preprocessing.video_extractor import VideoExtractor

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def _select(cands: list[tuple[int, float]], k: int, min_gap: int) -> list[tuple[int, float]]:
    """Greedily pick up to k highest-confidence frames, keeping a min frame gap."""
    chosen: list[tuple[int, float]] = []
    for idx, conf in sorted(cands, key=lambda c: c[1], reverse=True):
        if all(abs(idx - c[0]) >= min_gap for c in chosen):
            chosen.append((idx, conf))
        if len(chosen) >= k:
            break
    return sorted(chosen)


def _contact_sheet(thumbs: list[np.ndarray], cols: int = 4) -> np.ndarray:
    if not thumbs:
        return np.full((64, 64, 3), 40, np.uint8)
    rows = (len(thumbs) + cols - 1) // cols
    while len(thumbs) < rows * cols:
        thumbs.append(np.full_like(thumbs[0], 40))
    grid = [np.hstack(thumbs[r * cols : (r + 1) * cols]) for r in range(rows)]
    return np.vstack(grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="data/raw/mauritius/m_videos")
    ap.add_argument("--output", default="data/processed/mauritius_seg_train_candidates")
    ap.add_argument("--detector-weights", default="legacy/face_detection/chosen_model/best.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.4, help="Min detection confidence.")
    ap.add_argument("--per-video", type=int, default=16, help="Candidate frames kept per bat.")
    ap.add_argument("--oversample", type=int, default=40, help="Candidates scanned ~per-video x this.")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--only", default="", help="Comma-separated video stems (overrides sharding).")
    args = ap.parse_args()

    videos = sorted(p for p in Path(args.videos_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        videos = [v for v in videos if v.stem in want]
    else:
        videos = videos[args.shard_index :: args.shard_count]
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[shard {args.shard_index}/{args.shard_count}] {len(videos)} videos on {args.device}")
    detector = YOLODetector({"weights": args.detector_weights, "device": args.device})

    for vi, video in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0.0)
        rot = VideoExtractor._orientation_code(cap)
        stride = max(1, total // max(1, args.per_video * args.oversample))

        # Pass 1: scan for detection-positive frames — store only (idx, conf)
        # (keeping every full frame in memory bloats to GBs on busy clips).
        cands: list[tuple[int, float]] = []
        i = 0
        while i < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            idx = i
            i += stride
            if not ok or frame is None:
                continue
            if rot is not None:
                frame = cv2.rotate(frame, rot)
            d = detector.predict(frame, confidence_threshold=args.conf)
            if d is not None:
                cands.append((idx, d.confidence))

        picks = _select(cands, args.per_video, min_gap=max(1, total // (args.per_video * 4)))
        vdir = out_root / video.stem
        vdir.mkdir(parents=True, exist_ok=True)
        # Pass 2: re-read only the selected frames.
        thumbs = []
        for idx, conf in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            if rot is not None:
                fr = cv2.rotate(fr, rot)
            cv2.imwrite(str(vdir / f"{video.stem}__f{idx:06d}__c{conf:.2f}.jpg"), fr)
            d = detector.predict(fr, confidence_threshold=args.conf)
            t = fr.copy()
            if d is not None:
                x1, y1, x2, y2 = d.pixel_box(*fr.shape[:2])
                cv2.rectangle(t, (x1, y1), (x2, y2), (0, 255, 0), 6)
            t = cv2.resize(t, (300, int(300 * t.shape[0] / t.shape[1])))
            cv2.putText(t, f"f{idx} c{conf:.2f}", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            thumbs.append(np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=255))
        cap.release()
        if thumbs:
            cv2.imwrite(str(vdir / "_contactsheet.jpg"), _contact_sheet(thumbs))
        print(f"  [{vi}/{len(videos)}] {video.stem}: {len(cands)} det+ → kept {len(picks)}")
    print(f"[shard {args.shard_index}] done → {out_root}")


if __name__ == "__main__":
    main()
