"""Stage-1 frontal extraction over full videos (sequential read = fast).

Per bat video: orientation-correct → detection gate → segmentation → pose →
apply the calibrated geometric frontal filter (imported from
``filter_frontal_faces``). Saves Stage-1 survivors (clean frame + mask overlay)
+ per-bat metrics CSV + a keep montage. These survivors then go to the Stage-2
visual pass.

Shard across GPUs with --shard-index/--shard-count.

  python scripts/build_frontal_dataset.py --n-videos 40 --stride 2 --device cuda:0 --shard-index 0 --shard-count 2 &
  python scripts/build_frontal_dataset.py --n-videos 40 --stride 2 --device cuda:1 --shard-index 1 --shard-count 2 &
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from filter_frontal_faces import decide, geometry, mask_metrics  # noqa: E402

from bat_preprocessing import YOLODetector, YOLOPoseEstimator, YOLOSegmenter  # noqa: E402
from bat_preprocessing.video_extractor import VideoExtractor  # noqa: E402

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
# Calibrated Stage-1 thresholds (validated on the 2 review bats).
T = {
    "eye_conf": 0.5, "angle_max": 28.0, "min_nose_drop": 0.25, "sym_max": 0.38,
    "area_lo": 0.02, "area_hi": 0.65, "largest_frac": 0.95, "solidity": 0.90, "elong_max": 1.55,
}


def montage(over_dir: Path, out: Path, cols: int = 7) -> None:
    ims = sorted(over_dir.glob("*.jpg"))
    if not ims:
        return
    tiles = []
    for p in ims:
        im = cv2.imread(str(p))
        im = cv2.resize(im, (170, int(170 * im.shape[0] / im.shape[1])))
        tiles.append(np.pad(im, ((2, 2), (2, 2), (0, 0)), constant_values=255))
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, th - t.shape[0]), (0, tw - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full((th, tw, 3), 255, np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(out), grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="data/raw/mauritius/video")
    ap.add_argument("--output", default="data/work/mauritius_frontal")
    ap.add_argument("--n-videos", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride", type=int, default=2, help="Read every Nth frame (sequential).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seg-weights", default="models/preprocessing/face_seg_mauritius_v2.pt")
    ap.add_argument("--det-weights", default="legacy/face_detection/chosen_model/best.pt")
    ap.add_argument("--pose-weights", default="models/preprocessing/face_pose.pt")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    vids = sorted(p for p in Path(args.videos_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)
    random.Random(args.seed).shuffle(vids)
    vids = vids[: args.n_videos][args.shard_index :: args.shard_count]
    out_root = Path(args.output)
    print(f"[shard {args.shard_index}/{args.shard_count}] {len(vids)} bats on {args.device}, stride={args.stride}")

    det = YOLODetector({"weights": args.det_weights, "device": args.device})
    seg = YOLOSegmenter({"weights": args.seg_weights, "device": args.device})
    pose = YOLOPoseEstimator({"weights": args.pose_weights, "device": args.device})

    for vi, video in enumerate(vids, 1):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"  !! cannot open {video.name}")
            continue
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0.0)
        rot = VideoExtractor._orientation_code(cap)
        bat = video.stem
        fdir = out_root / bat / "frames"
        odir = out_root / bat / "overlay"
        fdir.mkdir(parents=True, exist_ok=True)
        odir.mkdir(parents=True, exist_ok=True)
        rows = []
        idx = kept = 0
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            i = idx
            idx += 1
            if i % max(1, args.stride) != 0:
                continue
            if rot is not None:
                fr = cv2.rotate(fr, rot)
            if det.predict(fr) is None:
                continue
            s = seg.predict(fr)
            pr = pose.predict(fr)
            if s is None or pr is None or pr.keypoints.shape[0] < 3:
                continue
            h, w = fr.shape[:2]
            g = geometry(pr.keypoints)
            msk = s.mask
            if msk.shape[:2] != (h, w):
                msk = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
            m = mask_metrics(msk, (h, w))
            keep, reason = decide(g, m, T)
            if not keep:
                continue
            area = float((msk > 0).mean()) * 100.0
            name = f"{bat}__f{i:06d}__a{area:04.1f}.jpg"
            cv2.imwrite(str(fdir / name), fr)
            cc = np.zeros_like(fr)
            cc[msk > 0] = (0, 255, 0)
            cv2.imwrite(str(odir / name), cv2.addWeighted(fr, 0.6, cc, 0.4, 0))
            rows.append({"file": name, "reason": reason, **{k: round(v, 4) for k, v in g.items()}})
            kept += 1
        cap.release()
        if rows:
            with open(out_root / bat / "metrics.csv", "w", newline="") as f:
                wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wtr.writeheader()
                wtr.writerows(rows)
            montage(odir, out_root / bat / "_keep_montage.jpg")
        print(f"  [{vi}/{len(vids)}] {bat}: kept {kept} frontal frames")
    print(f"[shard {args.shard_index}] done → {out_root}")


if __name__ == "__main__":
    main()
