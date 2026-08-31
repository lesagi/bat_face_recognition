"""Stage-1 frontal extraction over full videos (sequential read = fast).

Per bat video: orientation-correct → detection gate → segmentation → pose →
apply the calibrated geometric frontal filter (imported from
``filter_frontal_faces``). Saves Stage-1 survivors (clean frame + mask overlay)
+ per-bat metrics CSV + a keep montage. These survivors then go to the Stage-2
visual pass.

``--keep-all`` retains the whole *reviewable superset* instead of only the
survivors: every frame that reached a usable detection + segmentation + pose is
written, and the geometric filter's verdict is recorded per frame rather than
used as a hard gate. This is what makes a curation decision reversible -- the
default mode deletes rejects on the floor, so "put that frame back" used to mean
re-extracting from video. In this mode review thumbnails (mask outline, 320px)
replace the full-resolution overlays, which would otherwise double the footprint
of a set that is ~4x larger to begin with.

``--videos`` pins an explicit identity list (comma-separated stems, or a JSON
file with a ``target`` array) instead of drawing ``--n-videos`` by ``--seed``.

Shard across GPUs with --shard-index/--shard-count.

  python scripts/build_frontal_dataset.py --n-videos 40 --stride 2 --device cuda:0 --shard-index 0 --shard-count 2 &
  python scripts/build_frontal_dataset.py --n-videos 40 --stride 2 --device cuda:1 --shard-index 1 --shard-count 2 &
"""

from __future__ import annotations

import argparse
import csv
import json
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
    ap.add_argument("--videos", default=None,
                    help="Explicit identity list: comma-separated stems, or a JSON file "
                         "with a 'target' array. Overrides --n-videos/--seed.")
    ap.add_argument("--keep-all", action="store_true",
                    help="Retain every det+seg+pose survivor and record the frontal "
                         "verdict per frame instead of gating on it.")
    ap.add_argument("--thumb-edge", type=int, default=320,
                    help="Long edge of the review thumbnails written in --keep-all mode.")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    vids = sorted(p for p in Path(args.videos_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if args.videos:
        spec = Path(args.videos)
        if spec.suffix == ".json" and spec.exists():
            wanted = list(json.loads(spec.read_text())["target"])
        else:
            wanted = [s.strip() for s in args.videos.split(",") if s.strip()]
        by_stem = {v.stem: v for v in vids}
        missing = [w for w in wanted if w not in by_stem]
        if missing:
            raise SystemExit(f"no such video(s) under {args.videos_dir}: {missing}")
        vids = [by_stem[w] for w in wanted]
    else:
        random.Random(args.seed).shuffle(vids)
        vids = vids[: args.n_videos]
    vids = vids[args.shard_index :: args.shard_count]
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
        odir = out_root / bat / ("thumbs" if args.keep_all else "overlay")
        fdir.mkdir(parents=True, exist_ok=True)
        odir.mkdir(parents=True, exist_ok=True)
        rows = []
        idx = kept = seen = 0
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
            if not keep and not args.keep_all:
                continue
            seen += 1
            area = float((msk > 0).mean()) * 100.0
            name = f"{bat}__f{i:06d}__a{area:04.1f}.jpg"
            cv2.imwrite(str(fdir / name), fr)
            if args.keep_all:
                # A 320px thumb with the mask outlined is enough to judge a frame by,
                # and costs ~15 KB against ~480 KB for a full-resolution overlay.
                sc = args.thumb_edge / max(h, w)
                tim = cv2.resize(fr, (max(1, int(w * sc)), max(1, int(h * sc))))
                tmk = cv2.resize(msk, (tim.shape[1], tim.shape[0]), interpolation=cv2.INTER_NEAREST)
                cont, _ = cv2.findContours((tmk > 0).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(tim, cont, -1, (0, 255, 0), 2)
                cv2.imwrite(str(odir / name), tim, [cv2.IMWRITE_JPEG_QUALITY, 85])
            else:
                cc = np.zeros_like(fr)
                cc[msk > 0] = (0, 255, 0)
                cv2.imwrite(str(odir / name), cv2.addWeighted(fr, 0.6, cc, 0.4, 0))
            rows.append({"file": name, "reason": reason, "auto_keep": bool(keep),
                         "frame": i, "mask_area_pct": round(area, 2),
                         **{k: round(v, 4) for k, v in g.items()}})
            kept += bool(keep)
        cap.release()
        if rows:
            if args.keep_all:
                # JSON, not CSV: this is new output and the repo's rule is Parquet or JSON.
                (out_root / bat / "metrics.json").write_text(json.dumps(
                    {"identity": bat, "stride": args.stride, "thresholds": T,
                     "n_superset": len(rows), "n_auto_keep": kept, "frames": rows}, indent=1))
            else:
                with open(out_root / bat / "metrics.csv", "w", newline="") as f:
                    wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    wtr.writeheader()
                    wtr.writerows(rows)
                montage(odir, out_root / bat / "_keep_montage.jpg")
        if args.keep_all:
            print(f"  [{vi}/{len(vids)}] {bat}: superset {seen} frames "
                  f"({kept} pass the frontal gate)")
        else:
            print(f"  [{vi}/{len(vids)}] {bat}: kept {kept} frontal frames")
    print(f"[shard {args.shard_index}] done → {out_root}")


if __name__ == "__main__":
    main()
