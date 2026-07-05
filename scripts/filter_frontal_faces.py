"""Stage-1 geometric frontal-face filter (deterministic, GPU).

For each frame, run pose + segmentation and decide KEEP/REJECT from geometry
alone — no LLM. Criteria (pose model mapping: idx0=left_eye, idx1=right_eye,
idx2=nose, confirmed by the user):

  Frontal : both eyes confident; nose horizontally centred between the eyes
            (|nose_x - eye_mid_x| / (interocular/2) small); nose sits clearly
            below the eye line (vertical drop / interocular >= min).
  Upright : |eye-line angle| <= angle_max; nose below the eye-midpoint.
  Head-only mask : one dominant connected component; area within a band;
            solid (mask area / convex-hull area high).

Modes:
  * calibration — point --frames-dir at a dir of pre-extracted frames; pass
    --keeps / --negatives to print precision/recall + the flagged rows.
  * emit — writes <out>.csv (all metrics+verdict) and a montage of KEEPs.

Usage:
  python scripts/filter_frontal_faces.py --frames-dir data/processed/mauritius_seg_review \
      --recursive --keeps data/processed/mauritius_seg_review/_chunks/_all_keeps.txt \
      --negatives 20230831_163856__f000177,20230831_163856__f000261,20230831_163856__f001110 \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from bat_preprocessing import YOLOPoseEstimator, YOLOSegmenter

LEFT_EYE, RIGHT_EYE, NOSE = 0, 1, 2


def mask_metrics(mask: np.ndarray, hw: tuple[int, int]) -> dict:
    h, w = hw
    binary = (mask > 0).astype(np.uint8)
    area = int(binary.sum())
    if area == 0:
        return {"area_frac": 0.0, "largest_frac": 0.0, "solidity": 0.0, "border": True, "n_comp": 0}
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comp_areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([area])
    largest = int(comp_areas.max())
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(cnts, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(big)) or 1.0
    ys, xs = np.where(binary > 0)
    border = bool(xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2)
    # Rotation-invariant elongation: min-area rect long/short side. Frontal bat
    # heads are compact (~1.1-1.4); profiles elongate as the snout juts sideways.
    (rw, rh) = cv2.minAreaRect(big)[1]
    elong = (max(rw, rh) / min(rw, rh)) if min(rw, rh) > 0 else 99.0
    return {
        "area_frac": area / (h * w),
        "largest_frac": largest / area,
        "solidity": cv2.contourArea(big) / hull_area,
        "elong": elong,
        "border": border,
        "n_comp": int(n - 1),
    }


def geometry(kpts: np.ndarray) -> dict:
    le, re, no = kpts[LEFT_EYE], kpts[RIGHT_EYE], kpts[NOSE]
    inter = math.hypot(float(re[0] - le[0]), float(re[1] - le[1])) or 1e-6
    eye_mid = ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
    angle = math.degrees(math.atan2(float(re[1] - le[1]), float(re[0] - le[0])))
    nose_x_off = abs(float(no[0]) - eye_mid[0]) / (inter / 2.0)  # 0 centred, 1 under an eye
    nose_drop = (float(no[1]) - eye_mid[1]) / inter  # >0 nose below eyes
    return {
        "le_c": float(le[2]), "re_c": float(re[2]), "no_c": float(no[2]),
        "eye_angle": angle, "interocular": inter,
        "nose_x_off": nose_x_off, "nose_drop": nose_drop,
    }


def decide(g: dict, m: dict, t: dict) -> tuple[bool, str]:
    if g["le_c"] < t["eye_conf"] or g["re_c"] < t["eye_conf"]:
        return False, "eye_conf"
    if abs(g["eye_angle"]) > t["angle_max"]:
        return False, "tilt"
    if g["nose_drop"] < t["min_nose_drop"]:
        return False, "nose_not_below"
    if g["nose_x_off"] > t["sym_max"]:
        return False, "asymmetric"  # profile / three-quarter
    if not (t["area_lo"] <= m["area_frac"] <= t["area_hi"]):
        return False, "mask_area"
    if m["largest_frac"] < t["largest_frac"]:
        return False, "mask_fragmented"
    if m["solidity"] < t["solidity"]:
        return False, "mask_not_solid"
    if m["elong"] > t["elong_max"]:
        return False, "elongated_profile"
    return True, "keep"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seg-weights", default="models/preprocessing/face_seg_mauritius_v2.pt")
    ap.add_argument("--pose-weights", default="models/preprocessing/face_pose.pt")
    ap.add_argument("--out", default="data/processed/mauritius_seg_review/_stage1_metrics.csv")
    ap.add_argument("--keeps", default="")
    ap.add_argument("--negatives", default="")
    # thresholds
    ap.add_argument("--eye-conf", type=float, default=0.5)
    ap.add_argument("--angle-max", type=float, default=10.0)
    ap.add_argument("--min-nose-drop", type=float, default=0.20)
    ap.add_argument("--sym-max", type=float, default=0.55)
    ap.add_argument("--area-lo", type=float, default=0.02)
    ap.add_argument("--area-hi", type=float, default=0.65)
    ap.add_argument("--largest-frac", type=float, default=0.85)
    ap.add_argument("--solidity", type=float, default=0.80)
    ap.add_argument("--elong-max", type=float, default=1.55)
    args = ap.parse_args()
    t = {
        "eye_conf": args.eye_conf, "angle_max": args.angle_max,
        "min_nose_drop": args.min_nose_drop, "sym_max": args.sym_max,
        "area_lo": args.area_lo, "area_hi": args.area_hi,
        "largest_frac": args.largest_frac, "solidity": args.solidity,
        "elong_max": args.elong_max,
    }

    root = Path(args.frames_dir)
    globber = root.rglob if args.recursive else root.glob
    frames = sorted(
        p for p in globber("*__f*__*.jpg")
        if "_overlay" not in p.parts and "frontal" not in p.parts and "yolo_dataset" not in p.parts
    )
    print(f"{len(frames)} frames on {args.device}")

    seg = YOLOSegmenter({"weights": args.seg_weights, "device": args.device})
    pose = YOLOPoseEstimator({"weights": args.pose_weights, "device": args.device})

    rows = []
    for p in frames:
        fr = cv2.imread(str(p))
        if fr is None:
            continue
        h, w = fr.shape[:2]
        s = seg.predict(fr)
        pr = pose.predict(fr)
        row = {"file": p.name, "keep": False, "reason": "no_pose_or_seg"}
        if pr is not None and s is not None and pr.keypoints.shape[0] >= 3:
            g = geometry(pr.keypoints)
            msk = s.mask
            if msk.shape[:2] != (h, w):
                msk = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
            m = mask_metrics(msk, (h, w))
            keep, reason = decide(g, m, t)
            row = {"file": p.name, "keep": keep, "reason": reason,
                   **{k: round(v, 4) for k, v in g.items()},
                   **{f"mask_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}}
        rows.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["file", "keep", "reason"] + [c for c in cols if c not in ("file", "keep", "reason")])
        wtr.writeheader()
        wtr.writerows(rows)
    kept = [r for r in rows if r["keep"]]
    print(f"KEEP {len(kept)} / {len(rows)}  → {args.out}")

    # calibration report
    if args.keeps and Path(args.keeps).exists():
        human = {ln.strip() for ln in Path(args.keeps).read_text().splitlines() if ln.strip()}
        negs = {n for n in args.negatives.split(",") if n}
        filt_keep = {r["file"] for r in kept}
        tp = len(filt_keep & human)
        fp = len(filt_keep - human)
        fn = len(human - filt_keep)
        print(f"vs human keeps: TP={tp} FP={fp} FN={fn} "
              f"precision={tp/(tp+fp+1e-9):.2f} recall={tp/(tp+fn+1e-9):.2f}")
        print("flagged negatives (must be REJECT):")
        for r in rows:
            if any(n in r["file"] for n in negs):
                print(f"  {r['file']}: keep={r['keep']} reason={r['reason']} "
                      f"angle={r.get('eye_angle')} nose_x_off={r.get('nose_x_off')} "
                      f"nose_drop={r.get('nose_drop')} eyeC=({r.get('le_c')},{r.get('re_c')})")

    (Path(args.out).with_suffix(".keep.json")).write_text(
        json.dumps(sorted(r["file"] for r in kept), indent=2)
    )


if __name__ == "__main__":
    main()
