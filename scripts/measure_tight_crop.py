"""Measure crop geometry and head pose, and pick an output edge from data.

Written to size a `tight` (disc-bounding) crop. **It measured that crop out of the
design**, and the record of why is the point of keeping this script:

* A disc-bounding square must be centred on the mask, because eye-centred *framing*
  is a recorded failure and the keypoints straighten only. But the disc centroid sits
  below the mask centroid, so a mask-centred square of the disc-bounding side clips
  ~9-10% of the discs, and one grown to contain them is 0.887 / 0.919 of the head
  crop -- an 8-11% tightening, not the 24-29% the disc-bounding square suggests.
* Worse, it is *inconsistent*: 0.73-0.99 across images (p5-p95) against 0.081 spread
  for the disc-centred ideal. It adds scale variance rather than removing it.
* The cause is head **pitch**, not roll or yaw. `offset_y` correlates +0.888
  (mauritius) / +0.577 (rousettus) with `nose_drop`; residual roll after alignment is
  1.4-1.5 deg and the yaw proxy `nose_x_off` correlates < 0.18. A nodding head slides
  the eye/nose triangle down and the square grows to catch it.

That is what motivated `--crop eyes` in `build_variants_from_frames.py`: a similarity
alignment pinning the two eyes to fixed pixels removes the scale degree of freedom
outright, is defined by the eyes alone so pitch cannot disturb it, and equalises
anatomical face scale across species -- median interocular differs by 1.54x between
species against 1.82x for the head-crop side. Measured, it cuts between-bat framing
variation from 4.89 px to 2.11 px.

The script still earns its place: it produces the interocular and pose distributions
the canonical geometry was chosen against, and the per-image parquet those numbers
came from.

What it does:

1. Runs pose on the existing aligned crops and records ``f_disc`` (disc-bounding
   square / crop side), ``f_mask`` (the mask-centred square that contains the discs),
   the disc-centre offset, and the frontality geometry (`eye_angle`, `interocular`,
   `nose_x_off`, `nose_drop`). All are ratios within one crop, so they are
   scale-invariant and transfer to source pixels.
2. ``native_tight = native_side_px * f_mask``, joined on basename to
   ``outputs/quality/native_boxes.parquet``.
3. Picks the largest candidate edge at which at most ``--max-upscale`` of images in
   **either** species would need upscaling -- of both, because an edge comfortable for
   mauritius that upscales half of rousettus re-introduces the resolution gap.

Read-only: it measures existing crops and writes JSON plus a per-image parquet.

    uv run python scripts/measure_tight_crop.py --species mauritius rousettus
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from filter_frontal_faces import geometry  # noqa: E402
from species_saliency_maps import rois_from_keypoints  # noqa: E402

from bat_preprocessing import YOLOPoseEstimator  # noqa: E402

POSE_WEIGHTS = "models/preprocessing/face_pose.pt"
NATIVE_BOXES = pathlib.Path("outputs/quality/native_boxes.parquet")
OUT_JSON = pathlib.Path("outputs/quality/tight_crop_edge.json")
PER_IMAGE = pathlib.Path("outputs/quality/tight_crop_per_image.parquet")
# Measured on the green crops: the cleanest background for pose, and the one whose
# mask defines the face for every other variant of the same frame.
SOURCE = "data/processed/{sp}/video/not_augmented/base_320/green_bg"
CANDIDATE_EDGES = (128, 160, 192, 224, 320)


def disc_square(mask: np.ndarray) -> tuple[float, float, float] | None:
    """Side of the smallest square bounding the disc union, and the union's centre."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    w = float(xs.max() - xs.min() + 1)
    h = float(ys.max() - ys.min() + 1)
    return max(w, h), float(xs.min() + xs.max()) / 2.0, float(ys.min() + ys.max()) / 2.0


def outside_fraction(mask: np.ndarray, side: float) -> float:
    """Share of the disc union falling outside a *mask-centred* square of `side`.

    The image centre *is* the mask centre here, because the aligner already centres
    the crop on the segmentation mask.
    """
    h, w = mask.shape
    cy, cx = h / 2.0, w / 2.0
    half = side / 2.0
    keep = np.zeros_like(mask)
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    keep[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = True
    total = int(mask.sum())
    return 0.0 if total == 0 else float((mask & ~keep).sum()) / total


def mask_centred_side(mask: np.ndarray) -> float:
    """Smallest *mask-centred* square side that still contains the whole disc union.

    The disc-bounding square is centred on the discs, and the eyes and nose sit above
    the head's mask centroid -- so a mask-centred square of that same side clips them.
    Since the crop must stay mask-centred (eye-centred cropping is a recorded failure;
    keypoints straighten only), the honest construction is to keep the centre and grow
    the side until the discs fit. This returns that side.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0
    h, w = mask.shape
    cy, cx = h / 2.0, w / 2.0
    reach = max(np.abs(xs - cx).max(), np.abs(ys - cy).max())
    return float(2.0 * reach)


def measure(species: str, device: str, limit: int) -> dict:
    root = pathlib.Path(SOURCE.format(sp=species))
    if not root.is_dir():
        raise SystemExit(f"no source crops at {root}")
    pose = YOLOPoseEstimator({"weights": POSE_WEIGHTS, "device": device})

    imgs = sorted(root.rglob("*.jpg"))
    if limit:
        imgs = imgs[:limit]
    rows, n_pose_fail = [], 0
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            n_pose_fail += 1
            continue
        pr = pose.predict(img)
        kp = pr.keypoints if pr is not None else None
        rois = rois_from_keypoints(kp, img.shape[:2]) if pr is not None else None
        if rois is None:
            n_pose_fail += 1
            continue
        union = rois["eyes"] | rois["nose"]
        got = disc_square(union)
        if got is None:
            n_pose_fail += 1
            continue
        side, dcx, dcy = got
        w = float(img.shape[1])
        # Head pose in crop space. The aligner straightens ROLL (eye_angle ~ 0 here),
        # but nothing corrects YAW or PITCH -- and a yawed or pitched head deforms the
        # eye/nose triangle, which would move both the disc bounding box and its
        # offset. `nose_x_off` is the yaw proxy, `nose_drop` the pitch proxy, both
        # normalised by interocular distance so they are scale-free.
        g = geometry(kp)
        rows.append(
            {
                **{k: float(v) for k, v in g.items()},
                "basename": p.name,
                "identity": p.parent.name,
                # f_disc: the ideal tight square, centred on the discs themselves.
                # f_mask: what a mask-centred square must be to contain them. The
                # second is what gets built; the first is reported to show the cost.
                "f_disc": side / w,
                "f_mask": mask_centred_side(union) / w,
                "offset_x": (dcx - w / 2.0) / w,
                "offset_y": (dcy - float(img.shape[0]) / 2.0) / w,
                "outside_frac": outside_fraction(union, side),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"{species}: pose failed on every image")

    nb = pd.read_parquet(NATIVE_BOXES)
    nb = nb[nb.species == species][["basename", "native_side_px"]]
    df = df.merge(nb, on="basename", how="left")
    unjoined = int(df.native_side_px.isna().sum())
    df = df.dropna(subset=["native_side_px"])
    # The buildable crop is the mask-centred one, so it sets the native size.
    df["native_tight"] = df.native_side_px * df.f_mask
    # Per-image rows, so within-bat spread can be examined: a crop centred on a
    # per-BAT constant offset has no per-frame keypoint drift (the failure mode that
    # sank eye-centred cropping) while still following the animal's anatomy.
    PER_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    out_pq = PER_IMAGE.with_name(f"tight_crop_{species}.parquet")
    df.to_parquet(out_pq)

    def stats(s: pd.Series) -> dict:
        return {
            "mean": float(s.mean()),
            **{f"p{q}": float(s.quantile(q / 100)) for q in (5, 25, 50, 75, 95)},
        }

    return {
        "species": species,
        "n_measured": len(df),
        "n_pose_failed": n_pose_fail,
        "n_unjoined_to_native_boxes": unjoined,
        "f_disc": stats(df.f_disc),
        "f_mask": stats(df.f_mask),
        "offset_x": stats(df.offset_x),
        "offset_y": stats(df.offset_y),
        "native_tight": stats(df.native_tight),
        "outside_frac_if_disc_side_used": {
            "mean": float(df.outside_frac.mean()),
            "max": float(df.outside_frac.max()),
            "p95": float(df.outside_frac.quantile(0.95)),
        },
        "upscale_fraction_by_edge": {
            str(e): float((df.native_tight < e).mean()) for e in CANDIDATE_EDGES
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", nargs="+", default=["mauritius", "rousettus"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="Measure only the first N images.")
    ap.add_argument(
        "--max-upscale",
        type=float,
        default=0.05,
        help="Largest share of images allowed to need upscaling, in EITHER species.",
    )
    args = ap.parse_args()

    per = {sp: measure(sp, args.device, args.limit) for sp in args.species}

    # Largest edge tolerable for every species measured, not just the easy one.
    chosen, rejected = None, {}
    for edge in sorted(CANDIDATE_EDGES, reverse=True):
        worst = max(per[sp]["upscale_fraction_by_edge"][str(edge)] for sp in per)
        if worst <= args.max_upscale:
            chosen = edge
            break
        rejected[str(edge)] = worst

    out = {
        "candidate_edges": list(CANDIDATE_EDGES),
        "max_upscale_allowed": args.max_upscale,
        "chosen_edge": chosen,
        "rejected_edges_worst_upscale": rejected,
        "per_species": per,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))

    print(
        f"\n  candidates {CANDIDATE_EDGES}, cap {args.max_upscale:.0%} upscaled in either species\n"
    )
    for sp, r in per.items():
        print(
            f"  {sp:10s} n={r['n_measured']:5d} (pose fail {r['n_pose_failed']})  "
            f"native tight side median={r['native_tight']['p50']:.0f}px "
            f"(p5 {r['native_tight']['p5']:.0f})"
        )
        print(
            f"             crop side / head side: disc-centred {r['f_disc']['p50']:.3f}"
            f"  ->  mask-centred {r['f_mask']['p50']:.3f}  (the buildable one)"
        )
        print(
            f"             disc centre offset from mask centre: "
            f"x={r['offset_x']['p50']:+.3f} y={r['offset_y']['p50']:+.3f} of head side; "
            f"a disc-sided mask-centred square would clip "
            f"{r['outside_frac_if_disc_side_used']['mean']:.1%} of the discs"
        )
    print("\n  upscale fraction by edge:")
    print("    edge   " + "  ".join(f"{sp[:9]:>9s}" for sp in per))
    for e in CANDIDATE_EDGES:
        marks = "  ".join(f"{per[sp]['upscale_fraction_by_edge'][str(e)]:>8.1%} " for sp in per)
        flag = "  <- chosen" if e == chosen else ""
        print(f"    {e:<6d} {marks}{flag}")
    if chosen is None:
        print(
            f"\n  !! no candidate edge satisfies the cap; smallest tried was {min(CANDIDATE_EDGES)}"
        )
    print(f"\n  -> {OUT_JSON}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
