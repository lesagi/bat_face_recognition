"""Occlusion arms per background, with a colour-normalised second axis.

Extends the four-arm ROI-occlusion design of `build_roi_occlusion.py` (which stays
frozen: it is cited by `docs/occlusion_preregistration.md` and
`docs/occlusion_results.md` and must remain reproducible) along two axes and adds
two arms.

The arms
--------
``none``          clean, but through the same JPEG round-trip as the others so the
                  baseline is not sharper by one encode.
``centre``        the eye+nose discs destroyed (~24% of the crop).
``matched``       the same pixel COUNT destroyed, on the animal where possible,
                  furthest from the discs. The area control for `centre`.
``periphery``     everything except the discs destroyed (~76%).
``span``          everything except the CONVEX HULL of the discs destroyed. The
                  hull is the triangle of eye/eye/nose centres dilated by the disc
                  radius (the radii are equal, so the hull is exactly that), i.e.
                  the discs plus the space spanned between them.
``span_matched``  the discs kept plus the same extra area as `span` adds, placed
                  furthest from them instead of between them. The area control for
                  `span`: without it, `span` vs `periphery` measures how many
                  pixels survived, not where they were.

`span` - `periphery` isolates the inter-disc space; `span` - `span_matched` asks
whether its *location* matters rather than its size.

Why the colour axis is not optional
-----------------------------------
`docs/occlusion_results.md`: six global colour numbers (per-channel mean+std inside
the face matte) score 0.777 / 0.792 against trained models at 0.805 / 0.816 on the
same images. A global statistic cannot be removed by destroying any region -- mask
76% of the crop and the surviving 24% carries almost the same mean and standard
deviation. That is why the previous positive control could not have passed, and it
would not pass again. On the ``cn`` arms each channel is rescaled inside the face
matte to a fixed mean and standard deviation, so the six-number descriptor is
constant across bats by construction and a spatial ablation can finally mean
something.

Normalisation runs on the CLEAN image, before occlusion. Doing it after would
rescale the surviving region against the injected grey.

Backgrounds
-----------
`green` and `random` only. `original` is 93% solvable with the face deleted (arm
3C), so an occlusion result there would partly measure the background; and
`face_ellipse` already blacks out everything outside a fixed ellipse, so its
periphery is largely pre-removed.

ROIs are derived once per frame from the **green** image and reused for `random`.
Identical masks across backgrounds are required for the comparison to be about the
background rather than about geometry, and pose on a random natural background is
the failure mode `face_pose.pt` was never trained for. For the same reason the
`random` face matte is taken from the paired `green` image by basename.

    uv run python scripts/build_occlusion_arms.py --species mauritius --measure-only
    uv run python scripts/build_occlusion_arms.py --species mauritius --execute
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from species_saliency_maps import rois_from_keypoints  # noqa: E402

from bat_core.types import ImageRecord, Manifest  # noqa: E402
from bat_data import manifest_from_csv, manifest_to_csv  # noqa: E402
from bat_data.manifest import compute_quality  # noqa: E402
from bat_data.quality_metrics import face_mask_from_green  # noqa: E402
from bat_preprocessing import YOLOPoseEstimator  # noqa: E402

ARMS = ("none", "centre", "matched", "periphery", "span", "span_matched")
COLOURS = ("raw", "cn")
BACKGROUNDS = ("green", "random")

# Neutral grey, identical for every image and both species. The one property that
# matters is that it is not derived from the image: a per-image mean colour carries
# no spatial information but plenty of identity, which is what voided 160 runs
# (Amendment 2 of the pre-registration).
FILL_COLOUR = (128, 128, 128)
POSE_WEIGHTS = "models/preprocessing/face_pose.pt"
JPEG_QUALITY = 95

# Colour-normalisation targets, in 8-bit units. Arbitrary but fixed: what matters is
# that every image lands on the same six numbers, not which six.
CN_MEAN = 128.0
CN_STD = 40.0

# Declared bounds, as in the original build.
MAX_AREA_DIFF = 0.02
MAX_OVERLAP = 0.10
# Pre-declared gate: if the hull adds less than this over the discs it cannot be
# distinguished from `periphery` and the arm is dropped rather than reported.
MIN_SPAN_GAIN = 0.03

OUT_JSON = pathlib.Path("outputs/occlusion/occlusion_arms_build.json")


# --------------------------------------------------------------------- geometry


def hull_mask(discs: np.ndarray) -> np.ndarray:
    """Convex hull of the disc union.

    With equal eye and nose radii this is the triangle of centres dilated by the
    radius -- a rounded triangle -- so it is the discs plus exactly the space
    between them and nothing else.
    """
    pts = cv2.findNonZero(discs.astype(np.uint8))
    if pts is None:
        return np.zeros_like(discs)
    out = np.zeros(discs.shape, dtype=np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out.astype(bool)


def farthest_region(avoid: np.ndarray, face: np.ndarray | None, n: int) -> np.ndarray:
    """`n` pixels, on the animal if possible, as far from `avoid` as possible.

    Ranking rather than sampling, so the area is exact and the result deterministic.
    On-animal pixels outrank background by a large constant; within each group the
    tie-break is distance from `avoid`. Lifted from `build_roi_occlusion.matched_region`
    with the count made explicit, which is what `span_matched` needs.
    """
    h, w = avoid.shape
    n = int(min(max(n, 0), h * w - int(avoid.sum())))
    if n <= 0:
        return np.zeros_like(avoid)
    dist = cv2.distanceTransform((~avoid).astype(np.uint8), cv2.DIST_L2, 5)
    score = dist.astype(np.float64)
    if face is not None:
        score = score + face.astype(np.float64) * 1e4
    score[avoid] = -1.0
    idx = np.argpartition(score.ravel(), -n)[-n:]
    out = np.zeros(h * w, dtype=bool)
    out[idx] = True
    return out.reshape(h, w)


def destroy_masks(
    discs: np.ndarray, face: np.ndarray | None
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """The pixels each arm destroys, plus the realised area fractions."""
    total = float(discs.size)
    hull = hull_mask(discs)
    extra_n = int(hull.sum()) - int(discs.sum())
    extra = farthest_region(discs, face, extra_n)

    masks = {
        "none": np.zeros_like(discs),
        "centre": discs,
        "matched": farthest_region(discs, face, int(discs.sum())),
        "periphery": ~discs,
        "span": ~hull,
        "span_matched": ~(discs | extra),
    }
    areas = {
        "area_discs": float(discs.sum()) / total,
        "area_hull": float(hull.sum()) / total,
        "area_span_gain": float(hull.sum() - discs.sum()) / total,
        "kept_span": float(hull.sum()) / total,
        "kept_span_matched": float((discs | extra).sum()) / total,
    }
    areas["span_match_diff"] = abs(areas["kept_span"] - areas["kept_span_matched"])
    areas["span_matched_overlap"] = float((hull & (discs | extra)).sum()) / max(
        float(hull.sum()), 1.0
    ) - areas["area_discs"] / max(areas["kept_span"], 1e-9)
    return masks, areas


# ------------------------------------------------------------------- colour


def normalise_face_colour(img: np.ndarray, matte: np.ndarray | None) -> np.ndarray | None:
    """Rescale each channel inside the face matte to a fixed mean and sd.

    Makes the six-number descriptor (per-channel mean+std inside the matte) constant
    across bats by construction, which is the only way a spatial ablation can test
    spatial dependence in this dataset. Pixels outside the matte are left alone:
    green stays green, and `random` backgrounds are already randomised so they carry
    no identity.
    """
    if matte is None or not matte.any():
        return None
    out = img.astype(np.float64)
    sel = matte.astype(bool)
    for c in range(3):
        ch = out[:, :, c]
        vals = ch[sel]
        sd = float(vals.std())
        if sd < 1e-6:
            return None  # degenerate channel: rescaling would amplify noise without bound
        ch[sel] = (vals - float(vals.mean())) / sd * CN_STD + CN_MEAN
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_fill(img: np.ndarray, destroy: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[destroy] = FILL_COLOUR
    return out


# --------------------------------------------------------------------- build


def paired_path(path: pathlib.Path, background: str) -> pathlib.Path:
    """The `green` twin of an image in another background, by basename."""
    return pathlib.Path(str(path).replace(f"/{background}_bg/", "/green_bg/"))


def run(
    species: str,
    background: str,
    crop: str,
    edge: int,
    prefix: str,
    *,
    device: str,
    execute: bool,
    measure_only: bool,
    limit_identities: int,
) -> dict[str, Any]:
    tag = f"{prefix}{background}_bg"
    src_csv = pathlib.Path(f"data/manifests/{species}_{tag}_{crop}_{edge}_manifest.csv")
    if not src_csv.exists():
        raise SystemExit(f"no source manifest: {src_csv} (build the clean arm first)")
    manifest = manifest_from_csv(src_csv)

    pose = YOLOPoseEstimator({"weights": POSE_WEIGHTS, "device": device})
    base = pathlib.Path(f"data/processed/{species}/video/not_augmented/{tag}/{crop}/{edge}/occ")

    keep_ids: set[str] = set()
    if limit_identities:
        keep_ids = set(sorted({r.identity for r in manifest.records})[:limit_identities])

    rows: list[dict[str, Any]] = []
    written: dict[tuple[str, str], list[ImageRecord]] = {(c, a): [] for c in COLOURS for a in ARMS}
    n_pose_fail = 0

    for rec in manifest.records:
        if keep_ids and rec.identity not in keep_ids:
            continue
        p = pathlib.Path(rec.path)
        img = cv2.imread(str(p))
        if img is None:
            n_pose_fail += 1
            continue

        # ROI and matte both come from the green twin, so every background gets
        # bit-identical masks and pose never runs on a random natural background.
        green_p = p if background == "green" else paired_path(p, background)
        green = cv2.imread(str(green_p))
        if green is None:
            n_pose_fail += 1
            continue
        pr = pose.predict(green)
        rois = rois_from_keypoints(pr.keypoints, img.shape[:2]) if pr is not None else None
        if rois is None:
            n_pose_fail += 1
            continue
        discs = rois["eyes"] | rois["nose"]
        matte = face_mask_from_green(green)

        masks, areas = destroy_masks(discs, matte)
        rows.append({"basename": p.name, "identity": rec.identity, **areas})
        if measure_only:
            continue

        for colour in COLOURS:
            src = img if colour == "raw" else normalise_face_colour(img, matte)
            if src is None:
                continue
            for arm in ARMS:
                dst = base / colour / arm / rec.identity / p.name
                if execute:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(dst),
                        apply_fill(src, masks[arm]),
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                    )
                written[(colour, arm)].append(
                    ImageRecord(
                        path=dst,
                        identity=rec.identity,
                        species=rec.species,
                        background=rec.background,
                        source=rec.source,
                        augmented=rec.augmented,
                        split=rec.split,
                        quality=float(compute_quality(dst)) if execute else 0.0,
                    )
                )

    import pandas as pd

    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "species": species,
        "background": background,
        "crop": crop,
        "edge": edge,
        "prefix": prefix,
        "source_manifest": str(src_csv),
        "n_source": len(manifest.records),
        "n_measured": len(df),
        "n_pose_failed": n_pose_fail,
        "fill_colour": list(FILL_COLOUR),
        "cn_target": {"mean": CN_MEAN, "std": CN_STD},
        "declared_bounds": {
            "max_area_diff": MAX_AREA_DIFF,
            "max_overlap": MAX_OVERLAP,
            "min_span_gain": MIN_SPAN_GAIN,
        },
    }
    if not df.empty:
        summary["areas"] = {
            c: {"mean": float(df[c].mean()), "min": float(df[c].min()), "max": float(df[c].max())}
            for c in ("area_discs", "area_hull", "area_span_gain", "span_match_diff")
        }
        gain = float(df.area_span_gain.mean())
        summary["span_gate"] = {
            "mean_gain": gain,
            "threshold": MIN_SPAN_GAIN,
            "passes": bool(gain >= MIN_SPAN_GAIN),
        }

    if execute and not measure_only:
        mans = {}
        for (colour, arm), recs in written.items():
            if not recs:
                continue
            out_csv = pathlib.Path(
                f"data/manifests/{species}_{tag}_{crop}_{edge}_occ_{colour}_{arm}_manifest.csv"
            )
            man = Manifest.from_records(recs)
            man.assert_identity_disjoint()
            manifest_to_csv(man, out_csv)
            mans[f"{colour}/{arm}"] = str(out_csv)
        summary["manifests"] = mans
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", nargs="+", default=["mauritius"])
    ap.add_argument("--background", nargs="+", default=list(BACKGROUNDS), choices=BACKGROUNDS)
    ap.add_argument("--crop", default="head")
    ap.add_argument("--edge", type=int, default=192)
    ap.add_argument("--prefix", default="", help="Dataset prefix, e.g. 'day31_'.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--measure-only", action="store_true", help="Area gate only; write nothing.")
    ap.add_argument("--limit-identities", type=int, default=0)
    ap.add_argument("--execute", action="store_true", help="Actually write; default is a dry run.")
    args = ap.parse_args()

    prefix = args.prefix if not args.prefix or args.prefix.endswith("_") else f"{args.prefix}_"
    out = []
    for sp in args.species:
        for bg in args.background:
            s = run(
                sp,
                bg,
                args.crop,
                args.edge,
                prefix,
                device=args.device,
                execute=args.execute and not args.measure_only,
                measure_only=args.measure_only,
                limit_identities=args.limit_identities,
            )
            out.append(s)
            a = s.get("areas")
            print(
                f"\n  {sp}/{bg} {args.crop}@{args.edge}: {s['n_measured']} measured "
                f"({s['n_pose_failed']} pose/read failures)"
            )
            if a:
                print(
                    f"    discs {a['area_discs']['mean']:.3f}  hull {a['area_hull']['mean']:.3f}"
                    f"  gain {a['area_span_gain']['mean']:+.3f}"
                    f"  (gate {MIN_SPAN_GAIN:+.3f}: "
                    f"{'PASS' if s['span_gate']['passes'] else 'FAIL -> drop span'})"
                )
                print(
                    f"    span vs span_matched kept-area diff: "
                    f"max {a['span_match_diff']['max']:.4f} (bound {MAX_AREA_DIFF})"
                )
            if s.get("manifests"):
                print(f"    wrote {len(s['manifests'])} manifest(s)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"\n  -> {OUT_JSON}")
    if not args.execute and not args.measure_only:
        print("  DRY RUN -- nothing written. Re-run with --execute.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
