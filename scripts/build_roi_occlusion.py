"""Build the four ROI-occlusion arms for the periphery-dependence test.

Executes `docs/occlusion_preregistration.md` (including Amendment 1). The
question it serves: the saliency analysis found that bat-specific training moves
attribution away from the eye/nose discs and toward the periphery in 38 of 40
models. That is a claim about where a model looks. This build lets us ask whether
performance actually *depends* on those pixels.

The arms
--------
``none``       clean, but re-encoded through the same JPEG round-trip as the
               others so the baseline is not sharper by one encode.
``centre``     the eye+nose discs destroyed (~24% of the crop).
``matched``    the same NUMBER of pixels destroyed, on the animal wherever
               possible, chosen furthest from the eye/nose discs. The
               area-matched control, and the whole point of the design: the
               periphery is 76% of the crop, so comparing "remove the discs"
               against "remove the periphery" would mostly measure how many
               pixels were destroyed.
``periphery``  everything except the eye+nose discs destroyed (~76%).

Why ``matched`` is built this way rather than as displaced discs: the declared
construction (three discs rotated 180 degrees about the crop centre) is
infeasible here. The aligner centres the crop on the face, so the keypoint
centroid sits at the crop centre and a 180-degree rotation leaves 36% overlap.
Measured before building; see Amendment 1.

Destroyed pixels are filled with a **single global constant**, neutral grey.

Not the crop's own mean colour, which is what the first version used. A per-image
mean carries no *spatial* information but it does carry **colour** information,
and three numbers of mean colour identify a bat at ROC-AUC 0.613 (mauritius) /
0.734 (rousettus) with no model at all. That handed the `periphery` arm 76% of
every image painted with an identity cue, and the positive control duly failed:
the more the ablation destroyed, the more of the image became a summary of what
had been destroyed. See Amendment 2 in the pre-registration. A global constant
carries no per-image information of any kind.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import cv2
import numpy as np

from bat_core.types import ImageRecord, Manifest
from bat_data import manifest_from_csv, manifest_to_csv
from bat_data.manifest import compute_quality
from bat_data.quality_metrics import face_mask_from_green

ARMS = ("none", "centre", "matched", "periphery")
JPEG_QUALITY = 95

# Neutral grey, identical for every image and both species. The one property that
# matters is that it is not derived from the image.
FILL_COLOUR = (128, 128, 128)
POSE_WEIGHTS = "models/preprocessing/face_pose.pt"

# green only. `original` is 93% solvable with the face deleted (arm 3C), so an
# occlusion test there would partly measure the background.
SOURCES = {
    "mauritius": "data/manifests/mauritius_green_bg_manifest.csv",
    "rousettus": "data/manifests/rousettus_green_bg_intersect_manifest.csv",
}

# Declared in the pre-registration.
MAX_AREA_DIFF = 0.02
MAX_OVERLAP = 0.10
OUT_JSON = pathlib.Path("outputs/occlusion/roi_occlusion_build.json")


def matched_region(centre: np.ndarray, face: np.ndarray | None) -> np.ndarray:
    """Same pixel count as `centre`, on the animal if possible, furthest from it.

    Ranking, not sampling, so it is deterministic and the area is exact rather
    than approximate. On-animal pixels outrank background by a large constant, and
    within each group the tie-break is distance from the eye/nose ROI.
    """
    h, w = centre.shape
    dist = cv2.distanceTransform((~centre).astype(np.uint8), cv2.DIST_L2, 5)
    score = dist.astype(np.float64)
    if face is not None:
        score = score + face.astype(np.float64) * 1e4
    score[centre] = -1.0  # never re-select the region we are contrasting against
    n = int(centre.sum())
    idx = np.argpartition(score.ravel(), -n)[-n:]
    out = np.zeros(h * w, dtype=bool)
    out[idx] = True
    return out.reshape(h, w)


def build_species(species: str, source: str, *, device: str, force: bool) -> dict[str, Any]:
    from bat_preprocessing import YOLOPoseEstimator

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from species_saliency_maps import rois_from_keypoints

    manifest = manifest_from_csv(pathlib.Path(source))
    pose = YOLOPoseEstimator({"weights": POSE_WEIGHTS, "device": device})
    outdirs = {
        arm: pathlib.Path(f"data/processed/{species}/video/not_augmented/occ_roi/{arm}")
        for arm in ARMS
    }
    for d in outdirs.values():
        d.mkdir(parents=True, exist_ok=True)

    kept: list[ImageRecord] = []
    stats: list[dict[str, float]] = []
    n_pose_fail = n_bounds_fail = 0

    for rec in manifest.records:
        src = pathlib.Path(rec.path)
        img = cv2.imread(str(src))
        if img is None:
            n_pose_fail += 1
            continue
        pr = pose.predict(img)
        rois = rois_from_keypoints(pr.keypoints, img.shape[:2]) if pr is not None else None
        if rois is None:
            # Dropped from EVERY arm, so all four share one image set. A baseline
            # on a different set of images would confound the ablation.
            n_pose_fail += 1
            continue

        centre = rois["eyes"] | rois["nose"]
        face = face_mask_from_green(img)
        matched = matched_region(centre, face)

        a_c, a_m = float(centre.mean()), float(matched.mean())
        overlap = float((matched & centre).sum()) / max(int(matched.sum()), 1)
        if abs(a_m - a_c) > MAX_AREA_DIFF or overlap > MAX_OVERLAP:
            n_bounds_fail += 1
            continue

        # Global constant, NOT the crop's mean. See the module docstring: a
        # per-image mean colour is itself an identity cue worth ROC-AUC 0.61-0.73.
        fill = np.asarray(FILL_COLOUR, dtype=np.float64)
        masks = {
            "none": np.zeros_like(centre),
            "centre": centre,
            "matched": matched,
            "periphery": ~centre,
        }
        for arm, mask in masks.items():
            out = img.copy()
            if mask.any():
                out[mask] = fill.astype(img.dtype)
            dst = outdirs[arm] / rec.identity / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        kept.append(rec)
        stats.append(
            {
                "area_centre": a_c,
                "area_matched": a_m,
                "area_periphery": float((~centre).mean()),
                "overlap": overlap,
                "matched_on_animal": (
                    float((matched & face).sum()) / max(int(matched.sum()), 1)
                    if face is not None
                    else float("nan")
                ),
                "centre_on_animal": (
                    float((centre & face).sum()) / max(int(centre.sum()), 1)
                    if face is not None
                    else float("nan")
                ),
            }
        )

    # One manifest per arm, over the identical kept set.
    manifests: dict[str, str] = {}
    for arm in ARMS:
        out_csv = pathlib.Path(f"data/manifests/{species}_occ_roi_{arm}_manifest.csv")
        if out_csv.exists() and not force:
            print(f"    {out_csv} exists; pass --force to overwrite")
            manifests[arm] = str(out_csv)
            continue
        recs = [
            ImageRecord(
                path=outdirs[arm] / r.identity / pathlib.Path(r.path).name,
                identity=r.identity, species=r.species, background=r.background,
                source=r.source, augmented=r.augmented, split=r.split,
                quality=float(compute_quality(outdirs[arm] / r.identity / pathlib.Path(r.path).name)),
            )
            for r in kept
        ]
        man = Manifest.from_records(recs)
        man.assert_identity_disjoint()
        manifest_to_csv(man, out_csv)
        manifests[arm] = str(out_csv)

    df = {k: np.array([s[k] for s in stats]) for k in stats[0]} if stats else {}
    summary = {
        "species": species,
        "source_manifest": source,
        "n_source": len(manifest.records),
        "n_kept": len(kept),
        "n_pose_failed": n_pose_fail,
        "n_bounds_failed": n_bounds_fail,
        "n_identities": len({r.identity for r in kept}),
        "manifests": manifests,
        "area_checks": {
            k: {"mean": float(v.mean()), "max": float(v.max()), "min": float(v.min())}
            for k, v in df.items()
        },
        "max_abs_area_diff": float(np.abs(df["area_matched"] - df["area_centre"]).max()) if df else None,
        "declared_bounds": {"max_area_diff": MAX_AREA_DIFF, "max_overlap": MAX_OVERLAP},
        "fill_colour": list(FILL_COLOUR),
    }
    print(
        f"  [{species}] kept {len(kept)}/{len(manifest.records)} "
        f"(pose fail {n_pose_fail}, bounds fail {n_bounds_fail}), "
        f"{summary['n_identities']} identities\n"
        f"    area  centre={df['area_centre'].mean():.4f}  matched={df['area_matched'].mean():.4f}  "
        f"periphery={df['area_periphery'].mean():.4f}\n"
        f"    max |area diff| = {summary['max_abs_area_diff']:.6f} (bound {MAX_AREA_DIFF})   "
        f"max overlap = {df['overlap'].max():.6f} (bound {MAX_OVERLAP})\n"
        f"    on-animal: matched={df['matched_on_animal'].mean():.3f}  "
        f"centre={df['centre_on_animal'].mean():.3f}"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", nargs="+", default=list(SOURCES))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=OUT_JSON)
    args = ap.parse_args()

    print("=== ROI occlusion build (green background only) ===")
    out = [build_species(sp, SOURCES[sp], device=args.device, force=args.force) for sp in args.species]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
