"""Phase 2A variant `recrop`: re-crop mauritius from frames filmed "further away".

Reproduces the *cause* of the species image-quality gap rather than its measured
effect. Both species' frames are 1080x1920 and a mauritius face fills ~57% of the
frame width against rousettus' ~29%, so the gap is a shooting-distance
difference. Downsampling a mauritius frame by that ratio and re-cropping IS the
same bat filmed further away: the face then passes through the sensor scale, the
resampling and the frame-level JPEG quantisation that a rousettus face passed
through.

What is held fixed, and why it matters
--------------------------------------
Segmentation and pose run on the FULL-RESOLUTION frame, exactly as the original
build did, so the keypoints -- and therefore the crop centre, rotation and
extent -- are identical to the existing crops. Only the pixels the warp samples
from change. Re-detecting on the downsampled frame would let alignment drift
become a second difference between the arms and confound the comparison.

Granularity is per-BAT, matching the `blur` variant: each bat was filmed in
exactly one video, so shooting distance is a property of the bat. A per-frame
factor would model a camera that moved between frames of one clip.

Expected limitation, stated in advance
--------------------------------------
Downsampling cannot reproduce focus error or motion blur, and it *removes*
sensor noise while mauritius is the noisier species. So this variant is expected
to UNDER-degrade relative to `blur`. That is the point of running both: `blur`
closes the measured gap by construction but is not physically motivated, while
this one is physically motivated and reports whatever it achieves. The residual
is a finding, not a defect to patch.
"""

from __future__ import annotations

import pathlib
from dataclasses import replace as dc_replace
from typing import Any

import cv2
import numpy as np
import pandas as pd
from bat_preprocessing import (
    BackgroundGenerator,
    FaceAligner,
    YOLOPoseEstimator,
    YOLOSegmenter,
    replace_background,
)

from bat_core.types import ImageRecord, Manifest
from bat_data import manifest_from_csv, manifest_to_csv
from bat_data.manifest import compute_quality

SRC, REF = "mauritius", "rousettus"
EDGE, MARGIN, SMOOTH, FEATHER = 224, 0.03, 5, 4
POSE_CROP_MARGIN, MIN_KPT_CONF, NOSE_ROLL = 0.12, 0.35, 0.0
JPEG_QUALITY = 95
SEG_WEIGHTS = "models/preprocessing/face_seg_mauritius_v2.pt"
POSE_WEIGHTS = "models/preprocessing/face_pose.pt"
NATIVE_BOXES = pathlib.Path("outputs/quality/native_boxes.parquet")
REVIEW_DIR = pathlib.Path("outputs/phase2a_review")


def _quantile_map(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ranks = source.argsort().argsort()
    return np.quantile(reference, ranks / max(len(source) - 1, 1))


def build_recrop(force: bool, device: str = "cpu") -> dict[str, Any]:
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from build_variants_from_frames import pose_on_mask_crop

    native = pd.read_parquet(NATIVE_BOXES)
    mau = native[native.species == SRC]
    rou = native[native.species == REF]

    # Per-bat factor from identity-median native head-box, mapped onto the
    # rousettus identity-median distribution.
    src_med = mau.groupby("identity")["native_side_px"].median().sort_values()
    ref_med = rou.groupby("identity")["native_side_px"].median()
    targets = _quantile_map(src_med.to_numpy(), ref_med.to_numpy())
    factor = {
        ident: max(float(nat) / float(tg), 1.0)  # only ever shrink
        for ident, nat, tg in zip(src_med.index, src_med.to_numpy(), targets)
    }
    print("[recrop] per-bat downscale factors: "
          f"min={min(factor.values()):.3f} median={np.median(list(factor.values())):.3f} "
          f"max={max(factor.values()):.3f}")

    # Only rebuild the basenames the training manifest actually uses.
    base_man = manifest_from_csv(pathlib.Path(f"data/manifests/{SRC}_green_bg_manifest.csv"))
    wanted = {pathlib.Path(r.path).name for r in base_man.records}
    frame_of = dict(zip(mau.basename, mau.source_frame))
    ident_of = dict(zip(mau.basename, mau.identity))

    seg = YOLOSegmenter({"weights": SEG_WEIGHTS, "device": device})
    pose = YOLOPoseEstimator({"weights": POSE_WEIGHTS, "device": device})
    aligner = FaceAligner(
        edge_length=EDGE, margin_ratio=MARGIN, mode="eye_anchored",
        min_keypoint_confidence=MIN_KPT_CONF, require_confident_eyes=False,
        nose_roll_weight=NOSE_ROLL,
    )
    outdirs = {
        bg: pathlib.Path(f"data/processed/{SRC}/video/not_augmented/recrop/{bg}_bg")
        for bg in ("green", "original")
    }
    for d in outdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    frames_dir = REVIEW_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    n_ok = n_fail = 0
    saved_examples = 0
    for basename in sorted(wanted):
        fp = frame_of.get(basename)
        ident = ident_of.get(basename)
        if fp is None or ident is None:
            n_fail += 1
            continue
        frame = cv2.imread(str(fp))
        if frame is None:
            n_fail += 1
            continue
        s = seg.predict(frame)
        if s is None:
            n_fail += 1
            continue
        pr = pose_on_mask_crop(pose, frame, s.mask, POSE_CROP_MARGIN) or pose.predict(frame)

        f = factor[ident]
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(int(round(w / f)), 8), max(int(round(h / f)), 8)),
                           interpolation=cv2.INTER_AREA)
        # Re-encode so compression acts at the reduced face scale, which is a real
        # part of what a smaller face suffers: DCT blocks are 8x8 in FRAME pixels,
        # so a face spanning fewer frame pixels gets fewer coded coefficients.
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            small = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if saved_examples < 8:
            cv2.imwrite(str(frames_dir / f"{basename.rsplit('.jpg')[0]}__orig.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            cv2.imwrite(str(frames_dir / f"{basename.rsplit('.jpg')[0]}__f{f:.2f}.jpg"), small,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved_examples += 1

        # Geometry from the full-res detection, rescaled: mask is resized by the
        # aligner itself, keypoints must be scaled here.
        sy, sx = small.shape[0] / h, small.shape[1] / w
        pr_s = None
        if pr is not None:
            k = pr.keypoints.copy().astype(float)
            k[:, 0] *= sx
            k[:, 1] *= sy
            pr_s = dc_replace(pr, keypoints=k, original_image_shape=small.shape[:2])
        af = aligner.align_eye_anchored(small, s, pr_s)
        if af is None:
            n_fail += 1
            continue

        cv2.imwrite(str(outdirs["original"] / basename), af.image,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        green = replace_background(af.image, af.mask, BackgroundGenerator.solid_color,
                                   smooth=SMOOTH, feather=FEATHER, color=(0, 255, 0))
        cv2.imwrite(str(outdirs["green"] / basename), green, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        rows.append({"basename": basename, "identity": ident, "recrop_factor": f,
                     "native_side_px_before": float(mau.loc[mau.basename == basename,
                                                            "native_side_px"].iloc[0]),
                     "frame_shape_after": f"{small.shape[1]}x{small.shape[0]}"})
        n_ok += 1
        if n_ok % 100 == 0:
            print(f"    {n_ok}/{len(wanted)} ...", flush=True)

    fac = pd.DataFrame(rows)
    fac["native_side_px_after"] = fac.native_side_px_before / fac.recrop_factor
    fac.to_parquet(REVIEW_DIR / "recrop_factors.parquet")

    for bg in ("green", "original"):
        src_csv = f"data/manifests/{SRC}_{bg}_bg_manifest.csv"
        out_csv = f"data/manifests/{SRC}_recrop_{bg}_manifest.csv"
        src = manifest_from_csv(pathlib.Path(src_csv))
        out = pathlib.Path(out_csv)
        if out.exists() and not force:
            print(f"       {out} exists; pass --force to overwrite")
            continue
        recs = []
        for r in src.records:
            dst = outdirs[bg] / pathlib.Path(r.path).name
            if not dst.exists():
                continue
            recs.append(ImageRecord(
                path=dst, identity=r.identity, species=r.species, background=r.background,
                source=r.source, augmented=r.augmented, split=r.split,
                quality=float(compute_quality(dst))))
        man = Manifest.from_records(recs)
        man.assert_identity_disjoint()
        manifest_to_csv(man, out)
        print(f"       manifest {out}  n={len(recs)} ids={len(man.identities())}")

    summary = {
        "variant": "recrop", "granularity": "per-identity (one factor per bat)",
        "n_written": n_ok, "n_failed": n_fail,
        "factor_min": float(min(factor.values())),
        "factor_median": float(np.median(list(factor.values()))),
        "factor_max": float(max(factor.values())),
        "per_identity_factor": {k: float(v) for k, v in factor.items()},
        "native_side_px_median_before": float(fac.native_side_px_before.median()),
        "native_side_px_median_after": float(fac.native_side_px_after.median()),
        "reference_identity_median_native": float(ref_med.median()),
        "seg_weights": SEG_WEIGHTS, "pose_weights": POSE_WEIGHTS,
        "jpeg_quality": JPEG_QUALITY, "device": device,
    }
    print(f"[recrop] wrote {n_ok} (failed {n_fail}); native head-box median "
          f"{summary['native_side_px_median_before']:.1f} -> "
          f"{summary['native_side_px_median_after']:.1f} px "
          f"(rousettus identity-median {summary['reference_identity_median_native']:.1f})")
    return summary
