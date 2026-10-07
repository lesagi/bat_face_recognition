"""Degrade each mauritius bat to the resolution of the rousettus bat it is paired with.

`build_resolution_recrop.py` (frozen, Phase 2A) matches the two species'
resolution *distributions*: it sorts each species by native head-box size and maps
quantile to quantile. That is the right control when the comparison is
species-against-species.

This is for the paired design instead. `pair_species_bats.py` matches each
rousettus bat to the mauritius bat closest to it in **lighting**; this gives that
same mauritius bat the **resolution** of its partner. After both, a pair differs in
neither lighting nor resolution -- only in species. The quantile map cannot do that,
because a bat's rank in the resolution distribution has nothing to do with which bat
it was paired to on lighting.

Method, identical in spirit to the frozen script, because the point is to reproduce
the *cause* rather than fake its effect: downsample the full-resolution source frame
by the pair's factor, re-encode it as JPEG so compression acts at the reduced face
scale (DCT blocks are 8x8 in FRAME pixels, so a smaller face gets fewer coded
coefficients), then re-crop. Segmentation and pose run on the FULL-RESOLUTION frame
and the keypoints are scaled down, so crop centre, rotation and extent are identical
to the undegraded build -- only the pixels the warp samples change. Re-detecting on
the shrunken frame would let alignment drift become a second difference and confound
the comparison.

Factors are per bat, since each bat is one video and shooting distance is a property
of the clip. They only ever shrink: a mauritius bat already smaller than its partner
is left alone rather than upscaled, which would invent detail.

Stated in advance, as the frozen script does: downsampling cannot reproduce focus
error or motion blur, and it *removes* sensor noise while mauritius is the noisier
species. So this under-degrades. The residual is a finding, not a defect to patch.

    uv run python scripts/build_pair_matched_recrop.py --execute
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import replace as dc_replace
from typing import Any

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from build_variants_from_frames import pose_on_mask_crop  # noqa: E402

from bat_core.types import ImageRecord, Manifest  # noqa: E402
from bat_data import manifest_from_csv, manifest_to_csv  # noqa: E402
from bat_data.manifest import compute_quality  # noqa: E402
from bat_preprocessing import (  # noqa: E402
    BackgroundGenerator,
    FaceAligner,
    YOLOPoseEstimator,
    YOLOSegmenter,
    replace_background,
)

PAIRS = pathlib.Path("outputs/quality/species_pairs.json")
NATIVE_MAU = pathlib.Path("outputs/quality/native_boxes_cull2.parquet")
NATIVE_PUB = pathlib.Path("outputs/quality/native_boxes.parquet")
KEEP_ROOT = pathlib.Path("data/curated/mauritius/keep")
OUT_JSON = pathlib.Path("outputs/quality/pair_recrop_build.json")

SEG_WEIGHTS = "models/preprocessing/face_seg_mauritius_v2.pt"
POSE_WEIGHTS = "models/preprocessing/face_pose.pt"
# Must match the undegraded build, or the arms differ in framing as well as detail.
EDGE, MARGIN, SMOOTH, FEATHER = 224, 0.03, 5, 4
MIN_KPT_CONF, NOSE_ROLL, POSE_CROP_MARGIN = 0.25, 0.0, 0.12
JPEG_QUALITY = 95


def pair_factors() -> dict[str, dict[str, Any]]:
    """Per mauritius bat: the shrink factor onto its paired rousettus bat.

    Convention matches the frozen script: a factor >= 1 that the frame is divided
    by, clamped at 1.0 so this only ever shrinks.
    """
    pairs = json.loads(PAIRS.read_text())["pairs"]
    mau = pd.read_parquet(NATIVE_MAU).groupby("identity").native_side_px.median()
    pub = pd.read_parquet(NATIVE_PUB)
    rous = pub[pub.species == "rousettus"].groupby("identity").native_side_px.median()
    out = {}
    for p in pairs:
        m, r = p["mauritius"], p["rousettus"]
        if m not in mau.index or r not in rous.index:
            continue
        out[m] = {
            "rousettus": r,
            "native_mauritius": float(mau[m]),
            "native_rousettus": float(rous[r]),
            "factor": max(float(mau[m]) / float(rous[r]), 1.0),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prefix", default="cull2")
    ap.add_argument("--arm", default="pairres", help="Name for the degraded arm.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    fac = pair_factors()
    if not fac:
        raise SystemExit("no pair factors; run pair_species_bats.py and measure_native_boxes.py")
    v = np.array([f["factor"] for f in fac.values()])
    print(
        f"\n  {len(fac)} paired bats; shrink factor min={v.min():.2f} "
        f"median={np.median(v):.2f} max={v.max():.2f}"
    )
    for m, f in sorted(fac.items(), key=lambda kv: -kv[1]["factor"]):
        print(
            f"    {m:<18} -> {f['rousettus']:<12} {f['native_mauritius']:>5.0f}px "
            f"-> {f['native_rousettus']:>5.0f}px   /{f['factor']:.2f}"
        )
    if not args.execute:
        print("\n  DRY RUN -- nothing built. Re-run with --execute.\n")
        return 0

    seg = YOLOSegmenter({"weights": SEG_WEIGHTS, "device": args.device})
    pose = YOLOPoseEstimator({"weights": POSE_WEIGHTS, "device": args.device})
    aligner = FaceAligner(
        edge_length=EDGE,
        margin_ratio=MARGIN,
        mode="eye_anchored",
        min_keypoint_confidence=MIN_KPT_CONF,
        require_confident_eyes=False,
        nose_roll_weight=NOSE_ROLL,
    )
    base = pathlib.Path("data/processed/mauritius/video/not_augmented")
    outdirs = {
        bg: base / f"{args.prefix}_{args.arm}_{bg}_bg" / "head" / str(EDGE)
        for bg in ("green", "original")
    }

    # Mirror the undegraded arm's split exactly, so the only difference between the
    # two arms is pixel detail -- not which bats landed in test.
    src_man = manifest_from_csv(
        pathlib.Path(
            f"data/manifests/mauritius_{args.prefix}_paired_green_bg_head_{EDGE}_manifest.csv"
        )
    )
    split_of = {r.identity: r.split for r in src_man.records}

    recs: dict[str, list[ImageRecord]] = {"green": [], "original": []}
    rows, n_ok, n_fail = [], 0, 0
    for ident, info in sorted(fac.items()):
        f = info["factor"]
        for fp in sorted((KEEP_ROOT / ident).glob("*.jpg")):
            frame = cv2.imread(str(fp))
            if frame is None:
                n_fail += 1
                continue
            s = seg.predict(frame)
            if s is None:
                n_fail += 1
                continue
            pr = pose_on_mask_crop(pose, frame, s.mask, POSE_CROP_MARGIN) or pose.predict(frame)

            h, w = frame.shape[:2]
            small = cv2.resize(
                frame,
                (max(int(round(w / f)), 8), max(int(round(h / f)), 8)),
                interpolation=cv2.INTER_AREA,
            )
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                small = cv2.imdecode(buf, cv2.IMREAD_COLOR)

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

            tok = fp.name.split("__f")
            frame_id = int(tok[1].split("__")[0]) if len(tok) > 1 else n_ok
            name = f"m--{ident}--f{frame_id:06d}.jpg"
            imgs = {
                "original": af.image,
                "green": replace_background(
                    af.image,
                    af.mask,
                    BackgroundGenerator.solid_color,
                    smooth=SMOOTH,
                    feather=FEATHER,
                    color=(0, 255, 0),
                ),
            }
            for bg, img in imgs.items():
                dst = outdirs[bg] / ident / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                recs[bg].append(
                    ImageRecord(
                        path=dst,
                        identity=ident,
                        species="mauritius",
                        background="green" if bg == "green" else "original",
                        source="video",
                        augmented=False,
                        split=split_of.get(ident, "train"),
                        quality=float(compute_quality(dst)),
                    )
                )
            rows.append(
                {
                    "identity": ident,
                    "basename": name,
                    "factor": f,
                    "frame_after": f"{small.shape[1]}x{small.shape[0]}",
                }
            )
            n_ok += 1
            if n_ok % 100 == 0:
                print(f"    {n_ok} frames ...", flush=True)

    mans = {}
    for bg, rr in recs.items():
        if not rr:
            continue
        man = Manifest.from_records(rr)
        man.assert_identity_disjoint()
        out = pathlib.Path(
            f"data/manifests/mauritius_{args.prefix}_{args.arm}_{bg}_bg_head_{EDGE}_manifest.csv"
        )
        manifest_to_csv(man, out)
        mans[bg] = str(out)
        ids = {s: len({r.identity for r in rr if r.split == s}) for s in ("train", "val", "test")}
        print(
            f"  {bg:9s} {len(rr):5d} imgs  train/val/test ids = "
            f"{ids['train']}/{ids['val']}/{ids['test']}  -> {out.name}"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "factors": fac,
                "n_ok": n_ok,
                "n_failed": n_fail,
                "manifests": mans,
                "frames": rows[:50],
            },
            indent=1,
        )
    )
    print(f"\n  {n_ok} frames built, {n_fail} failed\n  -> {OUT_JSON}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
