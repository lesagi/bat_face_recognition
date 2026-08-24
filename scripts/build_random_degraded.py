"""Extend the Phase 2A degraded-mauritius arms to the RANDOM background.

Why the random background matters more than green or original
------------------------------------------------------------
The published sweep found a species difference on exactly one background:

    green     ArcFace  mau 0.815  rou 0.858   gap -0.043  (rousettus ahead)
    original  ArcFace  mau 0.933  rou 0.854   gap +0.079
    random    ArcFace  mau 0.698  rou 0.582   gap +0.116  <- the only
    random    AdaFace  mau 0.696  rou 0.560   gap +0.136  <- significant cells

Arms 3A and 3B were specified as green/original, so as originally planned they
control the image-quality confound everywhere *except* where the effect they
exist to explain actually appears. On green there is no mauritius advantage to
explain at all. This script closes that gap for 3A; the 3B side is a row filter
handled in scripts/build_band_restricted.py.

`original` is additionally compromised: arm 3C shows a model trained on
face-removed original images reaches 0.831 (mauritius) / 0.780 (rousettus)
against a random-background control at chance, so most of that arm's signal is
background matching. `random` is the clean high-signal arm.

Backgrounds are REUSED, not redrawn
-----------------------------------
`random_pool` in build_variants_from_frames.py fetches images over the network,
so redrawing would make the degraded arm differ from the published one in both
face pixels *and* background pixels. Instead the published `random_bg` crop is
used directly as the background plate, so the only thing that changes is the
face. (Geometry is shared: the recrop variant reuses full-resolution detections,
so its mask lines up with the published one.)

Per variant
-----------
``blur``    Blur the published random_bg crop with that bat's already-solved
            sigma. The published background is preserved exactly, then softened
            along with the face -- consistent with how the green/original blur
            arms were built (whole-crop blur), and what a lower-quality capture
            would do anyway.
``recrop``  Composite the degraded (re-cropped) face onto the published random
            crop, using the mask recovered from the recrop green crop and the
            same smooth/feather the original build used.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from multiprocessing import Pool
from typing import Any

import cv2
import numpy as np
import pandas as pd
from bat_preprocessing import replace_background

from bat_core.types import ImageRecord, Manifest
from bat_data import manifest_from_csv, manifest_to_csv
from bat_data.manifest import compute_quality
from bat_data.quality_metrics import face_mask_from_green

SRC = "mauritius"
PUB_RANDOM = f"data/manifests/{SRC}_manifest.csv"          # the published random arm
PUB_GREEN = f"data/manifests/{SRC}_green_bg_manifest.csv"
SMOOTH, FEATHER, JPEG_QUALITY = 5, 4, 95
REVIEW_DIR = pathlib.Path("outputs/phase2a_review")
N_WORKERS = 20


def _blur_one(a: tuple[str, str, float]) -> bool:
    src, dst, sigma = a
    img = cv2.imread(src)
    if img is None:
        return False
    out = img if sigma <= 1e-6 else cv2.GaussianBlur(img, (0, 0), sigma)
    return bool(cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]))


def _recrop_one(a: tuple[str, str, str, str]) -> bool:
    """Degraded face + published random plate."""
    pub_random, recrop_original, recrop_green, dst = a
    plate = cv2.imread(pub_random)
    face = cv2.imread(recrop_original)
    green = cv2.imread(recrop_green)
    if plate is None or face is None or green is None:
        return False
    raw = face_mask_from_green(green)
    if raw is None:
        return False
    out = replace_background(face, raw.astype(np.uint8), plate, smooth=SMOOTH, feather=FEATHER)
    return bool(cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]))


def _emit(outdir: pathlib.Path, out_csv: str) -> dict[str, Any]:
    src = manifest_from_csv(pathlib.Path(PUB_RANDOM))
    recs = []
    for r in src.records:
        dst = outdir / pathlib.Path(r.path).name
        if dst.exists():
            recs.append(ImageRecord(
                path=dst, identity=r.identity, species=r.species, background=r.background,
                source=r.source, augmented=r.augmented, split=r.split,
                quality=float(compute_quality(dst))))
    man = Manifest.from_records(recs)
    man.assert_identity_disjoint()
    manifest_to_csv(man, pathlib.Path(out_csv))
    print(f"       manifest {out_csv}  n={len(recs)} ids={len(man.identities())}")
    return {"out_manifest": out_csv, "out_manifest_hash": man.manifest_hash, "n": len(recs)}


def build(variant: str) -> dict[str, Any]:
    pub_rand = {pathlib.Path(p).name: p for p, _ in
                [(str(r.path), r.identity) for r in manifest_from_csv(pathlib.Path(PUB_RANDOM)).records]}
    ident_of = {pathlib.Path(str(r.path)).name: r.identity
                for r in manifest_from_csv(pathlib.Path(PUB_RANDOM)).records}
    outdir = pathlib.Path(f"data/processed/{SRC}/video/not_augmented/{variant}/random_bg")
    outdir.mkdir(parents=True, exist_ok=True)

    if variant == "blur":
        sol = pd.read_parquet(REVIEW_DIR / "blur_solve.parquet")
        sigma_by_ident = dict(zip(sol["identity"], sol["sigma"]))
        jobs = [(p, str(outdir / n), float(sigma_by_ident[ident_of[n]]))
                for n, p in pub_rand.items() if ident_of[n] in sigma_by_ident]
        with Pool(N_WORKERS) as pool:
            ok = int(sum(pool.map(_blur_one, jobs, chunksize=16)))
        extra = {"sigma_source": "outputs/phase2a_review/blur_solve.parquet (per-bat, unchanged)"}
    else:
        ro = pathlib.Path(f"data/processed/{SRC}/video/not_augmented/recrop/original_bg")
        rg = pathlib.Path(f"data/processed/{SRC}/video/not_augmented/recrop/green_bg")
        jobs = [(p, str(ro / n), str(rg / n), str(outdir / n))
                for n, p in pub_rand.items() if (ro / n).exists() and (rg / n).exists()]
        with Pool(N_WORKERS) as pool:
            ok = int(sum(pool.map(_recrop_one, jobs, chunksize=16)))
        extra = {"background_plate": "published random_bg crop (reused, not redrawn)",
                 "smooth": SMOOTH, "feather": FEATHER}

    print(f"[{variant}/random] wrote {ok}/{len(jobs)} -> {outdir}")
    summary = {"variant": variant, "background": "random", "n_written": ok,
               "n_jobs": len(jobs), "out_dir": str(outdir), **extra}
    summary.update(_emit(outdir, f"data/manifests/{SRC}_{variant}_random_manifest.csv"))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("blur", "recrop", "both"), default="both")
    args = ap.parse_args()
    variants = ("blur", "recrop") if args.variant == "both" else (args.variant,)
    out = [build(v) for v in variants]
    p = REVIEW_DIR / "random_degraded_build.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
