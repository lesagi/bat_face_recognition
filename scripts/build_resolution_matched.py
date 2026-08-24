"""Degrade mauritius to rousettus image quality, two independent ways.

Why this arm exists
-------------------
``docs/quality_parity.md`` established that mauritius faces are captured roughly
twice as large as rousettus faces with *zero* overlap (median native head-box 614
vs 312 px, Cliff's delta = +1.00), and that matching resolution collapses the
intrinsic-separability gap. So "rousettus individuals are less distinctive" is
confounded with "rousettus footage is worse". This build makes the mauritius
images as bad as the rousettus images so the species comparison can be re-run
with the confound removed (Phase 3 arm 3A).

Why there are two variants
--------------------------
The originally specified recipe -- quantile-map ``native_side_px`` onto the
rousettus distribution, downsample, upsample back to the model edge -- **is a
no-op at the resolution this project trains at.** ArcFace/AdaFace use
``input_edge_length: 112`` and ``outputs/quality/native_boxes.parquet`` shows
0.0% of images in *either* species are upscaled at 112 (rousettus' smallest
native head-box is 178 px). Both species are resized *down* to 112, after which
both hold exactly 112 px of face: there is no pixel-count difference left to
remove. The gap that does exist is optical -- a face filmed small in frame is
softer and noisier per face-area because focus, motion blur and frame-level JPEG
quantisation all act before cropping.

So the two variants attack the gap from opposite ends and fail in opposite
directions:

``recrop``
    Reproduces the *cause*. Downsamples the source frame by the per-image factor
    and re-crops, i.e. literally "the same bat filmed further away", then lets
    the measured effect fall where it may. Physically faithful, but it cannot
    reproduce focus or motion blur, so it will probably **under-degrade**.

``blur``
    Targets the measured *effect*. Solves per image for the Gaussian sigma that
    puts ``gradient_energy`` on the matching rousettus quantile. Closes the
    measured gap by construction, but the blur is not physically motivated and
    may remove the wrong frequencies.

If both shrink the species gap, the conclusion is robust to the
operationalisation. If they disagree, the disagreement localises which part of
image quality matters. Neither outcome is a failed build.

Which crops are calibrated against
----------------------------------
The *training* crops (``not_augmented/{green_bg,original_bg}``), not the
``aligned_224`` set that ``outputs/quality/battery.parquet`` measured. These are
byte-identical for mauritius but **different images for rousettus**, where the
training crops are sharper. Measured per-identity medians on the training crops:
laplacian_var 3.21x (docs: 3.78x), gradient_energy 1.87x (2.08x), tenengrad
1.07x (1.19x), noise 2.50x (3.13x). Calibrating against the published numbers
would over-degrade.

Only ``gradient_energy`` is targeted. Whether ``laplacian_var``, ``tenengrad``
and ``noise_sigma_immerkaer`` fall into line on their own is then real evidence;
aiming at all four would prove nothing.

Usage
-----
    uv run python scripts/build_resolution_matched.py --variant blur --verify
    uv run python scripts/build_resolution_matched.py --variant recrop --verify
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

from bat_core.types import ImageRecord, Manifest
from bat_data import manifest_from_csv, manifest_to_csv
from bat_data.manifest import compute_quality
from bat_data.quality_metrics import (
    _gray,
    _prepare_mask,
    compute_battery,
    face_mask_from_green,
    gradient_energy,
)

SRC = "mauritius"
REF = "rousettus"
BACKGROUNDS = ("green", "original")
TRAIN_MANIFEST = "data/manifests/{sp}_{bg}_bg_manifest.csv"
# rousettus green reference: the 1059-image intersection (Phase 2C), so the
# reference distribution matches the image set the green arms actually use.
REF_MANIFEST = "data/manifests/rousettus_green_bg_intersect_manifest.csv"
NATIVE_BOXES = pathlib.Path("outputs/quality/native_boxes.parquet")
REVIEW_DIR = pathlib.Path("outputs/phase2a_review")

SIGMA_MAX = 8.0          # a 224px crop blurred beyond this is unrecognisable
BISECT_ITERS = 14        # 8/2^14 < 0.001 px, far below any perceptual step
JPEG_QUALITY = 95        # matches build_variants_from_frames.py
N_WORKERS = 20


# ---------------------------------------------------------------------------
# Shared: per-image targets by quantile mapping
# ---------------------------------------------------------------------------
def _measure_one(args: tuple[str, str, str]) -> dict[str, Any] | None:
    path, identity, species = args
    m = compute_battery(path)
    if m is None:
        return None
    m.update(path=path, identity=identity, species=species, basename=pathlib.Path(path).name)
    return m


def measure(jobs: list[tuple[str, str, str]]) -> pd.DataFrame:
    cv2.setNumThreads(1)
    with Pool(N_WORKERS) as pool:
        rows = [r for r in pool.map(_measure_one, jobs, chunksize=16) if r]
    return pd.DataFrame(rows)


def quantile_map(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map each source value to the reference value at its own rank quantile.

    Rank-based rather than moment-based, so the degraded set reproduces the
    reference *distribution* rather than only its median.
    """
    n = len(source)
    ranks = source.argsort().argsort()
    q = ranks / max(n - 1, 1)
    return np.quantile(reference, q)


# ---------------------------------------------------------------------------
# Variant: blur
# ---------------------------------------------------------------------------
def _ge_through_jpeg(bgr, mask, sigma: float) -> float:
    """gradient_energy of the image as it will be WRITTEN, not as held in memory.

    Not optional. Solving on the in-memory array leaves the written file
    1.04-1.12x sharper than its target, because q95 re-encoding reintroduces
    high-frequency ringing -- enough to leave a residual Cliff's delta of +0.34
    where the solve believed it had reached zero. Measure the artifact.
    """
    img = bgr if sigma <= 1e-6 else cv2.GaussianBlur(bgr, (0, 0), sigma)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ok:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return gradient_energy(_gray(img), mask)


def _load_for_solve(path: str):
    """(bgr, eroded ROI) for one green crop, or None.

    Uses quality_metrics' own `_gray`/`_prepare_mask` rather than a local
    reimplementation: `_gray` returns float64, and computing gradient_energy on a
    uint8 array makes `np.diff` wrap modulo 256, so a hand-rolled version
    silently reports garbage. The ROI comes from the UNBLURRED crop and is held
    fixed across sigmas -- blurring bleeds green into the face edge and would
    move the mask under the measurement.
    """
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    raw = face_mask_from_green(bgr)
    if raw is None:
        return None
    mask = _prepare_mask(raw, bgr.shape[:2])
    if mask is None:
        return None
    return bgr, mask


def _solve_identity_sigma(args: tuple[str, list[str], float]) -> dict[str, Any] | None:
    """One sigma per BAT, solved so that bat's MEDIAN lands on its target.

    Per-identity rather than per-image, for two reasons. Statistically, the unit
    of analysis throughout this workstream (and in docs/quality_parity.md) is the
    per-identity median, and matching pooled per-image distributions does not
    match identity medians -- rousettus' per-image tail reaches 560 while no
    rousettus identity median comes close, so a per-image rank map pushes the
    sharpest mauritius bats far above any real rousettus bat. Physically, each
    bat was filmed in exactly ONE video, so capture quality is a property of the
    bat, not of the frame; a per-frame factor would model a camera that changed
    between frames of one clip.
    """
    identity, paths, target = args
    loaded = [x for x in (_load_for_solve(p) for p in paths) if x is not None]
    if not loaded:
        return None

    def med(sigma: float) -> float:
        return float(np.median([_ge_through_jpeg(b, m, sigma) for b, m in loaded]))

    base = med(0.0)
    if base <= target:
        return {"identity": identity, "sigma": 0.0, "ge_before": base, "ge_after": base,
                "target": target, "n_images": len(loaded), "reachable": True,
                "no_blur_needed": True}
    if med(SIGMA_MAX) > target:
        return {"identity": identity, "sigma": SIGMA_MAX, "ge_before": base,
                "ge_after": med(SIGMA_MAX), "target": target, "n_images": len(loaded),
                "reachable": False, "no_blur_needed": False}
    lo, hi = 0.0, SIGMA_MAX
    for _ in range(BISECT_ITERS):
        mid = (lo + hi) / 2.0
        if med(mid) > target:
            lo = mid
        else:
            hi = mid
    sigma = (lo + hi) / 2.0
    return {"identity": identity, "sigma": sigma, "ge_before": base, "ge_after": med(sigma),
            "target": target, "n_images": len(loaded), "reachable": True,
            "no_blur_needed": False}


def _write_blurred(args: tuple[str, str, float]) -> bool:
    src, dst, sigma = args
    img = cv2.imread(src)
    if img is None:
        return False
    out = img if sigma <= 1e-6 else cv2.GaussianBlur(img, (0, 0), sigma)
    return bool(cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]))


def build_blur(force: bool) -> dict[str, Any]:
    src_m = measure([(p, i, SRC) for p, i in _rows(TRAIN_MANIFEST.format(sp=SRC, bg="green"))])
    ref_m = measure([(p, i, REF) for p, i in _rows(REF_MANIFEST)])

    # Rank-map mauritius identity medians onto the rousettus identity-median
    # distribution -- same unit on both sides.
    src_med = src_m.groupby("identity")["gradient_energy"].median().sort_values()
    ref_med = ref_m.groupby("identity")["gradient_energy"].median()
    targets = quantile_map(src_med.to_numpy(), ref_med.to_numpy())
    by_ident = {k: g["path"].tolist() for k, g in src_m.groupby("identity")}
    jobs = [(ident, by_ident[ident], float(tg)) for ident, tg in zip(src_med.index, targets)]

    cv2.setNumThreads(1)
    with Pool(min(N_WORKERS, len(jobs))) as pool:
        sol = pd.DataFrame([r for r in pool.map(_solve_identity_sigma, jobs) if r])
    sigma_by_ident = dict(zip(sol["identity"], sol["sigma"]))

    # One sigma per bat, applied to BOTH backgrounds: the degradation models how
    # that bat's video was captured, not how its background was treated.
    written = {}
    for bg in BACKGROUNDS:
        outdir = pathlib.Path(f"data/processed/{SRC}/video/not_augmented/blur/{bg}_bg")
        outdir.mkdir(parents=True, exist_ok=True)
        jobs2 = [
            (p, str(outdir / pathlib.Path(p).name), float(sigma_by_ident[i]))
            for p, i in _rows(TRAIN_MANIFEST.format(sp=SRC, bg=bg))
            if i in sigma_by_ident
        ]
        with Pool(N_WORKERS) as pool:
            written[bg] = int(sum(pool.map(_write_blurred, jobs2, chunksize=16)))
        _emit_manifest(TRAIN_MANIFEST.format(sp=SRC, bg=bg), outdir,
                       f"data/manifests/{SRC}_blur_{bg}_manifest.csv", force)

    s = sol["sigma"].to_numpy()
    summary = {
        "variant": "blur", "granularity": "per-identity (one sigma per bat)",
        "n_identities": len(sol), "n_written": written,
        "sigma_median": float(np.median(s)), "sigma_min": float(s.min()),
        "sigma_max": float(s.max()),
        "n_no_blur_needed": int(sol["no_blur_needed"].sum()),
        "n_target_unreachable": int((~sol["reachable"]).sum()),
        "target_metric": "gradient_energy",
        "target_unit": "per-identity median",
        "reference": REF_MANIFEST, "reference_n_identities": int(ref_med.size),
        "per_identity": sol.to_dict("records"),
    }
    print(f"[blur] solved {len(sol)} per-identity sigmas: "
          f"min={s.min():.3f} median={np.median(s):.3f} max={s.max():.3f}\n"
          f"       {summary['n_no_blur_needed']} bats needed no blur, "
          f"{summary['n_target_unreachable']} could not reach target\n"
          f"       wrote {written}")
    sol.to_parquet(REVIEW_DIR / "blur_solve.parquet")
    return summary


# ---------------------------------------------------------------------------
# Helpers shared by both variants
# ---------------------------------------------------------------------------
def _rows(manifest_csv: str) -> list[tuple[str, str]]:
    m = manifest_from_csv(pathlib.Path(manifest_csv))
    return [(str(r.path), r.identity) for r in m.records]


def _emit_manifest(source_csv: str, outdir: pathlib.Path, out_csv: str, force: bool) -> None:
    src = manifest_from_csv(pathlib.Path(source_csv))
    out = pathlib.Path(out_csv)
    if out.exists() and not force:
        print(f"       {out} exists; pass --force to overwrite")
        return
    records = []
    for r in src.records:
        dst = outdir / pathlib.Path(r.path).name
        if not dst.exists():
            continue
        records.append(
            ImageRecord(
                path=dst, identity=r.identity, species=r.species, background=r.background,
                source=r.source, augmented=r.augmented, split=r.split,
                quality=float(compute_quality(dst)),  # new pixels -> recompute
            )
        )
    man = Manifest.from_records(records)
    man.assert_identity_disjoint()
    manifest_to_csv(man, out)
    print(f"       manifest {out}  n={len(records)} ids={len(man.identities())}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("blur", "recrop"), required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cpu", help="recrop only: YOLO device")
    args = ap.parse_args()

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    if args.variant == "blur":
        summary = build_blur(args.force)
    else:
        from build_resolution_recrop import build_recrop  # split for readability
        summary = build_recrop(args.force, args.device)

    out = REVIEW_DIR / f"{args.variant}_build.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}")

    if args.verify:
        from verify_resolution_matched import verify
        verify(args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
