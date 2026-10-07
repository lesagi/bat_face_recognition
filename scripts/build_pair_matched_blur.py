"""Blur each mauritius bat until it is as sharp as the rousettus bat it is paired with.

The companion to `build_pair_matched_recrop.py`, and necessary because that one
under-degrades by construction. Re-cropping matches *resolution* -- it gives the
mauritius face the same number of source pixels as its partner -- but resolution and
sharpness are not the same thing. A well-focused mauritius face downsampled 2x is
still sharper than a slightly soft rousettus face of the same pixel size, because
downsampling cannot reproduce focus error or motion blur, and it *removes* sensor
noise while mauritius is the noisier species. Measured on the 12 pairs, re-cropping
took the sharpness ratio from 4.15 to 1.82 -- better than half, nowhere near 1.

So this arm closes the measured gap directly: per bat, solve for the Gaussian sigma
whose gradient energy matches the paired rousettus bat's. The two arms answer
different questions and both are worth having. `recrop` is physically motivated and
reports whatever it achieves; `blur` matches by construction but is not a thing a
camera does. A result that survives both is not a resolution artifact.

Two traps, both previously paid for in this project:

* **The JPEG round-trip belongs inside the objective.** The output is written at
  quality 95, and that re-encode adds ringing worth 1.04-1.12x of gradient energy.
  Solving against the pre-encode image leaves a residual of about +0.34 in units of
  the quantity being matched -- the solve believes it hit zero and it did not.
* **Granularity is per bat, not per image.** Each bat is one video, so focus and
  exposure are properties of the clip. A per-image sigma models a camera whose
  focus changed between consecutive frames, and matching per-image quantiles does
  not make the identity medians agree anyway.

    uv run python scripts/build_pair_matched_blur.py --execute
"""

from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

from bat_core.types import ImageRecord, Manifest
from bat_data import manifest_from_csv, manifest_to_csv
from bat_data.manifest import compute_quality
from bat_data.quality_metrics import gradient_energy

PAIRS = pathlib.Path("outputs/quality/species_pairs.json")
# Arm-specific: two runs with different --arm must not overwrite each other.
OUT_JSON_TMPL = "outputs/quality/pair_blur_build_{arm}.json"
ROUS_MAN = "data/manifests/rousettus_green_aligned_224_manifest.csv"
JPEG_QUALITY = 95
EDGE = 224
# Bisection bounds, in pixels of the 224px crop. 6.0 already destroys the face; if a
# bat needs more than that, say so rather than silently clamping.
SIGMA_LO, SIGMA_HI, SIGMA_ITERS = 0.0, 6.0, 24


def _grey(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)


def ge_through_jpeg(bgr: np.ndarray, sigma: float) -> float:
    """Gradient energy as it will exist ON DISK: blur, then the quality-95 encode."""
    img = bgr if sigma <= 1e-6 else cv2.GaussianBlur(bgr, (0, 0), sigma)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ok:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return float(gradient_energy(_grey(img)))


def solve_sigma(images: list[np.ndarray], target: float) -> tuple[float, float]:
    """Smallest sigma whose median gradient energy meets `target`. Monotone, so bisect."""

    def med(s: float) -> float:
        return float(np.median([ge_through_jpeg(im, s) for im in images]))

    if med(SIGMA_LO) <= target:
        return SIGMA_LO, med(SIGMA_LO)
    lo, hi = SIGMA_LO, SIGMA_HI
    for _ in range(SIGMA_ITERS):
        mid = (lo + hi) / 2.0
        if med(mid) > target:
            lo = mid
        else:
            hi = mid
    return hi, med(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prefix", default="cull2")
    ap.add_argument(
        "--source-arm",
        default="paired",
        help="Arm to blur. `paired` blurs the undegraded crops; "
        "`pairres` blurs the already re-cropped ones.",
    )
    ap.add_argument("--arm", default="pairblur")
    ap.add_argument("--sample", type=int, default=25, help="Images per bat used in the solve.")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    pairs = {p["mauritius"]: p["rousettus"] for p in json.loads(PAIRS.read_text())["pairs"]}

    # Targets: each rousettus bat's own median gradient energy, measured on the
    # crops the models actually train on.
    rous: dict[str, list[float]] = {}
    for r in manifest_from_csv(pathlib.Path(ROUS_MAN)).records:
        im = cv2.imread(str(r.path))
        if im is not None:
            rous.setdefault(r.identity, []).append(float(gradient_energy(_grey(im))))
    target = {k: float(np.median(v)) for k, v in rous.items()}

    src_man = pathlib.Path(
        f"data/manifests/mauritius_{args.prefix}_{args.source_arm}_green_bg_head_{EDGE}_manifest.csv"
    )
    base_recs = manifest_from_csv(src_man).records
    by_id: dict[str, list] = {}
    for r in base_recs:
        by_id.setdefault(r.identity, []).append(r)

    print(
        f"\n  solving per-bat sigma against each pair's rousettus target "
        f"(source arm: {args.source_arm})\n"
    )
    print(
        f"  {'mauritius':<18} {'partner':<11} {'before':>8} {'target':>8} "
        f"{'sigma':>6} {'after':>8} {'ratio':>6}"
    )
    sigmas, report = {}, []
    for ident, recs in sorted(by_id.items()):
        r = pairs.get(ident)
        if r is None or r not in target:
            continue
        pick = [
            recs[i] for i in np.linspace(0, len(recs) - 1, min(args.sample, len(recs))).astype(int)
        ]
        imgs = [im for im in (cv2.imread(str(x.path)) for x in pick) if im is not None]
        if not imgs:
            continue
        before = float(np.median([ge_through_jpeg(im, 0.0) for im in imgs]))
        s, after = solve_sigma(imgs, target[r])
        sigmas[ident] = s
        report.append(
            {
                "mauritius": ident,
                "rousettus": r,
                "before": round(before, 1),
                "target": round(target[r], 1),
                "sigma": round(s, 3),
                "after": round(after, 1),
                "ratio": round(after / target[r], 3),
            }
        )
        flag = "  <- hit the sigma ceiling" if s >= SIGMA_HI - 1e-3 else ""
        print(
            f"  {ident:<18} {r:<11} {before:>8.0f} {target[r]:>8.0f} {s:>6.2f} "
            f"{after:>8.0f} {after / target[r]:>6.2f}{flag}"
        )

    if not args.execute:
        print("\n  DRY RUN -- nothing built. Re-run with --execute.\n")
        return 0

    out_recs: dict[str, list[ImageRecord]] = {"green": [], "original": []}
    n = 0
    for bg in ("green", "original"):
        man = pathlib.Path(
            f"data/manifests/mauritius_{args.prefix}_{args.source_arm}_{bg}_bg_head_{EDGE}_manifest.csv"
        )
        if not man.exists():
            print(f"  missing {man}, skipped")
            continue
        outroot = pathlib.Path(
            f"data/processed/mauritius/video/not_augmented/"
            f"{args.prefix}_{args.arm}_{bg}_bg/head/{EDGE}"
        )
        for rec in manifest_from_csv(man).records:
            s = sigmas.get(rec.identity)
            if s is None:
                continue
            im = cv2.imread(str(rec.path))
            if im is None:
                continue
            img = im if s <= 1e-6 else cv2.GaussianBlur(im, (0, 0), s)
            dst = outroot / rec.identity / pathlib.Path(rec.path).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            out_recs[bg].append(
                rec.model_copy(update={"path": dst, "quality": float(compute_quality(dst))})
            )
            n += 1

    mans = {}
    for bg, rr in out_recs.items():
        if not rr:
            continue
        m = Manifest.from_records(rr)
        m.assert_identity_disjoint()
        out = pathlib.Path(
            f"data/manifests/mauritius_{args.prefix}_{args.arm}_{bg}_bg_head_{EDGE}_manifest.csv"
        )
        manifest_to_csv(m, out)
        mans[bg] = str(out)
        print(f"  {bg:9s} {len(rr):5d} imgs -> {out.name}")

    out_json = pathlib.Path(OUT_JSON_TMPL.format(arm=args.arm))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "source_arm": args.source_arm,
                "jpeg_quality": JPEG_QUALITY,
                "n_written": n,
                "manifests": mans,
                "per_bat": report,
            },
            indent=1,
        )
    )
    print(f"\n  {n} images written\n  -> {out_json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
