"""Build a face-removed ("background only") training set.

Why this exists
---------------
Every bat was filmed in exactly one video, so a same-bat pair always shares a
background and a different-bat pair never does. ``scripts/probe_background_leakage.py``
showed that cue is worth ROC-AUC 0.740 (mauritius) / 0.792 (rousettus) with **no
model at all**, and no trained model significantly beats it. That probe answers
"is the cue available?". This build answers the complementary, causal question:
**can a real model, trained end to end on images with no face in them, identify
held-out bats?**

The existing ``make_C`` arm in ``scripts/occlusion_probe.py`` cannot serve here:
it ablates only the *test* split, so the model it evaluates was trained on faces.

Why inpainting rather than a flat fill
--------------------------------------
``make_C`` flat-fills the face region and says so in its own docstring: a filled
region still has the *outline* of the face, so that arm is "background plus
residual silhouette", not background alone. Silhouette is a known open confound
in this project (``docs/saliency_species.md``), so a positive result under a flat
fill would be ambiguous. ``cv2.inpaint`` extrapolates the surrounding background
inward and leaves no outline.

The cost of that choice, stated because it bounds the interpretation
-------------------------------------------------------------------
These are tight face crops. The 24 px-dilated face mask covers a median **89%**
of a mauritius crop and **77%** of a rousettus crop, so most of each output image
is extrapolated from the surviving background rim. Two consequences:

1. No face pixel survives -- everything in the output is a function of the
   background rim alone. That is what makes the arm valid.
2. Spreading rim colour across the frame **amplifies** the background rather than
   merely preserving it. So this arm's ROC-AUC is an *upper bound* on the
   background cue's strength, not an estimate of it.

Because of (2) the arm is uninterpretable without a negative control, which is
why ``--background random`` exists: random backgrounds are redrawn per image, so
the rim carries no bat identity, yet the images stay varied (unlike ``green``,
which inpaints to a degenerate flat field). A model trained on inpainted-random
data must land at chance. If it does not, this build manufactures a cue and the
``original`` number cannot be trusted -- the same logic that validates the
model-free probe.

Usage
-----
    uv run python scripts/build_background_only.py --species mauritius --background original
    uv run python scripts/build_background_only.py --all
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from occlusion_probe import FACE_DILATE_PX, mask_from_green  # noqa: E402

from bat_core.types import ImageRecord, Manifest  # noqa: E402
from bat_data import manifest_from_csv, manifest_to_csv  # noqa: E402
from bat_data.manifest import compute_quality  # noqa: E402

SPECIES = ("mauritius", "rousettus")
BACKGROUNDS = ("original", "random", "green")

# Telea radius. 3 px is the OpenCV default and is appropriate here: the mask is
# one large connected blob, so the fill is dominated by the boundary condition
# rather than by this parameter.
INPAINT_RADIUS = 3

# Above this dilated-mask coverage there is too little real background left to
# extrapolate from and the output is pure smear. Recorded rather than silently
# dropped -- see the coverage report.
COVERAGE_WARN = 0.95

DEFAULT_OUT = pathlib.Path("outputs/leakage/bgonly_build.json")


def _source_manifest(species: str, background: str) -> pathlib.Path:
    """The manifest to derive from.

    ``original``/``green`` have per-background manifests; ``random`` is the
    unsuffixed default (see configs/experiment/*_random_*).
    """
    stem = {
        "original": f"{species}_original_bg_manifest.csv",
        "green": f"{species}_green_bg_manifest.csv",
        "random": f"{species}_manifest.csv",
    }[background]
    return pathlib.Path("data/manifests") / stem


def _matte_path(image_path: pathlib.Path, background: str) -> pathlib.Path:
    """The green-screen sibling that defines the face region.

    Basenames are shared across every background build, so this is a directory
    substitution. Verified: all 520 mauritius and all 1059 rousettus rows of the
    original/random manifests have a green_bg sibling on disk.
    """
    token = {"original": "original_bg", "random": "random_bg", "green": "green_bg"}[background]
    return pathlib.Path(str(image_path).replace(token, "green_bg"))


def remove_face(img: np.ndarray, face_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Dilate the face mask and inpaint it away. Returns (image, coverage)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FACE_DILATE_PX * 2 + 1,) * 2)
    dilated = cv2.dilate(face_mask.astype(np.uint8), kernel)
    dilated = (dilated > 0).astype(np.uint8)
    coverage = float(dilated.mean())
    if coverage >= 1.0:
        # Nothing to extrapolate from; caller records the skip.
        return img, coverage
    out = cv2.inpaint(img, dilated, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return out, coverage


def build(species: str, background: str, *, force: bool = False) -> dict:
    src_path = _source_manifest(species, background)
    manifest = manifest_from_csv(src_path)  # verifies the .hash sidecar

    outdir = pathlib.Path(f"data/processed/{species}/video/not_augmented/bgonly/{background}")
    outdir.mkdir(parents=True, exist_ok=True)
    man_out = pathlib.Path(f"data/manifests/{species}_bgonly_{background}_manifest.csv")
    if man_out.exists() and not force:
        print(f"[{species}/{background}] {man_out} exists; pass --force to rebuild")
        return {"species": species, "background": background, "skipped": True}

    records: list[ImageRecord] = []
    coverages: list[float] = []
    n_written = n_skipped = n_high = 0
    for rec in manifest.records:
        p = pathlib.Path(rec.path)
        img = cv2.imread(str(p))
        mask = mask_from_green(_matte_path(p, background), img.shape[:2]) if img is not None else None
        if img is None or mask is None or (mask > 0).sum() == 0:
            n_skipped += 1
            continue
        out_img, coverage = remove_face(img, mask)
        if coverage >= 1.0:
            n_skipped += 1
            continue
        coverages.append(coverage)
        n_high += int(coverage >= COVERAGE_WARN)
        dst = outdir / rec.identity / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_written += 1
        # `background` stays the source literal -- bat_core.ImageRecord pins it to
        # {green, random, original}. The arm is labelled at run time via the
        # Hydra `data.background` key, which is free text.
        records.append(
            ImageRecord(
                path=dst,
                identity=rec.identity,
                species=rec.species,
                background=rec.background,
                source=rec.source,
                augmented=rec.augmented,
                split=rec.split,
                quality=float(compute_quality(dst)),  # recomputed: these are new pixels
            )
        )

    new_manifest = Manifest.from_records(records)
    new_manifest.assert_identity_disjoint()
    manifest_to_csv(new_manifest, man_out)

    cov = np.array(coverages) if coverages else np.array([0.0])
    summary = {
        "species": species,
        "background": background,
        "source_manifest": str(src_path),
        "source_manifest_hash": manifest.manifest_hash,
        "out_manifest": str(man_out),
        "out_manifest_hash": new_manifest.manifest_hash,
        "out_dir": str(outdir),
        "n_written": n_written,
        "n_skipped": n_skipped,
        "n_identities": len(new_manifest.identities()),
        "split_counts": {s: len(new_manifest.filter_split(s)) for s in ("train", "val", "test")},
        "dilate_px": FACE_DILATE_PX,
        "inpaint_radius": INPAINT_RADIUS,
        "coverage_median": float(np.median(cov)),
        "coverage_p90": float(np.percentile(cov, 90)),
        "coverage_max": float(cov.max()),
        f"n_coverage_ge_{COVERAGE_WARN}": n_high,
    }
    print(
        f"[{species}/{background}] wrote {n_written} (skipped {n_skipped}) -> {outdir}\n"
        f"    manifest={man_out}  ids={summary['n_identities']} splits={summary['split_counts']}\n"
        f"    dilated-mask coverage: median={summary['coverage_median']:.3f} "
        f"p90={summary['coverage_p90']:.3f} max={summary['coverage_max']:.3f} "
        f"(>={COVERAGE_WARN}: {n_high})"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", choices=SPECIES)
    ap.add_argument("--background", choices=BACKGROUNDS, default="original")
    ap.add_argument("--all", action="store_true", help="both species x {original, random}")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    jobs = (
        [(sp, bg) for sp in SPECIES for bg in ("original", "random")]
        if args.all
        else [(args.species, args.background)]
    )
    if any(sp is None for sp, _ in jobs):
        ap.error("--species is required unless --all is given")

    summaries = [build(sp, bg, force=args.force) for sp, bg in jobs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
