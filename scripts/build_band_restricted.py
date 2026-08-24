"""Restrict both species to the native-resolution band they share.

Why this exists
---------------
``docs/quality_parity.md`` establishes that effective resolution separates the
species **completely**: median native head-box 614 px (mauritius) vs 312 px
(rousettus), Cliff's delta = +1.00, with mauritius' minimum (441 px at identity
level) above rousettus' maximum (407 px). So "rousettus individuals are less
distinctive" is confounded with "rousettus footage is worse", and the intrinsic
separability analysis showed that matching resolution collapses the d' gap.

This build produces the training-side version of that control: keep only images
whose native head-box falls in the band both species occupy, so the species
comparison runs on **real in-band pixels** rather than simulated degradation.
It is the companion to ``scripts/build_resolution_matched.py`` (arm 3A), which
degrades mauritius instead and keeps all 16 bats. The two fail in different
directions -- this one drops bats, that one invents pixels -- so agreement
between them is much stronger than either alone.

Band definition
---------------
Taken verbatim from the published analysis (``scripts/analyze_separability.py``,
``outputs/quality/separability.json``): ``lo = max(per-species min)``,
``hi = min(per-species max)`` over ``native_side_px`` -> **[297.86, 623.28] px**.
Recomputed here rather than hardcoded, then asserted against the published value
so this arm stays tied to the documented number.

Both edges are set by single extreme images and there is no quantile robustness;
that is a property of the published band, kept deliberately for comparability.

Identity floor
--------------
Identities with fewer than ``--min-images`` in-band images are dropped, leaving
**10 of 16 mauritius** and **11 of 12 rousettus** at the default of 10. Note
rousettus ``vanilaice`` sits exactly on the boundary (10 in-band images, next
lowest is 29), so 11-of-12 is knife-edge.

No images-per-identity cap
--------------------------
``docs/quality_parity.md`` suggests combining the band with a per-identity cap.
That is right for the *separability* analysis, where the measure is geometric and
unbalanced clusters bias it. It is wrong here: the binding minimum is 10, so a
cap would leave ~100 images in total, which will not train. Uncapped keeps the
same ~2x species image asymmetry the published sweep already has, so the arm
stays comparable to it. Per-identity counts are reported instead -- rousettus
``arrowhead`` alone holds 165 of 656 in-band images.

Usage
-----
    uv run python scripts/build_band_restricted.py
    uv run python scripts/build_band_restricted.py --report-folds
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import pandas as pd

from bat_core.types import ImageRecord, Manifest
from bat_data import manifest_from_csv, manifest_to_csv
from bat_data.splitter import IdentitySplitter

NATIVE_BOXES = pathlib.Path("outputs/quality/native_boxes.parquet")
PUBLISHED_BAND = (297.86, 623.28)  # outputs/quality/separability.json
BAND_TOL = 0.01

# Source manifests. rousettus green uses the 1059-image INTERSECTION (Phase 2C):
# the full green build has 34 extra frames with no original_bg counterpart, and
# using it would make the green and original band arms differ by image set as
# well as by background. Verified: intersect and original share basenames exactly.
SOURCES = {
    ("mauritius", "green"): "data/manifests/mauritius_green_bg_manifest.csv",
    ("mauritius", "original"): "data/manifests/mauritius_original_bg_manifest.csv",
    ("mauritius", "random"): "data/manifests/mauritius_manifest.csv",
    ("rousettus", "green"): "data/manifests/rousettus_green_bg_intersect_manifest.csv",
    ("rousettus", "original"): "data/manifests/rousettus_original_bg_manifest.csv",
    ("rousettus", "random"): "data/manifests/rousettus_manifest.csv",
}
# `random` is included because it is the ONLY background on which the published
# sweep found a species difference (ArcFace +0.116, AdaFace +0.136; green is
# -0.043, i.e. rousettus slightly ahead). An arm that controls the image-quality
# confound only on green and original never tests its own premise. The unsuffixed
# `{species}_manifest.csv` is the random build -- see configs/experiment/*_random_*.

# Re-derived fold bounds for the smaller identity pools. The published bounds
# assume 16 / 12 bats; at 10 bats mauritius' `max_test 6` + `min_train 6` would
# clamp test_hi to min(6, 10-2-6) = 2 (splitter.py:217), pinning every fold to
# exactly 2 test identities and destroying the test-size variation that the whole
# k-fold design is built to measure.
# Both pools come out at n=10 (see the rousettus note below), so the bounds are
# deliberately IDENTICAL across species: a paired species comparison wants the
# two arms drawing fold shapes from the same distribution, otherwise part of any
# observed difference is just a different train/test ratio.
#
# The fractions are recorded for traceability but do not govern sizing here --
# splitter.py:187 uses an explicit min_*_identities in preference to the
# fraction, and every bound below is explicit.
_BAND_BOUNDS = dict(
    val_fraction=0.20, test_fraction=0.20,
    min_val_identities=2, max_val_identities=2,
    min_test_identities=2, max_test_identities=4,
    min_train_identities=4,
)
FOLD_BOUNDS = {"mauritius": dict(_BAND_BOUNDS), "rousettus": dict(_BAND_BOUNDS)}
FOLD_BASE_SEED = 1000
N_FOLDS = 20


def resolve_band(native: pd.DataFrame) -> tuple[float, float]:
    lo = max(float(g["native_side_px"].min()) for _, g in native.groupby("species"))
    hi = min(float(g["native_side_px"].max()) for _, g in native.groupby("species"))
    for got, want, name in ((lo, PUBLISHED_BAND[0], "lo"), (hi, PUBLISHED_BAND[1], "hi")):
        if abs(got - want) > BAND_TOL:
            raise SystemExit(
                f"band {name} = {got:.2f} does not match the published "
                f"{want:.2f}; refusing to build an arm that is not the documented one"
            )
    return lo, hi


def in_band_frame(source: str, native: pd.DataFrame, species: str, band: tuple[float, float]):
    manifest = manifest_from_csv(pathlib.Path(source))  # verifies the .hash sidecar
    frame = pd.DataFrame(
        [
            {
                "path": str(r.path), "basename": pathlib.Path(r.path).name,
                "identity": r.identity, "species": r.species, "background": r.background,
                "source": r.source, "augmented": r.augmented, "quality": r.quality,
            }
            for r in manifest.records
        ]
    )
    nat = native[native.species == species][["basename", "native_side_px"]]
    merged = frame.merge(nat, on="basename", how="left")
    if merged["native_side_px"].isna().any():
        raise SystemExit(f"{source}: {merged['native_side_px'].isna().sum()} rows have no native box")
    return merged[merged["native_side_px"].between(*band)].copy(), manifest.manifest_hash


def build(species: str, background: str, native: pd.DataFrame, band, min_images: int) -> dict[str, Any]:
    source = SOURCES[(species, background)]
    frame, src_hash = in_band_frame(source, native, species, band)

    counts = frame.groupby("identity").size().sort_values()
    keep = sorted(counts[counts >= min_images].index)
    dropped = {k: int(v) for k, v in counts[counts < min_images].items()}
    frame = frame[frame.identity.isin(keep)]

    bounds = FOLD_BOUNDS[species]
    splitter = IdentitySplitter(
        val_fraction=bounds["val_fraction"], test_fraction=bounds["test_fraction"],
        seed=42, size_mode="exact", min_per_split=2,
    )
    records = [
        ImageRecord(
            path=pathlib.Path(r.path), identity=r.identity, species=r.species,
            background=r.background, source=r.source, augmented=bool(r.augmented),
            split="train", quality=float(r.quality),
        )
        for r in frame.itertuples()
    ]
    split = splitter.split(Manifest.from_records(records))
    split.assert_identity_disjoint()

    out = pathlib.Path(f"data/manifests/{species}_band_{background}_manifest.csv")
    manifest_to_csv(split, out)

    per_id = {k: int(v) for k, v in sorted(frame.groupby("identity").size().items())}
    summary = {
        "species": species, "background": background,
        "source_manifest": source, "source_manifest_hash": src_hash,
        "out_manifest": str(out), "out_manifest_hash": split.manifest_hash,
        "band_px": list(band), "min_images_per_identity": min_images,
        "n_images": len(frame), "n_identities": len(keep),
        "identities_kept": keep, "identities_dropped": dropped,
        "images_per_identity": per_id,
        "baked_split_counts": {s: len(split.filter_split(s)) for s in ("train", "val", "test")},
        "fold_bounds": bounds,
    }
    print(
        f"[{species}/{background}] in-band {len(frame)} imgs / {len(keep)} ids "
        f"(dropped {len(dropped)}: {dropped})\n"
        f"    -> {out}  baked split {summary['baked_split_counts']}\n"
        f"    images/identity: min={min(per_id.values())} max={max(per_id.values())}"
    )
    return summary


def report_folds(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """The gate: confirm test size actually varies across the 20 folds.

    A pool of 10-11 identities is small enough that the bounds can silently pin
    every fold to the same shape, which would make the k-fold spread measure
    nothing but training noise.
    """
    out: dict[str, Any] = {}
    print("\n=== realised fold shapes (the 2B gate) ===")
    for species in ("mauritius", "rousettus"):
        s = next(x for x in summaries if x["species"] == species)
        ids = s["identities_kept"]
        recs = [
            ImageRecord(
                path=pathlib.Path(f"x/{i}.jpg"), identity=i, species=species,  # type: ignore[arg-type]
                background="green", source="video", augmented=False, split="train", quality=1.0,
            )
            for i in ids
        ]
        m = Manifest.from_records(recs)
        b = FOLD_BOUNDS[species]
        shapes = []
        for fold in range(N_FOLDS):
            sp = IdentitySplitter(
                val_fraction=b["val_fraction"], test_fraction=b["test_fraction"],
                seed=FOLD_BASE_SEED + fold, size_mode="seeded", min_per_split=1,
                min_val_identities=b["min_val_identities"], max_val_identities=b["max_val_identities"],
                min_test_identities=b["min_test_identities"], max_test_identities=b["max_test_identities"],
                min_train_identities=b["min_train_identities"],
            ).split(m)
            shapes.append(
                (len(sp.identities("train")), len(sp.identities("val")), len(sp.identities("test")))
            )
        dist: dict[str, int] = {}
        for sh in shapes:
            dist[f"{sh[0]}/{sh[1]}/{sh[2]}"] = dist.get(f"{sh[0]}/{sh[1]}/{sh[2]}", 0) + 1
        n_test_vals = sorted({sh[2] for sh in shapes})
        varies = len(n_test_vals) > 1
        print(f"  {species} (n={len(ids)}): train/val/test shapes over {N_FOLDS} folds -> {dist}")
        print(f"    n_test values: {n_test_vals}   VARIES: {varies}"
              f"{'' if varies else '   <-- GATE FAILS, bounds pin the split'}")
        out[species] = {"n_identities": len(ids), "shape_distribution": dist,
                        "n_test_values": n_test_vals, "test_size_varies": varies}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-images", type=int, default=10)
    ap.add_argument("--native-boxes", type=pathlib.Path, default=NATIVE_BOXES)
    ap.add_argument("--report-folds", action="store_true", help="only run the fold-shape gate")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("outputs/quality/band_restricted_build.json"))
    args = ap.parse_args()

    native = pd.read_parquet(args.native_boxes)[["basename", "species", "native_side_px"]]
    band = resolve_band(native)
    print(f"shared native-resolution band: [{band[0]:.2f}, {band[1]:.2f}] px "
          f"(matches published {PUBLISHED_BAND})\n")

    summaries = [
        build(sp, bg, native, band, args.min_images)
        for sp in ("mauritius", "rousettus")
        for bg in ("green", "original", "random")
    ]
    gate = report_folds(summaries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"band_px": list(band), "arms": summaries, "fold_gate": gate}, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0 if all(v["test_size_varies"] for v in gate.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
