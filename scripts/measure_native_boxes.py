"""Persist the native head-box size of every dataset frame.

The aligner warps a mask-centred square crop of the source frame to a fixed
edge (224 / 320). How big that crop was *before* the warp decides whether a
training image carries real detail or interpolated detail — and it is the
strongest candidate explanation for the per-species sharpness gap
(mauritius per-identity median Laplacian variance ~4x rousettus). The dataset
build already computes this number for its upscale check and then throws it
away; this script recomputes it once and writes it down.

Reuses ``mask_side_px`` and ``collect_groups`` from
``build_variants_from_frames.py`` so the measurement is the same quantity the
build measured, at the same margin, from the same frames.

Provenance (verified 2026-08-04): both species' manifests were built from every
frame in their frames root (``--sample 0``), so there is no subsampling to
replicate — measure everything and inner-join to the manifest on basename.

Usage:
    uv run python scripts/measure_native_boxes.py --species mauritius --device cpu
    uv run python scripts/measure_native_boxes.py --species rousettus --device cpu
    uv run python scripts/measure_native_boxes.py --verify-only
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_variants_from_frames import (  # noqa: E402
    SPECIES_DEFAULTS,
    collect_groups,
    mask_side_px,
)

from bat_preprocessing import YOLOSegmenter  # noqa: E402

# Must match ``build_variants_from_frames.py --margin`` for the measurement to
# describe the crops the dataset actually contains.
BUILD_MARGIN = 0.03
EDGES = (112, 224, 320)

DEFAULT_OUT = Path("outputs/quality/native_boxes.parquet")

# Manifest row counts each species must reach after the basename join. Guards
# against a silent partial join, which would bias the resolution comparison.
EXPECTED_MANIFEST_ROWS = {"mauritius": 520, "rousettus": 1092}
VERIFY_MANIFEST = "data/manifests/{species}_green_aligned_224_manifest.csv"


def measure_species(species: str, frames_root: Path, seg_weights: Path, device: str) -> pd.DataFrame:
    """Segment every frame under *frames_root* and return one row per frame."""
    defaults = SPECIES_DEFAULTS[species]
    groups = collect_groups(frames_root, defaults["prefix"], defaults["layout"], sample=0)
    if not groups:
        raise SystemExit(f"no frames found under {frames_root} (layout={defaults['layout']})")

    total = sum(len(v) for v in groups.values())
    print(f"[{species}] seg={seg_weights.name} device={device}")
    print(f"  frames_root={frames_root}")
    print(f"  {len(groups)} identities, {total} frames\n", flush=True)

    seg = YOLOSegmenter({"weights": str(seg_weights), "device": device})

    rows: list[dict[str, object]] = []
    n_done = 0
    n_no_mask = 0
    n_unreadable = 0
    started = time.time()

    for identity, items in sorted(groups.items()):
        for frame_path, out_name in items:
            frame = cv2.imread(str(frame_path))
            n_done += 1
            if frame is None:
                n_unreadable += 1
                continue

            h, w = frame.shape[:2]
            pred = seg.predict(frame)
            if pred is None:
                n_no_mask += 1
                continue

            side = mask_side_px(pred.mask, (h, w), BUILD_MARGIN)
            if side is None:
                n_no_mask += 1
                continue

            row: dict[str, object] = {
                "basename": out_name,
                "identity": identity,
                "species": species,
                "source_frame": str(frame_path),
                "native_side_px": float(side),
                "mask_area_frac": float(np.sum(pred.mask > 0)) / float(h * w),
                "frame_h": int(h),
                "frame_w": int(w),
            }
            for edge in EDGES:
                row[f"upscale_{edge}"] = bool(side < edge)
            rows.append(row)

            if n_done % 50 == 0:
                rate = n_done / max(1e-9, time.time() - started)
                eta = (total - n_done) / max(1e-9, rate)
                print(
                    f"  {n_done}/{total} frames  {rate:.2f}/s  eta {eta / 60:.1f} min",
                    flush=True,
                )

    elapsed = time.time() - started
    print(
        f"\n  measured {len(rows)}/{total} frames in {elapsed / 60:.1f} min "
        f"({n_no_mask} no-mask, {n_unreadable} unreadable)"
    )
    return pd.DataFrame(rows)


def verify_join(df: pd.DataFrame) -> None:
    """Assert every manifest row finds a measured frame, per species."""
    print("\n=== manifest join check ===")
    failures = []
    for species, expected in EXPECTED_MANIFEST_ROWS.items():
        manifest = Path(VERIFY_MANIFEST.format(species=species))
        if not manifest.exists():
            print(f"  {species:<10} SKIP — {manifest} not found")
            continue

        with manifest.open() as fh:
            manifest_names = {Path(r["path"]).name for r in csv.DictReader(fh)}
        measured = set(df.loc[df["species"] == species, "basename"])

        matched = manifest_names & measured
        missing = manifest_names - measured
        print(
            f"  {species:<10} manifest={len(manifest_names)} measured={len(measured)} "
            f"joined={len(matched)} (expected {expected})"
        )
        if missing:
            print(f"    MISSING {len(missing)}: {sorted(missing)[:5]}")
        if len(matched) != expected:
            failures.append(f"{species}: joined {len(matched)}, expected {expected}")

    if failures:
        raise SystemExit("JOIN FAILED — " + "; ".join(failures))
    print("  JOIN OK")


def summarize(df: pd.DataFrame) -> None:
    print("\n=== native head-box size (px, pre-warp) ===")
    for species, g in df.groupby("species"):
        q = g["native_side_px"].quantile([0.1, 0.5, 0.9])
        line = (
            f"  {species:<10} n={len(g):5d}  "
            f"p10={q[0.1]:6.1f}  median={q[0.5]:6.1f}  p90={q[0.9]:6.1f}"
        )
        for edge in EDGES:
            frac = 100.0 * g[f"upscale_{edge}"].mean()
            line += f"  up@{edge}={frac:5.1f}%"
        print(line)

    print("\n=== per-identity median native side ===")
    per_id = df.groupby(["species", "identity"])["native_side_px"].median()
    for species in sorted(df["species"].unique()):
        vals = per_id.loc[species].sort_values()
        print(f"  {species}: " + ", ".join(f"{k}={v:.0f}" for k, v in vals.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--species",
        choices=(*SPECIES_DEFAULTS, "all"),
        default="all",
        help="Species to measure (default: both).",
    )
    ap.add_argument("--frames-root", default="", help="Override the species default frames root.")
    ap.add_argument("--seg-weights", default="", help="Override the species default checkpoint.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-run the join check and summary against an existing parquet.",
    )
    args = ap.parse_args()

    if args.verify_only:
        if not args.out.exists():
            raise SystemExit(f"{args.out} does not exist — run a measurement first")
        df = pd.read_parquet(args.out)
        verify_join(df)
        summarize(df)
        return

    targets = tuple(SPECIES_DEFAULTS) if args.species == "all" else (args.species,)
    if len(targets) > 1 and (args.frames_root or args.seg_weights):
        raise SystemExit("--frames-root / --seg-weights require a single --species")

    frames: list[pd.DataFrame] = []
    for species in targets:
        defaults = SPECIES_DEFAULTS[species]
        frames_root = Path(args.frames_root or defaults["frames_root"])
        seg_weights = Path(args.seg_weights or defaults["seg_weights"])
        if not seg_weights.exists():
            raise SystemExit(f"seg checkpoint not found: {seg_weights}")
        frames.append(measure_species(species, frames_root, seg_weights, args.device))

    new = pd.concat(frames, ignore_index=True)

    # Merge with any previously measured species so single-species runs
    # accumulate into one file instead of clobbering it.
    if args.out.exists():
        old = pd.read_parquet(args.out)
        old = old[~old["species"].isin(new["species"].unique())]
        new = pd.concat([old, new], ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out} ({len(new)} rows)")

    summarize(new)
    verify_join(new)


if __name__ == "__main__":
    main()
