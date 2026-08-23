"""Compute the per-image quality battery over every manifest.

Walks the aligned manifests for both species and all three background variants,
runs :func:`bat_data.quality_metrics.compute_battery` on each image, and joins
the native head-box size from ``measure_native_boxes.py``. The result is the
input to ``analyze_quality_parity.py``.

Face-only measurement: for the ``green`` variant the background is flat green,
so the face ROI is recoverable exactly and every metric is restricted to it.
For ``original`` and ``random`` there is no such handle, so metrics cover the
whole crop and therefore include background texture — the ``roi_source`` column
records which happened. **Cross-species sharpness claims should be read off the
green rows**, where background clutter cannot inflate a gradient measure.

Output is one combined Parquet table (rather than a file per species/background)
because every downstream test groups across species, background and identity.

Usage:
    uv run python scripts/build_quality_battery.py
    uv run python scripts/build_quality_battery.py --species mauritius --background green
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from bat_data.quality_metrics import BATTERY_METRICS, compute_battery

SPECIES = ("mauritius", "rousettus")
BACKGROUNDS = ("green", "original", "random")
MANIFEST_TEMPLATE = "data/manifests/{species}_{background}_aligned_{edge}_manifest.csv"

DEFAULT_OUT = Path("outputs/quality/battery.parquet")
NATIVE_BOXES = Path("outputs/quality/native_boxes.parquet")


def load_native_boxes(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  ! {path} not found — native_side_px will be absent")
        print("    run: uv run python scripts/measure_native_boxes.py")
        return None
    cols = ["basename", "species", "native_side_px", "mask_area_frac"]
    df = pd.read_parquet(path)
    keep = [c for c in cols if c in df.columns]
    keep += [c for c in df.columns if c.startswith("upscale_")]
    return df[keep]


def measure_manifest(
    manifest_path: Path,
    species: str,
    background: str,
    *,
    include_nriqa: bool,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, object]] = []
    n_unreadable = 0
    started = time.time()

    for record in manifest.itertuples(index=False):
        metrics = compute_battery(record.path, include_nriqa=include_nriqa)
        if metrics is None:
            n_unreadable += 1
            continue
        rows.append(
            {
                "path": record.path,
                "basename": Path(record.path).name,
                "identity": record.identity,
                "species": species,
                "background": background,
                "split": record.split,
                "manifest_quality": float(record.quality),
                **metrics,
            }
        )

    elapsed = time.time() - started
    note = f" ({n_unreadable} unreadable)" if n_unreadable else ""
    print(f"  {species:<10} {background:<9} {len(rows):5d} images in {elapsed:5.1f}s{note}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", choices=(*SPECIES, "all"), default="all")
    ap.add_argument("--background", choices=(*BACKGROUNDS, "all"), default="all")
    ap.add_argument("--edge", type=int, default=224, help="Aligned manifest edge length.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--native-boxes", type=Path, default=NATIVE_BOXES)
    ap.add_argument(
        "--include-nriqa",
        action="store_true",
        help="Also compute BRISQUE (requires piq; exploratory only).",
    )
    args = ap.parse_args()

    species_list = SPECIES if args.species == "all" else (args.species,)
    bg_list = BACKGROUNDS if args.background == "all" else (args.background,)

    print("=== quality battery ===")
    frames: list[pd.DataFrame] = []
    for species in species_list:
        for background in bg_list:
            manifest_path = Path(
                MANIFEST_TEMPLATE.format(species=species, background=background, edge=args.edge)
            )
            if not manifest_path.exists():
                print(f"  {species:<10} {background:<9} SKIP — {manifest_path} not found")
                continue
            frames.append(
                measure_manifest(
                    manifest_path, species, background, include_nriqa=args.include_nriqa
                )
            )

    if not frames:
        raise SystemExit("no manifests measured")

    df = pd.concat(frames, ignore_index=True)

    native = load_native_boxes(args.native_boxes)
    if native is not None:
        before = len(df)
        df = df.merge(native, on=["basename", "species"], how="left")
        if len(df) != before:
            raise SystemExit(
                f"native-box join changed row count: {before} -> {len(df)} "
                "(duplicate basenames?)"
            )
        missing = int(df["native_side_px"].isna().sum())
        if missing:
            print(f"  ! {missing}/{len(df)} rows have no native_side_px")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows, {len(df.columns)} columns)")

    if args.include_nriqa and "brisque" not in df.columns:
        print("  ! BRISQUE unavailable — `piq` is not installed in this environment")

    print("\n=== per-identity medians, then median across identities ===")
    metrics = [m for m in (*BATTERY_METRICS, "native_side_px") if m in df.columns]
    for background in sorted(df["background"].unique()):
        sub = df[df["background"] == background]
        roi = sorted(sub["roi_source"].unique())
        print(f"\n  background={background}  (roi_source={'/'.join(roi)})")
        print(f"    {'metric':<24} {'mauritius':>12} {'rousettus':>12}  ratio")
        per_id = sub.groupby(["species", "identity"])[metrics].median()
        for metric in metrics:
            vals = {}
            for species in SPECIES:
                if species in per_id.index.get_level_values(0):
                    vals[species] = float(per_id.loc[species, metric].median())
            if len(vals) < 2:
                continue
            mau, rou = vals["mauritius"], vals["rousettus"]
            ratio = mau / rou if abs(rou) > 1e-12 else float("nan")
            print(f"    {metric:<24} {mau:12.3f} {rou:12.3f}  {ratio:6.2f}x")


if __name__ == "__main__":
    main()
