"""Smoke-test the two species-specific face-segmentation checkpoints.

Gate for ``measure_native_boxes.py``: before we trust ``mask_side_px`` as a
native-resolution measurement we have to know both seg models still detect
faces on their own frame sets. mauritius uses ``face_seg_mauritius_v2.pt``;
rousettus uses the legacy still-image checkpoint under ``legacy/`` (see
``SPECIES_DEFAULTS`` in ``build_variants_from_frames.py``) — they were trained
separately and are not interchangeable.

Reports per species: detection rate, mask-area fraction, and the
``mask_side_px`` distribution (the aligner's source-crop side in frame px, the
number the native-resolution comparison rests on). Writes an annotated contact
sheet so the masks can be eyeballed rather than trusted on statistics alone.

Usage:
    uv run python scripts/smoke_test_seg_models.py --species mauritius --n 10
    uv run python scripts/smoke_test_seg_models.py --species rousettus --n 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_variants_from_frames import (  # noqa: E402
    SPECIES_DEFAULTS,
    collect_groups,
    mask_side_px,
)

from bat_preprocessing import YOLOSegmenter  # noqa: E402

# Matches ``build_variants_from_frames.py --margin`` so the reported
# ``mask_side_px`` is the same quantity the dataset build measured.
BUILD_MARGIN = 0.03

# Acceptance thresholds (see the plan's A0 gate).
MIN_DETECTION_RATE = 0.90
# Calibrated 2026-08-04 against both frame sets: mauritius medians ~6% of the
# 1080x1920 frame, rousettus ~1.8% (bats held further from the camera — still a
# ~280px face, so a low area fraction is framing, not a mask failure). The floor
# only has to catch degenerate specks; the upper bound catches whole-frame masks.
PLAUSIBLE_AREA_FRAC = (0.005, 0.40)


def sample_frames(species: str, frames_root: Path, n: int) -> list[Path]:
    """Take up to *n* frames spread evenly across identities."""
    defaults = SPECIES_DEFAULTS[species]
    groups = collect_groups(frames_root, defaults["prefix"], defaults["layout"], sample=0)
    if not groups:
        raise SystemExit(f"no frames found under {frames_root} (layout={defaults['layout']})")

    picked: list[Path] = []
    identities = sorted(groups)
    # Round-robin over identities so a single bat can't dominate the sample.
    idx = 0
    while len(picked) < n:
        added = False
        for identity in identities:
            items = groups[identity]
            if idx < len(items):
                picked.append(items[idx][0])
                added = True
                if len(picked) >= n:
                    break
        if not added:
            break
        idx += 1
    return picked


def contact_sheet(panels: list[np.ndarray], cols: int = 5, cell: int = 320) -> np.ndarray:
    """Tile *panels* into a grid, each resized to a ``cell``-px square."""
    tiles = []
    for img in panels:
        h, w = img.shape[:2]
        scale = cell / max(h, w)
        resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        canvas = np.zeros((cell, cell, 3), dtype=np.uint8)
        rh, rw = resized.shape[:2]
        canvas[:rh, :rw] = resized
        tiles.append(canvas)

    while len(tiles) % cols:
        tiles.append(np.zeros((cell, cell, 3), dtype=np.uint8))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def annotate(frame: np.ndarray, mask: np.ndarray | None, label: str) -> np.ndarray:
    """Overlay the mask in green plus its bbox, and stamp *label*."""
    out = frame.copy()
    if mask is not None:
        overlay = out.copy()
        overlay[mask > 0] = (0, 255, 0)
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        ys, xs = np.where(mask > 0)
        if xs.size:
            cv2.rectangle(
                out,
                (int(xs.min()), int(ys.min())),
                (int(xs.max()), int(ys.max())),
                (0, 0, 255),
                2,
            )
    colour = (0, 255, 255) if mask is not None else (0, 0, 255)
    cv2.putText(out, label, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", choices=tuple(SPECIES_DEFAULTS), required=True)
    ap.add_argument("--n", type=int, default=10, help="Frames to test.")
    ap.add_argument("--frames-root", default="", help="Override the species default frames root.")
    ap.add_argument("--seg-weights", default="", help="Override the species default checkpoint.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="outputs/quality/seg_smoke")
    args = ap.parse_args()

    defaults = SPECIES_DEFAULTS[args.species]
    frames_root = Path(args.frames_root or defaults["frames_root"])
    seg_weights = Path(args.seg_weights or defaults["seg_weights"])
    if not seg_weights.exists():
        raise SystemExit(f"seg checkpoint not found: {seg_weights}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sample_frames(args.species, frames_root, args.n)
    print(f"[{args.species}] seg={seg_weights.name} device={args.device}")
    print(f"  frames_root={frames_root}")
    print(f"  testing {len(frames)} frames\n")

    seg = YOLOSegmenter({"weights": str(seg_weights), "device": args.device})

    panels: list[np.ndarray] = []
    n_detected = 0
    n_read = 0
    area_fracs: list[float] = []
    sides: list[float] = []

    for fp in frames:
        frame = cv2.imread(str(fp))
        if frame is None:
            print(f"  UNREADABLE {fp.name}")
            continue
        n_read += 1
        h, w = frame.shape[:2]
        pred = seg.predict(frame)

        if pred is None:
            panels.append(annotate(frame, None, "NO MASK"))
            print(f"  {fp.name:<52} {w}x{h}  NO MASK")
            continue

        n_detected += 1
        area_frac = float(np.sum(pred.mask > 0)) / float(h * w)
        side = mask_side_px(pred.mask, (h, w), BUILD_MARGIN)
        area_fracs.append(area_frac)
        if side is not None:
            sides.append(side)

        panels.append(annotate(frame, pred.mask, f"area {area_frac:.1%} side {side:.0f}px"))
        print(f"  {fp.name:<52} {w}x{h}  area={area_frac:6.2%}  side={side:7.1f}px")

    if not n_read:
        raise SystemExit("no readable frames — check --frames-root")

    detection_rate = n_detected / n_read
    sheet_path = out_dir / f"{args.species}_seg_smoke.png"
    cv2.imwrite(str(sheet_path), contact_sheet(panels))

    print(f"\n=== {args.species} summary ===")
    print(f"  detection rate   : {n_detected}/{n_read} = {detection_rate:.0%}")
    if area_fracs:
        print(
            f"  mask area frac   : min={min(area_fracs):.2%} "
            f"median={statistics.median(area_fracs):.2%} max={max(area_fracs):.2%}"
        )
    if sides:
        print(
            f"  mask_side_px     : min={min(sides):.0f} "
            f"median={statistics.median(sides):.0f} max={max(sides):.0f}"
        )
        for edge in (224, 320):
            n_up = sum(1 for s in sides if s < edge)
            print(f"    upscaled @ {edge}px: {n_up}/{len(sides)} ({100.0 * n_up / len(sides):.0f}%)")
    print(f"  contact sheet    : {sheet_path}")

    # Acceptance gate — report, don't raise: a soft failure still leaves the
    # contact sheet for inspection, which is the point of the smoke test.
    problems = []
    if detection_rate < MIN_DETECTION_RATE:
        problems.append(f"detection rate {detection_rate:.0%} < {MIN_DETECTION_RATE:.0%}")
    if area_fracs:
        median_area = statistics.median(area_fracs)
        lo, hi = PLAUSIBLE_AREA_FRAC
        if not lo <= median_area <= hi:
            problems.append(f"median mask area {median_area:.1%} outside {lo:.0%}-{hi:.0%}")
    print("  GATE             : " + ("PASS" if not problems else "FAIL — " + "; ".join(problems)))


if __name__ == "__main__":
    main()
