#!/usr/bin/env python3
"""One-off rousettus original_bg -> green_bg conversion.

This is intentionally a temporary data-prep script, not a supported package
entry point. It loads the legacy rousettus YOLO segmentation checkpoint,
segments each image in original_bg, paints non-mask pixels green, and writes
the result with the same filename.

Historical one-off, kept for provenance rather than reuse
---------------------------------------------------------
Not a supported entry point and not wired into the CLI. It is retained because it
is the only committed record of how a dataset the project still depends on was
produced: `data/processed/rousettus/video/not_augmented/green_bg` (1093 images),
which underpins the published rousettus green arm, the 1059-image intersection
manifest, every ROI-occlusion arm, and the six-number colour baseline. `data/` is
gitignored and the segmentation checkpoint it used lives outside the tracked tree,
so deleting this would leave no account of how those pixels came to exist.

Was `tmp_*` at the repository root until 2026-08-29; renamed here so the name says
what it is. PROJECT_ROOT was adjusted for the extra directory level at the same
time -- the defaults still resolve to the repository root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT / "data/processed/rousettus/video/not_augmented/original_bg"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/rousettus/video/not_augmented/green_bg"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "legacy/rousesttus_segmentation/training_results/runs/segment/"
    / "bat_face_seg/weights/best.pt"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
GREEN_BGR = np.array([0, 255, 0], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace rousettus original_bg image backgrounds with green using YOLO masks."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference size. Source/output image dimensions are preserved.",
    )
    parser.add_argument(
        "--selection",
        choices=("largest", "confidence"),
        default="largest",
        help="How to choose one mask if YOLO returns multiple masks.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path. Defaults to <output-dir>/green_bg_report.json.",
    )
    return parser.parse_args()


def image_paths(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def to_numpy(data: Any) -> np.ndarray | None:
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return data
    cpu = getattr(data, "cpu", None)
    if cpu is not None:
        data = cpu()
    numpy_fn = getattr(data, "numpy", None)
    if numpy_fn is not None:
        return numpy_fn()
    try:
        return np.asarray(data)
    except Exception:
        return None


def choose_mask(result: Any, image_shape: tuple[int, int], selection: str) -> tuple[np.ndarray, int] | None:
    masks = getattr(result, "masks", None)
    if masks is None or len(masks) == 0:
        return None

    masks_np = to_numpy(masks.data)
    if masks_np is None or masks_np.size == 0:
        return None
    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis, ...]

    boxes = getattr(result, "boxes", None)
    confs_np = to_numpy(getattr(boxes, "conf", None)) if boxes is not None else None
    if selection == "confidence" and confs_np is not None and len(confs_np) == len(masks_np):
        idx = int(np.argmax(confs_np))
    else:
        areas = [int(np.sum(mask > 0.5)) for mask in masks_np]
        if not areas or max(areas) == 0:
            return None
        idx = int(np.argmax(areas))

    mask = (masks_np[idx] > 0.5).astype(np.uint8)
    height, width = image_shape
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

    if int(mask.sum()) == 0:
        return None
    return mask, idx


def replace_with_green(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.empty_like(image)
    output[:, :] = GREEN_BGR
    foreground = mask.astype(bool)
    output[foreground] = image[foreground]
    return output


def write_image(path: Path, image: np.ndarray) -> bool:
    params: list[int] = []
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif suffix == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    return bool(cv2.imwrite(str(path), image, params))


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    report_path = (args.report or (output_dir / "green_bg_report.json")).resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images = image_paths(source_dir)
    selected_images = images[: args.limit] if args.limit is not None else images

    model = YOLO(str(checkpoint))
    report: dict[str, Any] = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "confidence_threshold": args.conf,
        "imgsz": args.imgsz,
        "selection": args.selection,
        "limit": args.limit,
        "total_source_images": len(images),
        "attempted": len(selected_images),
        "processed": 0,
        "skipped_existing": 0,
        "no_mask": 0,
        "read_errors": 0,
        "write_errors": 0,
        "errors": [],
        "no_mask_files": [],
        "mask_area_fraction": [],
    }

    for image_path in selected_images:
        out_path = output_dir / image_path.name
        if out_path.exists() and not args.overwrite:
            report["skipped_existing"] += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            report["read_errors"] += 1
            report["errors"].append({"file": image_path.name, "error": "cv2.imread returned None"})
            continue

        try:
            results = model.predict(
                source=image,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
                save=False,
            )
            result = results[0] if results else None
            chosen = choose_mask(result, image.shape[:2], args.selection) if result is not None else None
        except Exception as exc:
            report["no_mask"] += 1
            report["errors"].append({"file": image_path.name, "error": repr(exc)})
            continue

        if chosen is None:
            report["no_mask"] += 1
            report["no_mask_files"].append(image_path.name)
            continue

        mask, _ = chosen
        replaced = replace_with_green(image, mask)
        if not write_image(out_path, replaced):
            report["write_errors"] += 1
            report["errors"].append({"file": image_path.name, "error": "cv2.imwrite returned False"})
            continue

        report["processed"] += 1
        report["mask_area_fraction"].append(float(mask.sum()) / float(mask.size))

    area_fractions = report.pop("mask_area_fraction")
    report["mask_area_fraction_summary"] = {
        "count": len(area_fractions),
        "min": min(area_fractions) if area_fractions else None,
        "max": max(area_fractions) if area_fractions else None,
        "mean": mean(area_fractions) if area_fractions else None,
    }
    report["errors"] = report["errors"][:50]
    report["no_mask_files"] = report["no_mask_files"][:200]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["read_errors"] == 0 and report["write_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
