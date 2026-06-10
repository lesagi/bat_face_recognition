#!/usr/bin/env python3
"""Generate crop-level masks from legacy rousettus segmentation labels.

This is a temporary inspection/data-prep script. It does not train anything.
It reads legacy full-frame images + YOLO segmentation polygons, crops a square
around each labeled mask, resizes image and mask to the target size, and writes
an inspectable directory that can later be used for YOLO training.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LEGACY_DATA = PROJECT_ROOT / "legacy/rousesttus_segmentation/data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp_rousettus_crop_masks"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class CropResult(NamedTuple):
    image: np.ndarray
    mask: np.ndarray
    crop_box: tuple[int, int, int, int]
    source_size: tuple[int, int]
    mask_area_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create square cropped images, binary masks, overlays, and YOLO labels "
            "from the legacy rousettus full-frame segmentation dataset."
        )
    )
    parser.add_argument("--legacy-data", type=Path, default=DEFAULT_LEGACY_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.75,
        help=(
            "Extra square-crop margin as a fraction of the larger mask bbox dimension. "
            "Larger values keep more context around the labeled face."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_yolo_polygons(label_path: Path, width: int, height: int) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = np.array([float(value) for value in parts[1:]], dtype=np.float32)
        if coords.size % 2 != 0:
            continue
        points = coords.reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        polygons.append(np.round(points).astype(np.int32))
    return polygons


def polygons_to_mask(polygons: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon], 255)
    return mask


def square_crop_around_mask(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: int,
    margin: float,
) -> CropResult | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1
    side = int(round(max(bbox_w, bbox_h) * (1.0 + 2.0 * margin)))
    side = max(side, bbox_w, bbox_h, 1)

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    crop_x1 = int(round(center_x - side / 2.0))
    crop_y1 = int(round(center_y - side / 2.0))
    crop_x2 = crop_x1 + side
    crop_y2 = crop_y1 + side

    image_h, image_w = image.shape[:2]
    src_x1 = max(crop_x1, 0)
    src_y1 = max(crop_y1, 0)
    src_x2 = min(crop_x2, image_w)
    src_y2 = min(crop_y2, image_h)
    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return None

    dst_x1 = src_x1 - crop_x1
    dst_y1 = src_y1 - crop_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    square_image = np.zeros((side, side, 3), dtype=image.dtype)
    square_mask = np.zeros((side, side), dtype=np.uint8)
    square_image[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    square_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]

    resized_image = cv2.resize(
        square_image,
        (target_size, target_size),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_mask = cv2.resize(
        square_mask,
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST,
    )
    resized_mask = (resized_mask > 0).astype(np.uint8) * 255
    area_fraction = float(np.count_nonzero(resized_mask)) / float(resized_mask.size)

    return CropResult(
        image=resized_image,
        mask=resized_mask,
        crop_box=(crop_x1, crop_y1, crop_x2, crop_y2),
        source_size=(image_w, image_h),
        mask_area_fraction=area_fraction,
    )


def mask_to_yolo_label(mask: np.ndarray) -> str | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 8:
        return None

    epsilon = max(1.0, 0.0015 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        return None

    height, width = mask.shape[:2]
    coords: list[str] = []
    for x, y in approx:
        coords.append(f"{min(max(float(x) / width, 0.0), 1.0):.6f}")
        coords.append(f"{min(max(float(y) / height, 0.0), 1.0):.6f}")
    return "0 " + " ".join(coords)


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    mask_bool = mask > 0
    red = np.zeros_like(image)
    red[:, :] = (0, 0, 255)
    overlay[mask_bool] = cv2.addWeighted(image, 0.55, red, 0.45, 0)[mask_bool]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
    return overlay


def ensure_dirs(output_dir: Path, split: str) -> dict[str, Path]:
    dirs = {
        "images": output_dir / split / "images",
        "masks": output_dir / split / "masks",
        "overlays": output_dir / split / "overlays",
        "labels": output_dir / split / "labels",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def process_split(
    split: str,
    legacy_data: Path,
    output_dir: Path,
    target_size: int,
    margin: float,
) -> dict[str, object]:
    image_dir = legacy_data / split / "images"
    label_dir = legacy_data / split / "labels"
    dirs = ensure_dirs(output_dir, split)

    processed = 0
    skipped: list[dict[str, str]] = []
    records: list[dict[str, object]] = []

    for image_path in sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            skipped.append({"file": image_path.name, "reason": "missing label"})
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped.append({"file": image_path.name, "reason": "could not read image"})
            continue

        height, width = image.shape[:2]
        polygons = read_yolo_polygons(label_path, width, height)
        full_mask = polygons_to_mask(polygons, (height, width))
        crop = square_crop_around_mask(
            image=image,
            mask=full_mask,
            target_size=target_size,
            margin=margin,
        )
        if crop is None:
            skipped.append({"file": image_path.name, "reason": "empty crop/mask"})
            continue

        yolo_label = mask_to_yolo_label(crop.mask)
        if yolo_label is None:
            skipped.append({"file": image_path.name, "reason": "invalid cropped contour"})
            continue

        output_name = image_path.name
        mask_name = f"{image_path.stem}.png"
        label_name = f"{image_path.stem}.txt"
        cv2.imwrite(str(dirs["images"] / output_name), crop.image)
        cv2.imwrite(str(dirs["masks"] / mask_name), crop.mask)
        cv2.imwrite(str(dirs["overlays"] / output_name), make_overlay(crop.image, crop.mask))
        (dirs["labels"] / label_name).write_text(yolo_label + "\n", encoding="utf-8")

        records.append(
            {
                "source_image": str(image_path),
                "image": str(dirs["images"] / output_name),
                "mask": str(dirs["masks"] / mask_name),
                "overlay": str(dirs["overlays"] / output_name),
                "label": str(dirs["labels"] / label_name),
                "crop_box_xyxy": crop.crop_box,
                "source_size_wh": crop.source_size,
                "target_size": target_size,
                "mask_area_fraction": crop.mask_area_fraction,
            }
        )
        processed += 1

    return {
        "processed": processed,
        "skipped": skipped,
        "records": records,
    }


def write_data_yaml(output_dir: Path) -> None:
    (output_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output_dir}",
                "train: train/images",
                "val: val/images",
                "names:",
                "  0: bat_face",
                "nc: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    legacy_data = args.legacy_data.resolve()
    output_dir = args.output_dir.resolve()

    if not legacy_data.is_dir():
        raise FileNotFoundError(f"legacy data directory does not exist: {legacy_data}")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "legacy_data": str(legacy_data),
        "output_dir": str(output_dir),
        "target_size": args.target_size,
        "margin": args.margin,
        "splits": {},
    }

    all_records: list[dict[str, object]] = []
    for split in ("train", "val"):
        split_report = process_split(
            split=split,
            legacy_data=legacy_data,
            output_dir=output_dir,
            target_size=args.target_size,
            margin=args.margin,
        )
        all_records.extend(split_report.pop("records"))  # type: ignore[arg-type]
        report["splits"][split] = split_report  # type: ignore[index]

    write_data_yaml(output_dir)
    report["total_processed"] = len(all_records)
    report["records"] = all_records
    (output_dir / "crop_mask_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    print(f"wrote crop-mask dataset: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
