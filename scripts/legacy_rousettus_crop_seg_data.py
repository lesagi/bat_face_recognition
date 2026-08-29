#!/usr/bin/env python3
"""Build a temporary crop-matched YOLO segmentation dataset.

This fallback converts legacy full-frame rousettus segmentation labels into
224x224 face-centered crops so a temporary YOLO segmenter can be fine-tuned for
the already-cropped original_bg recognition images.

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
import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEGACY_DATA = PROJECT_ROOT / "legacy/rousesttus_segmentation/data"
DEFAULT_OUTPUT = PROJECT_ROOT / "tmp_rousettus_crop_seg_data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop legacy rousettus full-frame YOLO masks into a 224x224 YOLO dataset."
    )
    parser.add_argument("--legacy-data", type=Path, default=DEFAULT_LEGACY_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.35,
        help="Extra crop margin as a fraction of the larger mask bbox dimension.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_label(label_path: Path, width: int, height: int) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 7:
            continue
        values = [float(x) for x in parts[1:]]
        points = np.array(values, dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        polygons.append(np.round(points).astype(np.int32))
    return polygons


def make_mask(polygons: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon], 255)
    return mask


def crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1
    side = int(round(max(bbox_w, bbox_h) * (1.0 + 2.0 * margin)))
    side = max(side, bbox_w, bbox_h, 1)

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    crop_x1 = int(round(cx - side / 2.0))
    crop_y1 = int(round(cy - side / 2.0))
    crop_x2 = crop_x1 + side
    crop_y2 = crop_y1 + side

    height, width = image.shape[:2]
    src_x1 = max(crop_x1, 0)
    src_y1 = max(crop_y1, 0)
    src_x2 = min(crop_x2, width)
    src_y2 = min(crop_y2, height)
    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return None

    dst_x1 = src_x1 - crop_x1
    dst_y1 = src_y1 - crop_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    crop_img = np.zeros((side, side, 3), dtype=image.dtype)
    crop_mask = np.zeros((side, side), dtype=np.uint8)
    crop_img[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    crop_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]

    resized_img = cv2.resize(crop_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(
        crop_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST
    )
    resized_mask = (resized_mask > 0).astype(np.uint8) * 255
    return resized_img, resized_mask


def mask_to_yolo_polygon(mask: np.ndarray) -> str | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 8:
        return None

    epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        return None

    height, width = mask.shape[:2]
    coords: list[str] = []
    for x, y in approx:
        coords.append(f"{min(max(float(x) / width, 0.0), 1.0):.6f}")
        coords.append(f"{min(max(float(y) / height, 0.0), 1.0):.6f}")
    return "0 " + " ".join(coords)


def process_split(
    split: str,
    legacy_data: Path,
    output_dir: Path,
    target_size: int,
    margin: float,
) -> tuple[int, int]:
    image_dir = legacy_data / split / "images"
    label_dir = legacy_data / split / "labels"
    out_image_dir = output_dir / split / "images"
    out_label_dir = output_dir / split / "labels"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for image_path in sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    ):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            skipped += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            continue

        height, width = image.shape[:2]
        polygons = read_label(label_path, width, height)
        mask = make_mask(polygons, (height, width))
        cropped = crop_to_mask(image, mask, target_size, margin)
        if cropped is None:
            skipped += 1
            continue

        cropped_image, cropped_mask = cropped
        yolo_label = mask_to_yolo_polygon(cropped_mask)
        if yolo_label is None:
            skipped += 1
            continue

        out_image = out_image_dir / image_path.name
        out_label = out_label_dir / f"{image_path.stem}.txt"
        if not cv2.imwrite(str(out_image), cropped_image):
            skipped += 1
            continue
        out_label.write_text(yolo_label + "\n", encoding="utf-8")
        processed += 1

    return processed, skipped


def main() -> int:
    args = parse_args()
    legacy_data = args.legacy_data.resolve()
    output_dir = args.output_dir.resolve()

    if not legacy_data.is_dir():
        raise FileNotFoundError(f"legacy data directory does not exist: {legacy_data}")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ("train", "val"):
        processed, skipped = process_split(
            split=split,
            legacy_data=legacy_data,
            output_dir=output_dir,
            target_size=args.target_size,
            margin=args.margin,
        )
        summary[split] = {"processed": processed, "skipped": skipped}

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
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

    print(f"wrote dataset: {output_dir}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
