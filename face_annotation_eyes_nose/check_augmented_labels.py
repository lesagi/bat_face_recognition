#!/usr/bin/env python3
"""
Diagnostic script to check if augmented YOLO pose labels match the images correctly.
This will help identify issues with the augmentation process.
"""

import os
import sys
import cv2
import numpy as np
import random
from pathlib import Path


def read_yolo_label(label_path, img_width, img_height):
    """Read YOLO format label and convert to absolute coordinates."""
    if not os.path.exists(label_path):
        return None

    annotations = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:  # class + bbox + keypoints
                class_id = int(parts[0])

                # Bounding box (normalized -> absolute)
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                # Convert to top-left, bottom-right
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                # Keypoints (if present)
                keypoints = []
                if len(parts) > 5:
                    # YOLO pose format: x1 y1 v1 x2 y2 v2 x3 y3 v3 (normalized)
                    kpt_data = parts[5:]
                    for i in range(0, len(kpt_data), 3):
                        if i + 2 < len(kpt_data):
                            x = float(kpt_data[i]) * img_width
                            y = float(kpt_data[i + 1]) * img_height
                            v = int(float(kpt_data[i + 2]))  # visibility
                            keypoints.append((x, y, v))

                annotations.append(
                    {
                        "class": class_id,
                        "bbox": (x1, y1, x2, y2),
                        "keypoints": keypoints,
                    }
                )

    return annotations


def visualize_sample(image_path, label_path, output_path):
    """Visualize an image with its labels overlaid."""

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return False

    img_height, img_width = image.shape[:2]

    # Read labels
    annotations = read_yolo_label(label_path, img_width, img_height)
    if annotations is None:
        print(f"❌ Could not load labels: {label_path}")
        return False

    # Draw annotations
    vis_image = image.copy()

    for ann in annotations:
        # Draw bounding box
        x1, y1, x2, y2 = ann["bbox"]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis_image,
            f"Class {ann['class']}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Draw keypoints
        keypoint_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ]  # BGR: Blue, Green, Red
        keypoint_labels = ["Left Eye", "Right Eye", "Nose"]

        for i, (x, y, v) in enumerate(ann["keypoints"]):
            if v > 0:  # Visible keypoint
                color = keypoint_colors[i % len(keypoint_colors)]
                cv2.circle(vis_image, (int(x), int(y)), 5, color, -1)
                cv2.circle(vis_image, (int(x), int(y)), 7, (255, 255, 255), 1)

                # Add label
                label = keypoint_labels[i % len(keypoint_labels)]
                cv2.putText(
                    vis_image,
                    label,
                    (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

    # Save visualization
    success = cv2.imwrite(output_path, vis_image)
    if success:
        print(f"✅ Visualization saved: {output_path}")
        return True
    else:
        print(f"❌ Failed to save: {output_path}")
        return False


def check_augmented_dataset():
    """Check augmented dataset for label accuracy."""

    # Paths
    base_dir = "/Users/MAC/Documents/bat_face_rec/face_annotation_eyes_nose"
    augmented_dir = os.path.join(base_dir, "augmentation", "augmented")
    output_dir = "/Users/MAC/Documents/bat_face_rec/augmented_labels_check"

    print("🔍 CHECKING AUGMENTED DATASET LABELS")
    print("=" * 60)
    print(f"📁 Augmented data: {augmented_dir}")
    print(f"📤 Output: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check both train and val
    for split in ["train", "val"]:
        print(f"\n🔍 Checking {split} split...")

        images_dir = os.path.join(augmented_dir, split, "images")
        labels_dir = os.path.join(augmented_dir, split, "labels")

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"❌ Missing directories for {split}")
            continue

        # Get all images
        image_files = [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(image_files) == 0:
            print(f"❌ No images found in {split}")
            continue

        print(f"📊 Found {len(image_files)} images in {split}")

        # Sample some images for checking
        sample_size = min(10, len(image_files))
        sampled_files = random.sample(image_files, sample_size)

        print(f"🎲 Sampling {sample_size} images for visualization...")

        successful = 0
        for i, img_file in enumerate(sampled_files):
            # Get corresponding label file
            label_file = os.path.splitext(img_file)[0] + ".txt"

            image_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, label_file)

            # Create output filename
            output_filename = f"{split}_sample_{i+1}_{img_file}"
            output_path = os.path.join(output_dir, output_filename)

            print(f"   📝 Processing: {img_file}")

            if visualize_sample(image_path, label_path, output_path):
                successful += 1
            else:
                print(f"   ❌ Failed to process {img_file}")

        print(
            f"✅ Successfully processed {successful}/{sample_size} samples from {split}"
        )

    print(f"\n🎉 CHECKING COMPLETE!")
    print(f"📁 Check visualizations in: {output_dir}")
    print(f"🔍 Look for:")
    print(f"   - Keypoints (eyes, nose) matching the actual face features")
    print(f"   - Bounding boxes correctly encompassing the face")
    print(f"   - Consistent positioning across different augmentations")


def check_original_vs_augmented():
    """Compare original images with their augmented versions."""

    print(f"\n🔍 COMPARING ORIGINAL VS AUGMENTED")
    print("=" * 60)

    base_dir = "/Users/MAC/Documents/bat_face_rec/face_annotation_eyes_nose"
    original_dir = os.path.join(base_dir, "yolo_dataset_keypoints")
    augmented_dir = os.path.join(base_dir, "augmentation", "augmented")
    output_dir = "/Users/MAC/Documents/bat_face_rec/original_vs_augmented_check"

    os.makedirs(output_dir, exist_ok=True)

    # Check train split
    orig_images_dir = os.path.join(original_dir, "train", "images")
    orig_labels_dir = os.path.join(original_dir, "train", "labels")
    aug_images_dir = os.path.join(augmented_dir, "train", "images")
    aug_labels_dir = os.path.join(augmented_dir, "train", "labels")

    if not all(
        os.path.exists(d)
        for d in [orig_images_dir, orig_labels_dir, aug_images_dir, aug_labels_dir]
    ):
        print("❌ Missing required directories")
        return

    # Get original files
    orig_files = [
        f
        for f in os.listdir(orig_images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(orig_files) == 0:
        print("❌ No original images found")
        return

    # Pick one original file and its augmentations
    sample_orig = random.choice(orig_files)
    base_name = os.path.splitext(sample_orig)[0]

    print(f"📸 Original file: {sample_orig}")

    # Find augmented versions
    aug_files = [
        f for f in os.listdir(aug_images_dir) if f.startswith(base_name + "_aug_")
    ]

    print(f"🔄 Found {len(aug_files)} augmented versions")

    # Visualize original
    orig_img_path = os.path.join(orig_images_dir, sample_orig)
    orig_label_path = os.path.join(orig_labels_dir, base_name + ".txt")
    orig_output = os.path.join(output_dir, f"original_{sample_orig}")

    print(f"📝 Processing original...")
    visualize_sample(orig_img_path, orig_label_path, orig_output)

    # Visualize some augmented versions
    sample_aug_count = min(5, len(aug_files))
    sampled_aug = random.sample(aug_files, sample_aug_count)

    for i, aug_file in enumerate(sampled_aug):
        aug_img_path = os.path.join(aug_images_dir, aug_file)
        aug_label_path = os.path.join(
            aug_labels_dir, os.path.splitext(aug_file)[0] + ".txt"
        )
        aug_output = os.path.join(output_dir, f"augmented_{i+1}_{aug_file}")

        print(f"📝 Processing augmented {i+1}: {aug_file}")
        visualize_sample(aug_img_path, aug_label_path, aug_output)

    print(f"✅ Comparison complete! Check {output_dir}")


if __name__ == "__main__":
    # Set random seed for reproducible sampling
    random.seed(42)

    print("🔬 AUGMENTED LABELS DIAGNOSTIC")
    print("=" * 60)

    # Check augmented dataset
    check_augmented_dataset()

    # Compare original vs augmented
    check_original_vs_augmented()

    print(f"\n💡 NEXT STEPS:")
    print(f"1. Review the generated visualizations")
    print(f"2. Check if keypoints align with facial features")
    print(f"3. Verify bounding boxes encompass the face correctly")
    print(f"4. Compare original vs augmented to spot transformation issues")
