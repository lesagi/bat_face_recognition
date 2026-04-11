#!/usr/bin/env python3
"""
YOLO Pose Dataset Augmentation Script with Results Overview

This script augments the bat face keypoint dataset using albumentations and
provides comprehensive results checking and visualization.
Creates 60 augmented versions of each image with:
- Rotations
- Color modifications  
- Noise
- Properly transforms keypoints and bounding boxes
- Generates results overview with visualizations

Usage:
    python create_augmented_dataset.py
"""

import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml


class YoloPoseAugmenter:
    """Augments YOLO pose dataset with proper keypoint transformations."""

    def __init__(self, source_dir, target_dir, augmentations_per_image=60):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.augmentations_per_image = augmentations_per_image

        # Create target directory structure
        self.setup_directories()

        # Define augmentation transforms
        self.transform = A.Compose(
            [
                # Geometric transformations
                A.OneOf(
                    [
                        A.Rotate(limit=30, p=0.7),
                        A.Affine(
                            scale=(0.8, 1.2),
                            translate_percent=(-0.1, 0.1),
                            rotate=(-15, 15),
                            shear=(-10, 10),
                            p=0.5,
                        ),
                        A.Perspective(scale=(0.05, 0.1), p=0.3),
                    ],
                    p=0.8,
                ),
                # Color transformations
                A.OneOf(
                    [
                        A.ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.3, contrast_limit=0.3, p=0.7
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=0.6,
                        ),
                        A.RGBShift(
                            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5
                        ),
                    ],
                    p=0.9,
                ),
                # Noise and blur
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10, 50), p=0.6),
                        A.ISONoise(
                            color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4
                        ),
                        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.4),
                    ],
                    p=0.5,
                ),
                # Occasional blur/sharpening
                A.OneOf(
                    [
                        A.Blur(blur_limit=3, p=0.3),
                        A.MotionBlur(blur_limit=3, p=0.3),
                        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
                    ],
                    p=0.3,
                ),
                # Weather effects (light)
                A.OneOf(
                    [
                        A.RandomShadow(
                            shadow_roi=(0, 0, 1, 1),
                            num_shadows_lower=1,
                            num_shadows_upper=2,
                            p=0.2,
                        ),
                        A.RandomSunFlare(
                            flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, p=0.1
                        ),
                    ],
                    p=0.2,
                ),
            ],
            keypoint_params=A.KeypointParams(
                format="xy", remove_invisible=False, label_fields=["keypoint_labels"]
            ),
            bbox_params=A.BboxParams(format="yolo", label_fields=["bbox_labels"]),
        )

    def setup_directories(self):
        """Create the target directory structure."""
        dirs_to_create = [
            self.target_dir / "augmented" / "train" / "images",
            self.target_dir / "augmented" / "train" / "labels",
            self.target_dir / "augmented" / "val" / "images",
            self.target_dir / "augmented" / "val" / "labels",
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")

    def parse_yolo_label(self, label_path, img_width, img_height):
        """Parse YOLO pose format label file."""
        bboxes = []
        keypoints = []  # Flat list of (x, y) keypoints
        bbox_labels = []
        keypoint_labels = []  # Flat list of keypoint labels
        keypoint_detection_ids = []  # Track which detection each keypoint belongs to

        if not os.path.exists(label_path):
            return bboxes, keypoints, bbox_labels, keypoint_labels, keypoint_detection_ids

        with open(label_path, "r") as f:
            lines = f.read().strip().split("\n")
            for detection_id, line in enumerate(lines):
                if not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) < 14:  # class + bbox(4) + keypoints(3*3) = 14
                    continue

                class_id = int(parts[0])

                # Parse bounding box (normalized YOLO format)
                x_center, y_center, width, height = map(float, parts[1:5])
                bboxes.append([x_center, y_center, width, height])
                bbox_labels.append(class_id)

                # Parse keypoints (3 keypoints: left_eye, right_eye, nose)
                # Add each keypoint as a separate (x, y) tuple to the flat list
                for i in range(3):
                    x = float(parts[5 + i * 3]) * img_width  # Convert to absolute coordinates
                    y = float(parts[6 + i * 3]) * img_height
                    visibility = int(parts[7 + i * 3])
                    
                    # Only add visible keypoints
                    if visibility > 0:
                        keypoints.append((x, y))
                        keypoint_labels.append(i)  # 0=left_eye, 1=right_eye, 2=nose
                        keypoint_detection_ids.append(detection_id)

        return bboxes, keypoints, bbox_labels, keypoint_labels, keypoint_detection_ids

    def save_yolo_label(
        self,
        bboxes,
        keypoints,
        bbox_labels,
        keypoint_labels,
        keypoint_detection_ids,
        label_path,
        img_width,
        img_height,
    ):
        """Save augmented annotations in YOLO pose format."""
        # Reconstruct keypoints by detection
        detection_keypoints = {}
        for kpt, kpt_label, detection_id in zip(keypoints, keypoint_labels, keypoint_detection_ids):
            if detection_id not in detection_keypoints:
                detection_keypoints[detection_id] = [None, None, None]  # [left_eye, right_eye, nose]
            detection_keypoints[detection_id][kpt_label] = kpt

        with open(label_path, "w") as f:
            for detection_id, (bbox, bbox_label) in enumerate(zip(bboxes, bbox_labels)):
                # Write bbox in YOLO format (normalized)
                x_center, y_center, width, height = bbox
                line_parts = [
                    str(bbox_label),
                    str(x_center),
                    str(y_center),
                    str(width),
                    str(height),
                ]

                # Write keypoints in YOLO pose format: x y v x y v x y v
                kpts = detection_keypoints.get(detection_id, [None, None, None])
                for kpt in kpts:
                    if kpt is not None:
                        x_norm = kpt[0] / img_width  # Normalize to [0,1]
                        y_norm = kpt[1] / img_height
                        visibility = 2  # Visible after augmentation
                    else:
                        # If keypoint was not transformed (invisible or missing), set to 0
                        x_norm = 0.0
                        y_norm = 0.0
                        visibility = 0  # Not visible
                    
                    line_parts.extend([str(x_norm), str(y_norm), str(visibility)])

                f.write(" ".join(line_parts) + "\n")

    def augment_single_image(self, img_path, label_path, split):
        """Create augmented versions of a single image."""
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"❌ Could not load image: {img_path}")
            return 0

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]

        # Parse labels
        bboxes, keypoints, bbox_labels, keypoint_labels, keypoint_detection_ids = self.parse_yolo_label(
            label_path, img_width, img_height
        )

        if not bboxes:
            print(f"⚠️ No labels found for {img_path}")
            return 0

        successful_augmentations = 0
        base_name = img_path.stem

        for aug_idx in range(self.augmentations_per_image):
            try:
                # Apply augmentation
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    keypoints=keypoints,
                    bbox_labels=bbox_labels,
                    keypoint_labels=keypoint_labels,
                )

                # Check if transformation was successful
                if not transformed["bboxes"] or not transformed["keypoints"]:
                    continue

                # Save augmented image
                aug_img_name = f"{base_name}_aug_{aug_idx:03d}.jpg"
                aug_img_path = (
                    self.target_dir / "augmented" / split / "images" / aug_img_name
                )

                aug_image_bgr = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(aug_img_path), aug_image_bgr)

                # Save augmented labels
                aug_label_name = f"{base_name}_aug_{aug_idx:03d}.txt"
                aug_label_path = (
                    self.target_dir / "augmented" / split / "labels" / aug_label_name
                )

                # Use transformed image dimensions for proper normalization
                transformed_img_height, transformed_img_width = transformed["image"].shape[:2]
                
                self.save_yolo_label(
                    transformed["bboxes"],
                    transformed["keypoints"],
                    transformed["bbox_labels"],
                    transformed["keypoint_labels"],
                    keypoint_detection_ids,  # Use original detection IDs
                    aug_label_path,
                    transformed_img_width,
                    transformed_img_height,
                )

                successful_augmentations += 1

            except Exception as e:
                if aug_idx < 5:  # Only print errors for first few attempts
                    print(f"⚠️ Augmentation {aug_idx} failed for {base_name}: {e}")
                continue

        return successful_augmentations

    def augment_dataset(self):
        """Augment the entire dataset."""
        print("🔄 Starting dataset augmentation...")
        print(f"📊 Target: {self.augmentations_per_image} augmentations per image")
        print()

        total_successful = 0
        total_attempted = 0

        for split in ["train", "val"]:
            print(f"📁 Processing {split} split...")

            images_dir = self.source_dir / split / "images"
            labels_dir = self.source_dir / split / "labels"

            if not images_dir.exists():
                print(f"❌ Images directory not found: {images_dir}")
                continue

            image_files = list(images_dir.glob("*.jpg")) + list(
                images_dir.glob("*.png")
            )
            print(f"📷 Found {len(image_files)} images in {split}")

            split_successful = 0

            for img_path in tqdm(image_files, desc=f"Augmenting {split}"):
                label_path = labels_dir / f"{img_path.stem}.txt"

                successful = self.augment_single_image(img_path, label_path, split)
                split_successful += successful
                total_attempted += self.augmentations_per_image

            total_successful += split_successful
            print(f"✅ {split}: {split_successful} successful augmentations")
            print()

        print(f"🎯 Total Summary:")
        print(f"   Attempted: {total_attempted} augmentations")
        print(f"   Successful: {total_successful} augmentations")
        print(f"   Success rate: {total_successful/total_attempted*100:.1f}%")

        # Create updated dataset.yaml
        self.create_dataset_yaml()

    def create_dataset_yaml(self):
        """Create dataset.yaml for the augmented dataset."""
        dataset_config = {
            "path": str(self.target_dir / "augmented"),
            "train": "train/images",
            "val": "val/images",
            "nc": 1,
            "names": ["face"],
            "kpt_shape": [3, 3],
            "flip_idx": [1, 0, 2],
        }

        yaml_path = self.target_dir / "augmented" / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"📝 Created dataset.yaml: {yaml_path}")


class ResultsChecker:
    """Check and visualize augmented dataset results."""
    
    def __init__(self, source_dir, target_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def read_yolo_label(self, label_path, img_width, img_height):
        """Read YOLO format label and convert to absolute coordinates."""
        if not os.path.exists(label_path):
            return None

        annotations = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 14:  # class + bbox(4) + keypoints(3*3) = 14
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

                    # Keypoints (YOLO pose format: x1 y1 v1 x2 y2 v2 x3 y3 v3)
                    keypoints = []
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

    def visualize_sample(self, image_path, label_path, output_path):
        """Visualize an image with its labels overlaid."""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return False

        img_height, img_width = image.shape[:2]

        # Read labels
        annotations = self.read_yolo_label(label_path, img_width, img_height)
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
                (255, 0, 0),    # Blue for left eye
                (0, 255, 0),    # Green for right eye  
                (0, 0, 255),    # Red for nose
            ]  # BGR format
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
        success = cv2.imwrite(str(output_path), vis_image)
        if success:
            print(f"✅ Visualization saved: {output_path}")
            return True
        else:
            print(f"❌ Failed to save: {output_path}")
            return False

    def check_augmented_dataset(self):
        """Check augmented dataset for label accuracy."""
        print("\n🔍 CHECKING AUGMENTED DATASET LABELS")
        print("=" * 60)
        print(f"📁 Augmented data: {self.target_dir / 'augmented'}")
        print(f"📤 Output: {self.output_dir}")

        total_successful = 0
        total_samples = 0

        # Check both train and val
        for split in ["train", "val"]:
            print(f"\n🔍 Checking {split} split...")

            images_dir = self.target_dir / "augmented" / split / "images"
            labels_dir = self.target_dir / "augmented" / split / "labels"

            if not images_dir.exists() or not labels_dir.exists():
                print(f"❌ Missing directories for {split}")
                continue

            # Get all images
            image_files = [
                f for f in images_dir.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
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
                label_file = img_file.with_suffix('.txt')
                label_path = labels_dir / label_file.name

                # Create output filename
                output_filename = f"{split}_sample_{i+1}_{img_file.name}"
                output_path = self.output_dir / output_filename

                print(f"   📝 Processing: {img_file.name}")

                if self.visualize_sample(img_file, label_path, output_path):
                    successful += 1
                else:
                    print(f"   ❌ Failed to process {img_file.name}")

            print(f"✅ Successfully processed {successful}/{sample_size} samples from {split}")
            total_successful += successful
            total_samples += sample_size

        return total_successful, total_samples

    def check_original_vs_augmented(self):
        """Compare original images with their augmented versions."""
        print(f"\n🔍 COMPARING ORIGINAL VS AUGMENTED")
        print("=" * 60)

        # Check train split
        orig_images_dir = self.source_dir / "train" / "images"
        orig_labels_dir = self.source_dir / "train" / "labels"
        aug_images_dir = self.target_dir / "augmented" / "train" / "images"
        aug_labels_dir = self.target_dir / "augmented" / "train" / "labels"

        required_dirs = [orig_images_dir, orig_labels_dir, aug_images_dir, aug_labels_dir]
        if not all(d.exists() for d in required_dirs):
            print("❌ Missing required directories")
            return False

        # Get original files
        orig_files = [
            f for f in orig_images_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        if len(orig_files) == 0:
            print("❌ No original images found")
            return False

        # Pick one original file and its augmentations
        sample_orig = random.choice(orig_files)
        base_name = sample_orig.stem

        print(f"📸 Original file: {sample_orig.name}")

        # Find augmented versions
        aug_files = [
            f for f in aug_images_dir.iterdir() 
            if f.name.startswith(base_name + "_aug_")
        ]

        print(f"🔄 Found {len(aug_files)} augmented versions")

        # Visualize original
        orig_label_path = orig_labels_dir / f"{base_name}.txt"
        orig_output = self.output_dir / f"original_{sample_orig.name}"

        print(f"📝 Processing original...")
        self.visualize_sample(sample_orig, orig_label_path, orig_output)

        # Visualize some augmented versions
        sample_aug_count = min(5, len(aug_files))
        if aug_files:
            sampled_aug = random.sample(aug_files, sample_aug_count)

            for i, aug_file in enumerate(sampled_aug):
                aug_label_path = aug_labels_dir / f"{aug_file.stem}.txt"
                aug_output = self.output_dir / f"augmented_{i+1}_{aug_file.name}"

                print(f"📝 Processing augmented {i+1}: {aug_file.name}")
                self.visualize_sample(aug_file, aug_label_path, aug_output)

        print(f"✅ Comparison complete! Check {self.output_dir}")
        return True


def main():
    """Main function to run the augmentation and results checking."""
    
    # Configuration - Update these paths as needed
    BASE_DIR = Path(__file__).parent.parent  # bat_face_rec directory
    SOURCE_DIR = BASE_DIR / "face_annotation_eyes_nose" / "yolo_dataset_keypoints"
    TARGET_DIR = BASE_DIR / "face_annotation_eyes_nose" / "augmentation"
    RESULTS_DIR = BASE_DIR / "augmented_labels_check"
    AUGMENTATIONS_PER_IMAGE = 60

    print("🦇 YOLO Pose Dataset Augmentation with Results Overview")
    print("=" * 70)
    print(f"📂 Source: {SOURCE_DIR}")
    print(f"📂 Target: {TARGET_DIR}")
    print(f"📤 Results: {RESULTS_DIR}")
    print(f"🔢 Augmentations per image: {AUGMENTATIONS_PER_IMAGE}")
    print()

    # Check dependencies
    try:
        import albumentations
        print(f"✅ Albumentations version: {albumentations.__version__}")
    except ImportError:
        print("❌ Albumentations not installed. Install with: pip install albumentations")
        return

    # Verify source directory exists
    if not SOURCE_DIR.exists():
        print(f"❌ Source directory not found: {SOURCE_DIR}")
        print("Please update SOURCE_DIR in the script to match your setup.")
        return

    # Step 1: Create augmenter and run augmentation
    print("\n🔄 STEP 1: RUNNING AUGMENTATION")
    print("=" * 50)
    augmenter = YoloPoseAugmenter(SOURCE_DIR, TARGET_DIR, AUGMENTATIONS_PER_IMAGE)
    augmenter.augment_dataset()
    
    # Step 2: Check results and create visualizations
    print("\n🔍 STEP 2: CHECKING RESULTS & CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Set random seed for reproducible sampling
    random.seed(42)
    
    checker = ResultsChecker(SOURCE_DIR, TARGET_DIR, RESULTS_DIR)
    
    # Check augmented dataset
    successful_samples, total_samples = checker.check_augmented_dataset()
    
    # Compare original vs augmented
    comparison_success = checker.check_original_vs_augmented()

    # Final summary
    print("\n🎉 AUGMENTATION AND CHECKING COMPLETE!")
    print("=" * 70)
    print(f"📈 Augmentation Results:")
    print(f"   • Augmented dataset created in: {TARGET_DIR}/augmented/")
    print(f"   • Dataset config: {TARGET_DIR}/augmented/dataset.yaml")
    print(f"\n📊 Results Overview:")
    print(f"   • Successfully visualized: {successful_samples}/{total_samples} samples")
    print(f"   • Visualizations saved to: {RESULTS_DIR}")
    print(f"   • Original vs augmented comparison: {'✅ Success' if comparison_success else '❌ Failed'}")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"1. 🔍 Review visualizations in: {RESULTS_DIR}")
    print(f"2. ✅ Check if keypoints align with facial features")
    print(f"3. ✅ Verify bounding boxes encompass faces correctly")
    print(f"4. 🚀 Train new model with:")
    print(f"   yolo pose train data={TARGET_DIR}/augmented/dataset.yaml model=yolov8n-pose.pt epochs=100")
    print(f"\n📝 What to look for in results:")
    print(f"   • Blue circles = Left Eye keypoints")
    print(f"   • Green circles = Right Eye keypoints") 
    print(f"   • Red circles = Nose keypoints")
    print(f"   • Green rectangles = Face bounding boxes")


if __name__ == "__main__":
    main()
