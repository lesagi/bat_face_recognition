#!/usr/bin/env python3
"""
YOLO Segmentation Dataset Augmentation

This module augments YOLO segmentation datasets using albumentations.
Properly transforms bounding boxes and segmentation masks.
"""

import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import List, Tuple, Dict, Any
import sys

# Add parent directory to path for config imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config.loader import load_config
    from config.yolo_segmentation import YOLOSegmentationConfig
except ImportError:
    print("⚠️ Config module not found, using default settings")
    load_config = None
    YOLOSegmentationConfig = None


class YoloSegmentationAugmenter:
    """Augments YOLO segmentation dataset with proper mask and bbox transformations."""

    def __init__(self, source_dir: str, target_dir: str, augmentations_per_image: int = 10, debug: bool = False):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.augmentations_per_image = augmentations_per_image
        self.debug = debug
        
        # Load config if available
        self.config = self._load_config()
        
        # Create target directory structure
        self.setup_directories()
        
        # Define augmentation transforms
        self.transform = self._create_transforms()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from app/config if available."""
        if load_config and YOLOSegmentationConfig:
            try:
                config = load_config()
                yolo_config = YOLOSegmentationConfig(config.yolo_segmentation)
                aug_config = yolo_config.augmentation
                
                return {
                    'default_epochs': yolo_config.training.get('default_epochs', 80),
                    'default_batch_size': yolo_config.training.get('default_batch_size', 16),
                    'default_image_size': yolo_config.training.get('default_image_size', 640),
                    'supported_extensions': yolo_config.data.get('supported_image_extensions', ['.png', '.jpg', '.jpeg']),
                    # Augmentation config
                    'geometric': aug_config.get('geometric_transforms', {}),
                    'color': aug_config.get('color_transforms', {}),
                    'noise': aug_config.get('noise_transforms', {}),
                    'blur': aug_config.get('blur_transforms', {}),
                    'scale_crop': aug_config.get('scale_crop_transforms', {}),
                }
            except Exception as e:
                print(f"⚠️ Could not load config: {e}")
        
        # Default config
        return {
            'default_epochs': 80,
            'default_batch_size': 16,
            'default_image_size': 640,
            'supported_extensions': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'],
            'geometric': {},
            'color': {},
            'noise': {},
            'blur': {},
            'scale_crop': {},
        }

    def _create_transforms(self) -> A.Compose:
        """Create albumentations transform pipeline for segmentation."""
        return A.Compose([
            # Geometric transformations
            A.OneOf([
                A.Rotate(limit=25, p=0.7),
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-10, 10),
                    shear=(-8, 8),
                    p=0.5,
                ),
                A.Perspective(scale=(0.02, 0.08), p=0.3),
            ], p=0.8),
            
            # Color transformations
            A.OneOf([
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.7
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=25,
                    val_shift_limit=15,
                    p=0.6,
                ),
                A.RGBShift(
                    r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5
                ),
            ], p=0.9),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(5, 30), p=0.6),
                A.ISONoise(
                    color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.4
                ),
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.4),
            ], p=0.4),
            
            # Light blur/sharpening
            A.OneOf([
                A.Blur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.2),
            ], p=0.2),
            
            # Weather effects (very light for segmentation)
            A.OneOf([
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=1,
                    p=0.1,
                ),
            ], p=0.1),
            
            # Scale/Crop variations to simulate different margins and zoom levels
            A.OneOf([
                # Simulate tighter crops (less margin)
                A.RandomResizedCrop(
                    height=self.config['default_image_size'], 
                    width=self.config['default_image_size'],
                    scale=tuple(self.config['scale_crop'].get('tight_crop_scale', [0.7, 0.9])),
                    ratio=tuple(self.config['scale_crop'].get('crop_ratio_range', [0.9, 1.1])),
                    p=0.3
                ),
                # Simulate looser crops (more margin) 
                A.RandomResizedCrop(
                    height=self.config['default_image_size'], 
                    width=self.config['default_image_size'],
                    scale=tuple(self.config['scale_crop'].get('loose_crop_scale', [0.4, 0.7])),
                    ratio=tuple(self.config['scale_crop'].get('crop_ratio_range', [0.9, 1.1])),
                    p=0.3
                ),
                # Different resize ratios to simulate various input sizes
                *self._create_resize_variations(),
            ], p=0.4),
        ],
        bbox_params=A.BboxParams(
            format="yolo", 
            label_fields=["bbox_labels"],
            min_visibility=0.3
        ),
        additional_targets={
            'mask': 'mask'
        })

    def _create_resize_variations(self) -> List[A.Compose]:
        """Create resize variation transforms based on config."""
        resize_variations = []
        sizes = self.config['scale_crop'].get('resize_variations', [480, 640, 800])
        target_size = self.config['default_image_size']
        
        for size in sizes:
            if size != target_size:  # Don't create identity transform
                resize_variations.append(
                    A.Compose([
                        A.Resize(height=size, width=size, p=1.0),
                        A.Resize(height=target_size, width=target_size, p=1.0),
                    ], p=0.2)
                )
        
        return resize_variations

    def setup_directories(self):
        """Create the target directory structure."""
        dirs_to_create = [
            self.target_dir / "train" / "images",
            self.target_dir / "train" / "labels", 
            self.target_dir / "val" / "images",
            self.target_dir / "val" / "labels",
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")

    def parse_yolo_segmentation_label(self, label_path: Path) -> List[Dict[str, Any]]:
        """Parse YOLO segmentation format label file."""
        annotations = []
        
        if not label_path.exists():
            return annotations

        with open(label_path, "r") as f:
            lines = f.read().strip().split("\n")
            for line in lines:
                if not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) < 5:  # At minimum: class + 4 bbox coords
                    continue

                class_id = int(parts[0])
                
                # Parse segmentation points (all remaining coordinates)
                coords = [float(x) for x in parts[1:]]
                
                # Convert segmentation polygon to bbox if needed
                if len(coords) >= 4:  # At least 2 points (x,y pairs)
                    # Reshape to pairs
                    points = np.array(coords).reshape(-1, 2)
                    
                    # Calculate bounding box from polygon
                    x_coords = points[:, 0]
                    y_coords = points[:, 1]
                    
                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                    y_min, y_max = np.min(y_coords), np.max(y_coords)
                    
                    # Convert to YOLO bbox format (center_x, center_y, width, height)
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    annotations.append({
                        'class_id': class_id,
                        'bbox': [center_x, center_y, width, height],
                        'segmentation': coords
                    })

        return annotations

    def create_mask_from_segmentation(self, segmentation: List[float], img_width: int, img_height: int) -> np.ndarray:
        """Create binary mask from segmentation coordinates."""
        # Convert normalized coordinates to absolute
        points = np.array(segmentation).reshape(-1, 2)
        points[:, 0] *= img_width
        points[:, 1] *= img_height
        points = points.astype(np.int32)
        
        # Create mask
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        return mask

    def mask_to_segmentation(self, mask: np.ndarray) -> List[float]:
        """Convert binary mask back to normalized segmentation coordinates."""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour to reduce points
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to normalized coordinates
        img_height, img_width = mask.shape
        segmentation = []
        
        for point in simplified:
            x, y = point[0]
            segmentation.extend([x / img_width, y / img_height])
        
        return segmentation

    def save_yolo_segmentation_label(self, annotations: List[Dict[str, Any]], label_path: Path):
        """Save augmented annotations in YOLO segmentation format."""
        with open(label_path, "w") as f:
            for ann in annotations:
                line_parts = [str(ann['class_id'])]
                line_parts.extend([str(coord) for coord in ann['segmentation']])
                f.write(" ".join(line_parts) + "\n")



    def augment_single_image(self, img_path: Path, label_path: Path, split: str) -> int:
        """Create augmented versions of a single image."""
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"❌ Could not load image: {img_path}")
            return 0

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]

        # Parse labels
        annotations = self.parse_yolo_segmentation_label(label_path)
        
        if not annotations:
            print(f"⚠️ No labels found for {img_path}")
            return 0

        successful_augmentations = 0
        base_name = img_path.stem
        original_extension = img_path.suffix

        for aug_idx in range(self.augmentations_per_image):
            try:
                # Create combined mask for all objects
                combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                bboxes = []
                bbox_labels = []
                
                for ann in annotations:
                    # Create mask for this annotation
                    mask = self.create_mask_from_segmentation(ann['segmentation'], img_width, img_height)
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                    
                    # Add bbox
                    bboxes.append(ann['bbox'])
                    bbox_labels.append(ann['class_id'])

                # Apply augmentation
                transformed = self.transform(
                    image=image,
                    mask=combined_mask,
                    bboxes=bboxes,
                    bbox_labels=bbox_labels,
                )

                # Extract augmented image (RGB → BGR for OpenCV saving)
                aug_image = transformed["image"]
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)

                # Check if transformation was successful
                if not transformed["bboxes"]:
                    continue


                
                # Debug: Create visualization with all predictions if debug mode is enabled
                if self.debug:
                    try:
                        # Print transformation details for verification
                        if aug_idx == 0:  # Only print once per image to avoid spam
                            print(f"🔍 Debug mode enabled for {base_name}")
                            print(f"   Original image size: {image.shape[1]}x{image.shape[0]}")
                            print(f"   Transformed image size: {transformed['image'].shape[1]}x{transformed['image'].shape[0]}")
                            print(f"   Bounding boxes: {len(transformed['bboxes'])}")
                            print(f"   Mask shape: {transformed['mask'].shape if transformed['mask'] is not None else 'None'}")
                            
                            # Debug: Show original vs transformed annotations
                            print(f"   Original annotations: {len(annotations)}")
                            for i, ann in enumerate(annotations):
                                print(f"     Original {i}: class={ann['class_id']}, bbox={ann['bbox']}, seg_points={len(ann['segmentation'])}")
                            
                            print(f"   Transformed bboxes: {len(transformed['bboxes'])}")
                            for i, bbox in enumerate(transformed['bboxes']):
                                print(f"     Transformed {i}: bbox={bbox}")
                            
                            if transformed['mask'] is not None:
                                print(f"   Transformed mask: shape={transformed['mask'].shape}, dtype={transformed['mask'].dtype}")
                                print(f"   Mask value range: {transformed['mask'].min():.3f} to {transformed['mask'].max():.3f}")
                                print(f"   Mask non-zero pixels: {np.sum(transformed['mask'] > 0)}")
                        
                        # Create debug image by drawing the transformed labels directly
                        debug_image = transformed["image"].copy()
                        
                        # Draw the transformed segmentation mask as a red half-opaque polygon
                        if transformed["mask"] is not None:
                            mask = transformed["mask"]
                            img_h, img_w = debug_image.shape[:2]
                            
                            # Ensure mask is the right size
                            if mask.shape[:2] != (img_h, img_w):
                                mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                            
                            # Create red overlay for segmentation
                            red_overlay = np.zeros_like(debug_image)
                            red_overlay[mask > 0] = [255, 0, 0]  # Red color
                            
                            # Blend with original image (50% opacity)
                            alpha = 0.5
                            debug_image = cv2.addWeighted(debug_image, 1-alpha, red_overlay, alpha, 0)
                            
                            if aug_idx == 0:  # Debug info
                                mask_area = np.sum(mask > 0)
                                print(f"   Mask: area={mask_area} pixels, coverage={mask_area/(img_h*img_w)*100:.1f}%")
                        
                        # Draw the transformed bounding boxes
                        if transformed["bboxes"]:
                            for i, (bbox, class_id) in enumerate(zip(transformed["bboxes"], transformed["bbox_labels"])):
                                # bbox is in YOLO format [center_x, center_y, width, height] (normalized)
                                # Convert to pixel coordinates
                                img_h, img_w = debug_image.shape[:2]
                                center_x, center_y, width, height = bbox
                                
                                # Convert normalized to absolute coordinates
                                x1 = int((center_x - width/2) * img_w)
                                y1 = int((center_y - height/2) * img_h)
                                x2 = int((center_x + width/2) * img_w)
                                y2 = int((center_y + height/2) * img_h)
                                
                                # Clamp to image bounds
                                x1 = max(0, min(x1, img_w - 1))
                                y1 = max(0, min(y1, img_h - 1))
                                x2 = max(x1 + 1, min(x2, img_w))
                                y2 = max(y1 + 1, min(y2, img_h))
                                
                                # Draw bounding box
                                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Cyan
                                
                                # Draw label
                                label = f"Class{class_id}"
                                cv2.putText(debug_image, label, (x1, y1-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                
                                if aug_idx == 0:  # Debug info
                                    print(f"   BBox {i}: normalized=({center_x:.3f}, {center_y:.3f}, {width:.3f}, {height:.3f}) -> pixels=({x1}, {y1}, {x2}, {y2})")
                                    
                            # Also draw the corrected bounding boxes from the mask for comparison
                            if transformed["mask"] is not None:
                                mask_coords = np.where(transformed["mask"] > 0)
                                if len(mask_coords[0]) > 0:
                                    y_min, y_max = np.min(mask_coords[0]), np.max(mask_coords[0])
                                    x_min, x_max = np.min(mask_coords[1]), np.max(mask_coords[1])
                                    
                                    # Draw the actual mask bounding box in a different color
                                    cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)  # Magenta
                                    
                                    if aug_idx == 0:
                                        print(f"   Mask bbox: ({x_min}, {y_min}) to ({x_max}, {y_max})")
                                        print(f"   Mask bbox vs albumentations bbox: mask=({x_min}, {y_min}, {x_max-x_min}, {y_max-y_min}) vs albu={bbox}")
                        
                        # Save debug image (this replaces the original augmented image when debug is enabled)
                        aug_img_name = f"{base_name}_a-{aug_idx:003d}_debug{original_extension}"
                        aug_img_path = self.target_dir / split / "images" / aug_img_name
                        
                        # Convert to BGR for saving
                        debug_image_bgr = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(aug_img_path), debug_image_bgr)
                        
                        if aug_idx == 0:  # Only print once per image to avoid spam
                            print(f"🔍 Debug image saved: {aug_img_name}")
                                
                    except Exception as debug_e:
                        if aug_idx == 0:  # Only print once per image
                            print(f"⚠️ Debug visualization failed: {debug_e}")
                            import traceback
                            traceback.print_exc()
                        
                        # If debug fails, still save the original augmented image
                        aug_img_name = f"{base_name}_a-{aug_idx:003d}{original_extension}"
                        aug_img_path = self.target_dir / split / "images" / aug_img_name
                        cv2.imwrite(str(aug_img_path), aug_image_bgr)
                else:
                    # Normal mode: save the augmented image without debug overlays
                    aug_img_name = f"{base_name}_a-{aug_idx:003d}{original_extension}"
                    aug_img_path = self.target_dir / split / "images" / aug_img_name
                    cv2.imwrite(str(aug_img_path), aug_image_bgr)

                # Convert transformed mask back to segmentation format
                transformed_mask = transformed["mask"]
                new_segmentation = self.mask_to_segmentation(transformed_mask)
                
                if new_segmentation:
                    # Create new annotations with transformed data
                    new_annotations = []
                    
                    # Use the transformed bounding boxes from albumentations
                    for bbox, class_id in zip(transformed["bboxes"], transformed["bbox_labels"]):
                        # Verify that the bbox actually contains the segmentation
                        # If not, recalculate bbox from the transformed mask
                        if transformed_mask is not None:
                            # Find the actual bounding box of the transformed mask
                            mask_coords = np.where(transformed_mask > 0)
                            if len(mask_coords[0]) > 0:
                                y_min, y_max = np.min(mask_coords[0]), np.max(mask_coords[0])
                                x_min, x_max = np.min(mask_coords[1]), np.max(mask_coords[1])
                                
                                # Convert to normalized YOLO format
                                img_h, img_w = transformed_mask.shape[:2]
                                center_x = (x_min + x_max) / (2 * img_w)
                                center_y = (y_min + y_max) / (2 * img_h)
                                width = (x_max - x_min) / img_w
                                height = (y_max - y_min) / img_h
                                
                                # Use the recalculated bbox to ensure consistency
                                corrected_bbox = [center_x, center_y, width, height]
                                
                                if aug_idx == 0 and self.debug:
                                    print(f"   Corrected bbox {len(new_annotations)}: {corrected_bbox}")
                                
                                new_annotations.append({
                                    'class_id': class_id,
                                    'bbox': corrected_bbox,
                                    'segmentation': new_segmentation
                                })
                            else:
                                # Mask is empty, skip this annotation
                                if aug_idx == 0 and self.debug:
                                    print(f"   Skipping empty mask for class {class_id}")
                                continue
                        else:
                            # No mask, use the albumentations bbox as-is
                            new_annotations.append({
                                'class_id': class_id,
                                'bbox': bbox,
                                'segmentation': new_segmentation
                            })
                    
                    if new_annotations:
                        # Save augmented labels
                        aug_label_name = f"{base_name}_a-{aug_idx:003d}.txt"
                        aug_label_path = self.target_dir / split / "labels" / aug_label_name

                        self.save_yolo_segmentation_label(new_annotations, aug_label_path)
                        
                        # Debug: Show label content for verification
                        if self.debug and aug_idx == 0:
                            print(f"   Label file content:")
                            for i, ann in enumerate(new_annotations):
                                print(f"     Object {i}: class={ann['class_id']}, bbox={ann['bbox']}")
                                print(f"       segmentation points: {len(ann['segmentation'])} coordinates")
                        
                        successful_augmentations += 1
                    else:
                        if aug_idx == 0 and self.debug:
                            print(f"   No valid annotations to save")
                else:
                    if aug_idx == 0 and self.debug:
                        print(f"   Failed to create segmentation from transformed mask")

            except Exception as e:
                if aug_idx < 3:  # Only print errors for first few attempts
                    print(f"⚠️ Augmentation {aug_idx} failed for {base_name}: {e}")
                continue

        return successful_augmentations

    def augment_dataset(self) -> Dict[str, int]:
        """Augment the entire dataset."""
        print("🔄 Starting YOLO segmentation dataset augmentation...")
        print(f"📊 Target: {self.augmentations_per_image} augmentations per image")
        print()

        total_successful = 0
        total_attempted = 0
        results = {}

        for split in ["train", "val"]:
            print(f"📁 Processing {split} split...")

            images_dir = self.source_dir / split / "images"
            labels_dir = self.source_dir / split / "labels"

            if not images_dir.exists():
                print(f"❌ Images directory not found: {images_dir}")
                results[split] = 0
                continue

            # Get image files with supported extensions
            image_files = []
            for ext in self.config['supported_extensions']:
                image_files.extend(list(images_dir.glob(f"*{ext}")))
            
            print(f"📷 Found {len(image_files)} images in {split}")

            split_successful = 0

            for img_path in tqdm(image_files, desc=f"Augmenting {split}"):
                label_path = labels_dir / f"{img_path.stem}.txt"

                successful = self.augment_single_image(img_path, label_path, split)
                split_successful += successful
                total_attempted += self.augmentations_per_image

            total_successful += split_successful
            results[split] = split_successful
            print(f"✅ {split}: {split_successful} successful augmentations")
            print()

        print(f"🎯 Total Summary:")
        print(f"   Attempted: {total_attempted} augmentations")
        print(f"   Successful: {total_successful} augmentations")
        print(f"   Success rate: {total_successful/total_attempted*100:.1f}%")

        # Create updated dataset.yaml
        self.create_dataset_yaml()
        
        return results

    def create_dataset_yaml(self):
        """Create dataset.yaml for the augmented dataset."""
        dataset_config = {
            "path": str(self.target_dir.absolute()),
            "train": "train/images",
            "val": "val/images", 
            "nc": 1,  # Number of classes - update as needed
            "names": ["object"]  # Class names - update as needed
        }

        yaml_path = self.target_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"📝 Created dataset.yaml: {yaml_path}")


def main():
    """Main function to run the YOLO segmentation augmenter from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Segmentation Dataset Augmenter")
    parser.add_argument("--source", "-s", required=True, help="Source dataset directory")
    parser.add_argument("--target", "-t", required=True, help="Target directory for augmented dataset")
    parser.add_argument("--augmentations", "-a", type=int, default=10, help="Number of augmentations per image (default: 10)")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode with detailed logging and visualization")
    
    args = parser.parse_args()
    
    print("🚀 YOLO Segmentation Dataset Augmenter")
    print("=" * 50)
    
    if args.debug:
        print("🔍 Debug mode enabled - will save detailed visualizations and logs")
        print()
    
    augmenter = YoloSegmentationAugmenter(
        source_dir=args.source,
        target_dir=args.target,
        augmentations_per_image=args.augmentations,
        debug=args.debug
    )
    
    results = augmenter.augment_dataset()
    
    if args.debug:
        print("\n🔍 Debug Summary:")
        print("   Debug images saved with '_debug' suffix")
        print("   Label verification images saved with '_labels' suffix") 
        print("   Before/after comparison images saved with '_summary' suffix")
        print("   Check the target directory for all debug outputs")
    
    print(f"\n✅ Augmentation completed! Results: {results}")


if __name__ == "__main__":
    main()