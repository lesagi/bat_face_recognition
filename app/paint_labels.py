#!/usr/bin/env python3
"""
Standalone script to paint YOLO segmentation labels on images.

This script reads images and their corresponding label files from a dataset directory
and paints the segmentation masks and bounding boxes on the images.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse


class LabelPainter:
    """Paints YOLO segmentation labels on images."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Colors for different elements
        self.colors = {
            'segmentation': (0, 255, 0),      # Green for segmentation
            'bbox': (255, 0, 0),              # Red for bounding boxes
            'text': (255, 255, 255),          # White for text
            'text_bg': (0, 0, 0)             # Black background for text
        }
    
    def parse_yolo_segmentation_label(self, label_path: Path) -> list:
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
    
    def create_mask_from_segmentation(self, segmentation: list, img_width: int, img_height: int) -> np.ndarray:
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
    
    def paint_labels_on_image(self, image: np.ndarray, annotations: list) -> np.ndarray:
        """Paint all labels on the image."""
        img_height, img_width = image.shape[:2]
        annotated_image = image.copy()
        
        for i, ann in enumerate(annotations):
            class_id = ann['class_id']
            bbox = ann['bbox']
            segmentation = ann['segmentation']
            
            # Draw segmentation mask
            if segmentation:
                mask = self.create_mask_from_segmentation(segmentation, img_width, img_height)
                
                # Create colored overlay
                mask_colored = np.zeros_like(annotated_image)
                mask_colored[mask > 0] = self.colors['segmentation']
                
                # Blend with original image (30% opacity)
                alpha = 0.3
                annotated_image = cv2.addWeighted(annotated_image, 1-alpha, mask_colored, alpha, 0)
                
                # Draw mask contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_image, contours, -1, self.colors['segmentation'], 2)
            
            # Draw bounding box
            if bbox:
                center_x, center_y, width, height = bbox
                
                # Convert normalized to absolute coordinates
                x1 = int((center_x - width/2) * img_width)
                y1 = int((center_y - height/2) * img_height)
                x2 = int((center_x + width/2) * img_width)
                y2 = int((center_y + height/2) * img_height)
                
                # Clamp to image bounds
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(x1 + 1, min(x2, img_width))
                y2 = max(y1 + 1, min(y2, img_height))
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), self.colors['bbox'], 2)
                
                # Draw label with background
                label = f"Class{class_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw text background
                cv2.rectangle(annotated_image, 
                             (x1, y1 - text_height - baseline - 5),
                             (x1 + text_width, y1),
                             self.colors['text_bg'], -1)
                
                # Draw text
                cv2.putText(annotated_image, label, 
                           (x1, y1 - baseline - 5), 
                           font, font_scale, self.colors['text'], thickness)
        
        return annotated_image
    
    def process_dataset(self):
        """Process all images in the dataset directory."""
        images_dir = self.input_dir / "images"
        labels_dir = self.input_dir / "labels"
        
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            return
        
        if not labels_dir.exists():
            print(f"❌ Labels directory not found: {labels_dir}")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        print(f"📷 Found {len(image_files)} images to process")
        print(f"📁 Input directory: {self.input_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print()
        
        processed_count = 0
        
        for img_path in image_files:
            try:
                # Find corresponding label file
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                if not label_path.exists():
                    print(f"⚠️  No label file found for {img_path.name}")
                    continue
                
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"❌ Could not load image: {img_path}")
                    continue
                
                # Parse labels
                annotations = self.parse_yolo_segmentation_label(label_path)
                
                if not annotations:
                    print(f"⚠️  No annotations found in {label_path.name}")
                    continue
                
                # Paint labels on image
                annotated_image = self.paint_labels_on_image(image, annotations)
                
                # Save annotated image
                output_path = self.output_dir / f"{img_path.stem}_annotated{img_path.suffix}"
                cv2.imwrite(str(output_path), annotated_image)
                
                print(f"✅ Processed {img_path.name} -> {output_path.name} ({len(annotations)} annotations)")
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Error processing {img_path.name}: {e}")
                continue
        
        print(f"\n🎯 Processing completed!")
        print(f"   Processed: {processed_count} images")
        print(f"   Output directory: {self.output_dir}")


def main():
    """Main function with hardcoded paths."""
    
    # Hardcoded paths - modify these as needed
    INPUT_DIR = "/Users/MAC/Documents/bat_face_rec/unified_segmentation/data/augemented/train"
    OUTPUT_DIR = "/Users/MAC/Documents/bat_face_rec/unified_segmentation/data/augemented/train_annotated"
    
    print("🎨 Label Painter - YOLO Segmentation Visualization")
    print("=" * 60)
    print(f"📁 Input directory: {INPUT_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory does not exist: {INPUT_DIR}")
        print("Please modify the INPUT_DIR variable in the script to point to your dataset.")
        return
    
    # Create painter and process dataset
    painter = LabelPainter(INPUT_DIR, OUTPUT_DIR)
    painter.process_dataset()


if __name__ == "__main__":
    main()
