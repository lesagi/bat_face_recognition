#!/usr/bin/env python3
"""
Example usage of YOLO Segmentation Augmenter

This script demonstrates how to use the YoloSegmentationAugmenter class
programmatically without the CLI.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo_augmenter.segmentation_augmenter import YoloSegmentationAugmenter


def main():
    """Example usage of the augmenter."""
    
    # Example paths - update these for your use case
    # The source_dir should contain train/ and val/ subdirectories
    source_dir = Path("/Users/MAC/Documents/bat_face_rec/unified_segmentation/data")  # Should have train/images, train/labels, val/images, val/labels
    target_dir = Path("/Users/MAC/Documents/bat_face_rec/unified_segmentation/data/augemented")
    
    print("🦇 YOLO Segmentation Augmentation Example")
    print("=" * 50)
    print(f"📂 Source: {source_dir}")
    print(f"📂 Target: {target_dir}")
    
    # Check if source exists
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        print("Please update the source_dir path in this example script.")
        print("\nExpected directory structure:")
        print("  source_dir/")
        print("  ├── train/")
        print("  │   ├── images/")
        print("  │   └── labels/")
        print("  └── val/")
        print("      ├── images/")
        print("      └── labels/")
        return
    
    # Create augmenter with 5 augmentations per image
    augmenter = YoloSegmentationAugmenter(
        source_dir=str(source_dir),
        target_dir=str(target_dir),
        augmentations_per_image=5,
        debug=True
    )
    
    # Run augmentation
    try:
        results = augmenter.augment_dataset()
        
        print("\n✅ Augmentation completed!")
        print("Results:")
        for split, count in results.items():
            print(f"  {split}: {count} augmentations")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()