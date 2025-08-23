"""
Plain image augmenter for images without labels or annotations.

This augmenter processes directories of plain images and creates augmented versions
using the systematic transformation approach.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List
from ..base.base_augmenter import BaseAugmenter


class PlainImageAugmenter(BaseAugmenter):
    """Augments plain images without any labels or annotations."""
    
    def __init__(self, source_dir: str, target_dir: str, preset: str = 'medium', 
                 config_path: str = None, debug: bool = False):
        super().__init__(source_dir, target_dir, preset, config_path, debug)
    
    def setup_directories(self):
        """Create target directory for plain images."""
        self.target_dir.mkdir(parents=True, exist_ok=True)
        if self.debug:
            print(f"✅ Created target directory: {self.target_dir}")
    
    def validate_input(self) -> bool:
        """Validate that source directory contains images."""
        if not super().validate_input():
            return False
        
        # Check if there are any image files
        image_files = self.get_image_files(self.source_dir)
        if not image_files:
            print(f"❌ No image files found in {self.source_dir}")
            print(f"   Supported formats: {', '.join(self.get_supported_extensions())}")
            return False
        
        print(f"✅ Found {len(image_files)} images in source directory")
        return True
    
    def _get_items_to_process(self) -> List[Path]:
        """Get list of image files to process."""
        return self.get_image_files(self.source_dir)
    
    def augment_single_item(self, image_path: Path, index: int) -> bool:
        """Create augmented versions of a single image."""
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return False
        
        base_name = image_path.stem
        extension = image_path.suffix
        
        successful_augmentations = 0
        
        # Create the specified number of augmentations
        for aug_idx in range(self.total_augmentations):
            try:
                # Apply augmentation
                augmented = self.transform(image=image)
                aug_image = augmented['image']
                
                # Generate output filename
                output_filename = self.get_output_filename(base_name, aug_idx, extension)
                output_path = self.target_dir / output_filename
                
                # Save augmented image
                self.save_image(aug_image, output_path)
                
                successful_augmentations += 1
                
                if self.debug and aug_idx < 3:  # Only show first few for debug
                    print(f"   Created augmentation {aug_idx}: {output_filename}")
                
            except Exception as e:
                if aug_idx < 3:  # Only show first few errors
                    print(f"⚠️ Augmentation {aug_idx} failed for {base_name}: {e}")
                continue
        
        if self.debug:
            print(f"✅ {base_name}: {successful_augmentations}/{self.total_augmentations} augmentations created")
        
        return successful_augmentations > 0
