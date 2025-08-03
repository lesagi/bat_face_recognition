"""
Batch processing utilities for segmentation operations.

This module provides functionality for processing multiple images in batch operations.
"""

import os
from typing import Callable
import numpy as np

from utils.image_utils import is_img_file


class BatchProcessor:
    """Utility class for batch processing images."""
    
    def __init__(self, processor):
        """Initialize with a segmentation processor.
        
        Args:
            processor: Processor instance (e.g., SegmentationProcessor)
        """
        self.processor = processor
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         background_generator: Callable[[int, int], np.ndarray],
                         suffix: str = "") -> int:
        """Process all images in a directory with a custom background.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            background_generator: Function that generates backgrounds
            suffix: Suffix for output filenames
            
        Returns:
            Number of successfully processed images
        """
        if not os.path.exists(input_dir):
            print(f"Input directory does not exist: {input_dir}")
            return 0
        
        processed_count = 0
        
        for dirpath, dirnames, filenames in os.walk(input_dir):
            # Create corresponding output directory structure
            relative_path = os.path.relpath(dirpath, input_dir)
            current_output_dir = os.path.join(output_dir, relative_path) if relative_path != '.' else output_dir
            
            for filename in filenames:
                if is_img_file(filename):
                    image_path = os.path.join(dirpath, filename)
                    print(f"Processing: {image_path}")
                    
                    if self.processor.apply_background(image_path, current_output_dir, 
                                                     background_generator, suffix):
                        processed_count += 1
        
        return processed_count
    
    def process_image_list(self, image_paths: list, output_dir: str,
                          background_generator: Callable[[int, int], np.ndarray],
                          suffix: str = "") -> int:
        """Process a specific list of images with a custom background.
        
        Args:
            image_paths: List of image file paths to process
            output_dir: Directory to save processed images
            background_generator: Function that generates backgrounds
            suffix: Suffix for output filenames
            
        Returns:
            Number of successfully processed images
        """
        processed_count = 0
        
        for image_path in image_paths:
            if os.path.exists(image_path) and is_img_file(image_path):
                print(f"Processing: {image_path}")
                
                if self.processor.apply_background(image_path, output_dir, 
                                                 background_generator, suffix):
                    processed_count += 1
            else:
                print(f"Skipping invalid or non-existent file: {image_path}")
        
        return processed_count 