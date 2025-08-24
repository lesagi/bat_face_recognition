#!/usr/bin/env python3
"""
Square Segmentation Cropper for YOLO Datasets

This module provides functionality to crop square regions around YOLO segmentation
boundaries with configurable buffer and random color padding for edge cases.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SquareSegmentationCropper:
    """Crops square regions around YOLO segmentation boundaries with buffer and padding."""
    
    def __init__(self, 
                 input_dir: str, 
                 output_dir: str, 
                 buffer_multiplier: Optional[float] = None,
                 output_format: Optional[str] = None,
                 quality: Optional[int] = None,
                 verbose: bool = False):
        """
        Initialize the SquareSegmentationCropper.
        
        Args:
            input_dir: Directory containing images and labels
            output_dir: Directory to save cropped images and labels
            buffer_multiplier: Buffer around segmentation (default: from config or 1.1)
            output_format: Output image format (default: from config or jpg)
            quality: Image quality for lossy formats (default: from config or 95)
            verbose: Enable verbose logging
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Load configuration
        self.config = self._load_config()
        
        # Set parameters with config fallbacks
        self.buffer_multiplier = buffer_multiplier or self.config.get('default_buffer_multiplier', 1.1)
        self.output_format = (output_format or self.config.get('default_output_format', 'jpg')).lower()
        self.quality = quality or self.config.get('default_quality', 95)
        self.verbose = verbose
        
        # Validate input directory
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        # Create output directory structure
        self.setup_directories()
        
        # Supported image extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from app/config if available."""
        if load_config and YOLOSegmentationConfig:
            try:
                config = load_config()
                yolo_config = config.yolo_segmentation
                square_crop_config = yolo_config.square_crop
                
                # Access dictionary values directly since square_crop_config is already a dict
                return {
                    'default_buffer_multiplier': square_crop_config.get('default_buffer_multiplier', 1.1),
                    'default_output_format': square_crop_config.get('default_output_format', 'jpg'),
                    'default_quality': square_crop_config.get('default_quality', 95),
                    'default_padding_strategy': square_crop_config.get('default_padding_strategy', 'random_colors'),
                    'auto_discover_datasets': square_crop_config.get('auto_discover_datasets', True),
                }
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
        
        # Default config
        return {
            'default_buffer_multiplier': 1.1,
            'default_output_format': 'jpg',
            'default_quality': 95,
            'default_padding_strategy': 'random_colors',
            'auto_discover_datasets': True,
        }
    
    def setup_directories(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
    
    def parse_yolo_segmentation_label(self, label_path: Path) -> List[Dict[str, Any]]:
        """
        Parse YOLO segmentation label file.
        
        Args:
            label_path: Path to the label file
            
        Returns:
            List of segmentation objects with class_id and polygon coordinates
        """
        objects = []
        
        try:
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 7:  # class_id + at least 3 points (6 coordinates)
                        logger.warning(f"Invalid label format in {label_path}:{line_num}: {line}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        # Convert normalized coordinates to absolute
                        coords = []
                        for i in range(1, len(parts), 2):
                            if i + 1 < len(parts):
                                x = float(parts[i])
                                y = float(parts[i + 1])
                                coords.extend([x, y])
                        
                        if len(coords) >= 6:  # At least 3 points
                            objects.append({
                                'class_id': class_id,
                                'coords': coords
                            })
                        else:
                            logger.warning(f"Insufficient coordinates in {label_path}:{line_num}")
                    
                    except ValueError as e:
                        logger.warning(f"Invalid number format in {label_path}:{line_num}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error reading label file {label_path}: {e}")
        
        return objects
    
    def calculate_segmentation_bounds(self, 
                                   objects: List[Dict[str, Any]], 
                                   image_width: int, 
                                   image_height: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate bounding box around all segmentation objects.
        
        Args:
            objects: List of segmentation objects
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max) or None if no objects
        """
        if not objects:
            return None
        
        x_coords = []
        y_coords = []
        
        for obj in objects:
            coords = obj['coords']
            for i in range(0, len(coords), 2):
                x = coords[i] * image_width
                y = coords[i + 1] * image_height
                x_coords.append(x)
                y_coords.append(y)
        
        if not x_coords or not y_coords:
            return None
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return int(x_min), int(y_min), int(x_max), int(y_max)
    
    def calculate_square_crop(self, 
                            bounds: Tuple[int, int, int, int], 
                            image_width: int, 
                            image_height: int) -> Tuple[int, int, int, int]:
        """
        Calculate square crop dimensions with buffer.
        
        Args:
            bounds: Bounding box (x_min, y_min, x_max, y_max)
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            Tuple of (x, y, width, height) for the crop
        """
        x_min, y_min, x_max, y_max = bounds
        
        # Calculate center of segmentation
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # Calculate dimensions with buffer
        width = int((x_max - x_min) * self.buffer_multiplier)
        height = int((y_max - y_min) * self.buffer_multiplier)
        
        # Make it square by choosing the bigger dimension
        size = max(width, height)
        
        # Calculate crop coordinates
        crop_x = center_x - size // 2
        crop_y = center_y - size // 2
        
        return crop_x, crop_y, size, size
    
    def generate_random_colors(self, size: int) -> np.ndarray:
        """
        Generate random non-solid colors for padding.
        
        Args:
            size: Size of the color array to generate
            
        Returns:
            Array of random colors (BGR format)
        """
        # Generate random colors with some variation to avoid solid appearance
        colors = np.random.randint(0, 256, (size, 3), dtype=np.uint8)
        
        # Add some noise to make it less uniform
        noise = np.random.randint(-20, 21, (size, 3), dtype=np.int16)
        colors = np.clip(colors.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return colors
    
    def crop_image_with_padding(self, 
                               image: np.ndarray, 
                               crop_x: int, 
                               crop_y: int, 
                               crop_size: int) -> np.ndarray:
        """
        Crop image with padding for edge cases.
        
        Args:
            image: Input image
            crop_x: X coordinate of crop
            crop_y: Y coordinate of crop
            crop_size: Size of the crop
            
        Returns:
            Cropped and padded image
        """
        img_height, img_width = image.shape[:2]
        
        # Create output image
        output = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        
        # Calculate source and destination regions
        src_x_start = max(0, crop_x)
        src_y_start = max(0, crop_y)
        src_x_end = min(img_width, crop_x + crop_size)
        src_y_end = min(img_height, crop_y + crop_size)
        
        # Calculate destination regions
        dst_x_start = max(0, -crop_x)
        dst_y_start = max(0, -crop_y)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        # Copy valid image region
        if src_x_end > src_x_start and src_y_end > src_y_start:
            output[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                image[src_y_start:src_y_end, src_x_start:src_x_end]
        
        # Fill padding areas with random colors
        if crop_x < 0 or crop_y < 0 or crop_x + crop_size > img_width or crop_y + crop_size > img_height:
            # Generate random colors for padding
            padding_colors = self.generate_random_colors(crop_size * crop_size)
            padding_colors = padding_colors.reshape(crop_size, crop_size, 3)
            
            # Apply padding where needed
            if crop_x < 0:
                output[:, :dst_x_start] = padding_colors[:, :dst_x_start]
            if crop_x + crop_size > img_width:
                output[:, dst_x_end:] = padding_colors[:, dst_x_end:]
            if crop_y < 0:
                output[:dst_y_start, :] = padding_colors[:dst_y_start, :]
            if crop_y + crop_size > img_height:
                output[dst_y_end:, :] = padding_colors[dst_y_end:, :]
        
        return output
    
    def transform_coordinates(self, 
                            coords: List[float], 
                            crop_x: int, 
                            crop_y: int, 
                            crop_size: int,
                            image_width: int,
                            image_height: int) -> List[float]:
        """
        Transform coordinates from original image space to crop space.
        
        Args:
            coords: List of normalized coordinates [x1, y1, x2, y2, ...]
            crop_x: X coordinate of crop
            crop_y: Y coordinate of crop
            crop_size: Size of the crop
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            Transformed normalized coordinates
        """
        transformed = []
        
        for i in range(0, len(coords), 2):
            # Convert normalized to absolute coordinates
            x_abs = coords[i] * image_width
            y_abs = coords[i + 1] * image_height
            
            # Transform to crop space
            x_crop = x_abs - crop_x
            y_crop = y_abs - crop_y
            
            # Convert back to normalized coordinates
            x_norm = x_crop / crop_size
            y_norm = y_crop / crop_size
            
            # Clip to valid range [0, 1]
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            
            transformed.extend([x_norm, y_norm])
        
        return transformed
    
    def process_single_image(self, image_path: Path, label_path: Path) -> bool:
        """
        Process a single image-label pair.
        
        Args:
            image_path: Path to the image file
            label_path: Path to the label file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            img_height, img_width = image.shape[:2]
            
            # Parse label
            objects = self.parse_yolo_segmentation_label(label_path)
            if not objects:
                logger.warning(f"No valid segmentation objects in {label_path}")
                return False
            
            # Calculate bounds
            bounds = self.calculate_segmentation_bounds(objects, img_width, img_height)
            if bounds is None:
                logger.warning(f"Could not calculate bounds for {label_path}")
                return False
            
            # Calculate crop
            crop_x, crop_y, crop_size, _ = self.calculate_square_crop(bounds, img_width, img_height)
            
            # Crop image
            cropped_image = self.crop_image_with_padding(image, crop_x, crop_y, crop_size)
            
            # Transform coordinates
            transformed_objects = []
            for obj in objects:
                transformed_coords = self.transform_coordinates(
                    obj['coords'], crop_x, crop_y, crop_size, img_width, img_height
                )
                transformed_objects.append({
                    'class_id': obj['class_id'],
                    'coords': transformed_coords
                })
            
            # Save cropped image
            output_image_path = self.output_dir / "images" / f"{image_path.stem}_cropped.{self.output_format}"
            
            if self.output_format == 'jpg':
                cv2.imwrite(str(output_image_path), cropped_image, 
                           [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            else:
                cv2.imwrite(str(output_image_path), cropped_image)
            
            # Save transformed label
            output_label_path = self.output_dir / "labels" / f"{label_path.stem}_cropped.txt"
            with open(output_label_path, 'w') as f:
                for obj in transformed_objects:
                    line = f"{obj['class_id']} {' '.join(map(str, obj['coords']))}\n"
                    f.write(line)
            
            if self.verbose:
                logger.info(f"Processed {image_path.name} -> {output_image_path.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return False
    
    def process_dataset(self) -> Dict[str, Any]:
        """
        Process the entire dataset.
        
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Starting square crop processing...")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Buffer multiplier: {self.buffer_multiplier}")
        
        # Find all image files
        image_files = []
        for ext in self.supported_extensions:
            image_files.extend(self.input_dir.glob(f"images/*{ext}"))
            image_files.extend(self.input_dir.glob(f"images/*{ext.upper()}"))
        
        if not image_files:
            logger.error(f"No image files found in {self.input_dir}")
            return {"success": False, "error": "No image files found"}
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Process each image
        successful = 0
        failed = 0
        
        for image_path in image_files:
            # Find corresponding label
            label_path = self.input_dir / "labels" / f"{image_path.stem}.txt"
            
            if not label_path.exists():
                logger.warning(f"No label file found for {image_path.name}")
                continue
            
            if self.process_single_image(image_path, label_path):
                successful += 1
            else:
                failed += 1
        
        # Summary
        logger.info(f"Processing complete!")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total: {len(image_files)}")
        
        return {
            "success": True,
            "total_images": len(image_files),
            "successful": successful,
            "failed": failed
        }
