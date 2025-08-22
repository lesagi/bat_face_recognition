"""
YOLO Data Augmentation Module

This module provides augmentation capabilities for YOLO datasets including:
- Segmentation data augmentation
- Bounding box transformations
- Mask transformations
- CLI interface for batch processing
"""

from .segmentation_augmenter import YoloSegmentationAugmenter
from .cli import main

__all__ = ['YoloSegmentationAugmenter', 'main']