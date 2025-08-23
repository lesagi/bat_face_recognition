"""
YOLO Data Augmentation Module

This module provides augmentation capabilities for YOLO datasets including:
- Segmentation data augmentation
- Bounding box transformations
- Mask transformations
- CLI interface for batch processing
- NEW: Modular augmenter system with systematic transformations
"""

# Legacy imports (maintain backward compatibility)
from .segmentation_augmenter import YoloSegmentationAugmenter

# New modular system imports
from .base import BaseAugmenter, AugmentationPipelineFactory, AugmentationConfig
from .augmenters import PlainImageAugmenter
from .cli import cli

__all__ = [
    # Legacy
    'YoloSegmentationAugmenter',
    
    # New modular system
    'BaseAugmenter',
    'AugmentationPipelineFactory', 
    'AugmentationConfig',
    'PlainImageAugmenter',
    'cli'
]