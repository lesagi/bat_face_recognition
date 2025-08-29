"""
YOLO Data Augmentation Module

This module provides augmentation capabilities for YOLO datasets including:
- Plain image augmentation
- CLI interface for batch processing
- Modular augmenter system with systematic transformations
"""

# New modular system imports
from .base import BaseAugmenter, AugmentationPipelineFactory, AugmentationConfig
from .augmenters import PlainImageAugmenter
from .cli import cli

__all__ = [
    # New modular system
    'BaseAugmenter',
    'AugmentationPipelineFactory', 
    'AugmentationConfig',
    'PlainImageAugmenter',
    'cli'
]