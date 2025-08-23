"""
Augmenter implementations for different data types.

This module provides specialized augmenters for plain images, YOLO detection,
and YOLO segmentation datasets.
"""

from .plain_image_augmenter import PlainImageAugmenter

__all__ = ['PlainImageAugmenter']
