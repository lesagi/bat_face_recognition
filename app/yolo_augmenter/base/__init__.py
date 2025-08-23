"""
Base augmenter module for the YOLO augmenter system.

This module provides the foundation classes and utilities for all augmenter types.
"""

from .base_augmenter import BaseAugmenter
from .pipeline_factory import AugmentationPipelineFactory
from .config import AugmentationConfig

__all__ = ['BaseAugmenter', 'AugmentationPipelineFactory', 'AugmentationConfig']
