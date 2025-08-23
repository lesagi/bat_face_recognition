"""
Image Processor Module

This module provides comprehensive image processing capabilities:
- ImageTransforms: Image transformations and model-based processing

Models are created on-demand using factory functions from the models package.
Configuration is loaded automatically when needed.
"""

from .transforms import ImageTransforms

__all__ = ["ImageTransforms"]
