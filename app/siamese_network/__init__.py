"""
Image processor package for enhanced image processing capabilities.

This package provides:
- ImageProcessor: Main orchestration class for model-based and plain processing
- ImageTransforms: Static class containing all image processing transforms
"""

from .input_processor import SiamesePreprocessingPipeline

__all__ = ["SiamesePreprocessingPipeline"]
