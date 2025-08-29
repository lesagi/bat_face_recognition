"""
Image processor package for enhanced image processing capabilities.

This package provides:
- ImageProcessor: Main orchestration class for model-based and plain processing
- ImageTransforms: Static class containing all image processing transforms
- Prediction structures and transformation utilities for augmented image processing
"""

from .input_processor import SiamesePreprocessingPipeline
from .prediction_structures import (
    SegmentationPrediction,
    PosePrediction,
    PredictionBundle,
    PredictionMetadata
)
from .prediction_transforms import (
    TransformationMatrix,
    MaskTransformer,
    KeypointTransformer,
    BoundingBoxTransformer,
    CoordinateMapper,
    PredictionTransformer
)
from .prediction_cache import (
    CacheConfig,
    CacheEntry,
    PredictionCache,
    CacheManager
)

__all__ = [
    "SiamesePreprocessingPipeline",
    "SegmentationPrediction",
    "PosePrediction", 
    "PredictionBundle",
    "PredictionMetadata",
    "TransformationMatrix",
    "MaskTransformer",
    "KeypointTransformer",
    "BoundingBoxTransformer",
    "CoordinateMapper",
    "PredictionTransformer",
    "CacheConfig",
    "CacheEntry",
    "PredictionCache",
    "CacheManager"
]
