"""
YOLO Segmentation Training Module.

This module provides functionality for training YOLO segmentation models,
including data preparation and model training utilities.
"""

# Core trainer class
from .trainer import YoloSegmentationTrainer

# Utility functions
from .utils import YoloTrainingUtils

# Configuration constants
from .config import (
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_CONFIDENCE,
    SUPPORTED_BASE_MODELS
)

# Define what gets imported with "from yolo_segmentation_trainer import *"
__all__ = [
    'YoloSegmentationTrainer',
    'YoloTrainingUtils',
    'DEFAULT_EPOCHS',
    'DEFAULT_BATCH_SIZE', 
    'DEFAULT_IMAGE_SIZE',
    'DEFAULT_CONFIDENCE',
    'SUPPORTED_BASE_MODELS'
] 