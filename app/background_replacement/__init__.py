"""
Background replacement processing module.

This module provides functionality for replacing backgrounds in images using segmentation models.
"""

# Core processor classes
from .processor import BackgroundReplacementProcessor
from .batch_processor import BatchProcessor

# Background utilities (imported from separate background_generation module)
from background_generation import BackgroundGenerator, BackgroundPresets

# Factory functions
from .factory import create_processor, create_batch_processor, load_model

# Define what gets imported with "from background_replacement import *"
__all__ = [
    # Core classes
    "BackgroundReplacementProcessor",
    "BatchProcessor",
    "BackgroundGenerator",
    "BackgroundPresets",
    # Factory functions
    "create_processor",
    "create_batch_processor",
    "load_model",
]
