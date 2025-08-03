"""
Background generation module for image processing.

This module provides utilities for creating various types of backgrounds
that can be used for image processing and background replacement tasks.
"""

# Core classes
from .background_generator import BackgroundGenerator
from .background_presets import BackgroundPresets

# Define what gets imported with "from background_generation import *"
__all__ = [
    'BackgroundGenerator',
    'BackgroundPresets',
] 