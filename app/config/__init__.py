"""
Configuration module for Bat Face Recognition Project.

This module provides centralized configuration management for the entire project.
"""

from .loader import ConfigLoader, load_config
from .model_config import ImageModelConfig

__all__ = ["ConfigLoader", "load_config", "ImageModelConfig"] 