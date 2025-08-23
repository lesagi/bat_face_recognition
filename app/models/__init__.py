"""
Model Factory Package

This package provides factory functions for creating model instances from configuration data.
The factory pattern separates model instantiation from configuration loading, ensuring that
heavy imports (like ultralytics, torch, etc.) only occur when models are actually needed.

Key Components:
- ModelFactory: Protocol interface for type safety
- YOLO factory functions: For creating YOLO segmentation and pose models
- Error handling: Proper exception classes for model creation failures

Usage:
    from models.yolo_factory import create_yolo_model
    from config.loader import load_config
    
    config = load_config()
    seg_config = config.models.segmentation
    model = create_yolo_model(seg_config)
"""

from typing import Protocol, Any
from config.model_config import ImageModelConfig


class ModelFactory(Protocol):
    """Protocol for model factory functions.
    
    This protocol defines the interface that all model factory functions should follow.
    It ensures type safety and consistency across different model creation functions.
    """
    
    def create_model(self, config: ImageModelConfig) -> Any:
        """Create a model instance from configuration.
        
        Args:
            config: ImageModelConfig instance containing model configuration data
            
        Returns:
            Model instance ready for inference or training
            
        Raises:
            ModelConfigurationError: If configuration is invalid
            ModelCreationError: If model creation fails
        """
        ...


# Import exception classes
from .exceptions import ModelConfigurationError, ModelCreationError

# Export the protocol and exceptions for external use
__all__ = ["ModelFactory", "ModelConfigurationError", "ModelCreationError"]