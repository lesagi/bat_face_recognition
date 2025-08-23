"""
YOLO Model Factory Functions

This module provides factory functions for creating YOLO model instances from configuration data.
Heavy imports (ultralytics) are contained within these functions to avoid import bottlenecks
during configuration loading.

The factory functions handle:
- Model validation and error handling
- Lazy loading of heavy dependencies
- Specialized creation for different YOLO model types
"""

import os
from typing import Any, Optional
from .exceptions import ModelConfigurationError, ModelCreationError


def create_yolo_model(config: Any) -> Any:
    """Create a YOLO model instance from configuration.
    
    This is the main factory function that creates YOLO models for both
    segmentation and pose estimation tasks. Heavy imports are contained
    within this function to avoid slowing down configuration loading.
    
    Args:
        config: Model configuration instance containing model configuration data
        
    Returns:
        YOLO model instance ready for inference
        
    Raises:
        ModelConfigurationError: If configuration is invalid
        ModelCreationError: If model creation fails
    """
    # Validate configuration
    if not config:
        raise ModelConfigurationError("Model configuration is required")
    
    if not config.model_path:
        raise ModelConfigurationError("Model path is required")
    
    if not os.path.exists(config.model_path):
        raise ModelConfigurationError(f"Model file not found: {config.model_path}")
    
    # Validate model type
    if config.model_type not in ["yolo_segmentation", "yolo_pose"]:
        raise ModelConfigurationError(f"Unsupported model type: {config.model_type}")
    
    try:
        # Heavy import only happens here, when model is actually needed
        from ultralytics import YOLO
        
        # Create model instance
        model = YOLO(config.model_path)
        
        return model
        
    except ImportError as e:
        raise ModelCreationError(f"ultralytics package not available: {e}")
    except Exception as e:
        raise ModelCreationError(f"Failed to create YOLO model: {e}")


def create_segmentation_model(config: Any) -> Any:
    """Create a YOLO segmentation model with validation.
    
    Specialized factory function for segmentation models that includes
    additional validation specific to segmentation tasks.
    
    Args:
        config: Model configuration instance for segmentation model
        
    Returns:
        YOLO segmentation model instance
        
    Raises:
        ModelConfigurationError: If configuration is invalid for segmentation
        ModelCreationError: If model creation fails
    """
    # Validate that this is a segmentation model
    if config.model_type != "yolo_segmentation":
        raise ModelConfigurationError(
            f"Expected yolo_segmentation model type, got: {config.model_type}"
        )
    
    # Use the main factory function
    model = create_yolo_model(config)
    
    # Additional validation for segmentation models could go here
    # For example, checking if the model actually supports segmentation
    
    return model


def create_pose_model(config: Any) -> Any:
    """Create a YOLO pose estimation model with validation.
    
    Specialized factory function for pose estimation models that includes
    additional validation specific to pose estimation tasks.
    
    Args:
        config: Model configuration instance for pose model
        
    Returns:
        YOLO pose estimation model instance
        
    Raises:
        ModelConfigurationError: If configuration is invalid for pose estimation
        ModelCreationError: If model creation fails
    """
    # Validate that this is a pose model
    if config.model_type != "yolo_pose":
        raise ModelConfigurationError(
            f"Expected yolo_pose model type, got: {config.model_type}"
        )
    
    # Use the main factory function
    model = create_yolo_model(config)
    
    # Additional validation for pose models could go here
    # For example, checking if the model actually supports pose estimation
    
    return model


def validate_model_config(config: Any) -> bool:
    """Validate model configuration without creating the model.
    
    This function performs validation checks on the configuration
    without actually loading the heavy ultralytics dependency or
    creating the model instance.
    
    Args:
        config: Model configuration instance to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ModelConfigurationError: If configuration is invalid
    """
    if not config:
        raise ModelConfigurationError("Model configuration is required")
    
    if not config.model_path:
        raise ModelConfigurationError("Model path is required")
    
    if not os.path.exists(config.model_path):
        raise ModelConfigurationError(f"Model file not found: {config.model_path}")
    
    if config.model_type not in ["yolo_segmentation", "yolo_pose"]:
        raise ModelConfigurationError(f"Unsupported model type: {config.model_type}")
    
    if not (0.0 <= config.confidence_threshold <= 1.0):
        raise ModelConfigurationError(
            f"Confidence threshold must be between 0.0 and 1.0, got: {config.confidence_threshold}"
        )
    
    return True


def get_model_info(config: Any) -> dict:
    """Get information about a model without loading it.
    
    This function provides model information that can be obtained
    without loading the heavy ultralytics dependency.
    
    Args:
        config: Model configuration instance
        
    Returns:
        Dictionary containing model information
    """
    validate_model_config(config)
    
    return {
        "model_path": config.model_path,
        "model_type": config.model_type,
        "confidence_threshold": config.confidence_threshold,
        "file_exists": os.path.exists(config.model_path),
        "file_size": os.path.getsize(config.model_path) if os.path.exists(config.model_path) else None
    }