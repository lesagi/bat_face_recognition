"""
Configuration class for image processing models.

This module provides the ImageModelConfig class for configuring
different types of YOLO models with their specific parameters.
"""

from typing import Literal


class ImageModelConfig:
    """Pure data holder for model configuration from config.yml"""
    
    def __init__(
        self,
        model_path: str,
        model_type: Literal["yolo_segmentation", "yolo_pose"],
        confidence_threshold: float = 0.3
    ):
        """Initialize model configuration.
        
        Args:
            model_path: Path to YOLO model file
            model_type: Type of model ("yolo_segmentation" or "yolo_pose")
            confidence_threshold: Minimum confidence threshold for detections (0.0-1.0)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        
        # Validate confidence threshold
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"ImageModelConfig(model_type='{self.model_type}', model_path='{self.model_path}', confidence_threshold={self.confidence_threshold})"
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return self.__repr__() 