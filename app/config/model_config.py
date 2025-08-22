"""
Configuration class for image processing models.

This module provides the ImageModelConfig class for configuring
different types of YOLO models with their specific parameters.
"""

from typing import Union, Literal
from ultralytics import YOLO


class ImageModelConfig:
    """Configuration class for image processing models."""
    
    def __init__(
        self,
        model: Union[str, YOLO],
        model_type: Literal["yolo_segmentation", "yolo_pose"],
        confidence_threshold: float = 0.3
    ):
        """Initialize model configuration.
        
        Args:
            model: Path to YOLO model file or YOLO model instance
            model_type: Type of model ("yolo_segmentation" or "yolo_pose")
            confidence_threshold: Minimum confidence threshold for detections (0.0-1.0)
        """
        self.model = model
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        
        # Validate confidence threshold
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    @property
    def confidence_threshold(self) -> float:
        """Get the confidence threshold."""
        return self._confidence_threshold
    
    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        """Set the confidence threshold with validation."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        self._confidence_threshold = value
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        model_path = self.model if isinstance(self.model, str) else "YOLO instance"
        return f"ImageModelConfig(model_type='{self.model_type}', model='{model_path}', confidence_threshold={self.confidence_threshold})"
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return self.__repr__() 