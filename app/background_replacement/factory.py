"""
Factory functions for creating segmentation processing instances.

This module provides convenient factory functions for creating various processor instances.
"""

import os
from typing import Union
from ultralytics import YOLO

from .processor import BackgroundReplacementProcessor
from .batch_processor import BatchProcessor


def create_batch_processor(
    model: Union[str, YOLO],
    confidence_threshold: float = 0.3,
) -> BatchProcessor:
    """Factory function to create a batch processor instance.

    Args:
        model: Model to use for creating the processor (required)
        confidence_threshold: Confidence threshold for the processor

    Returns:
        BatchProcessor instance

    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If model path doesn't exist
        RuntimeError: If model file is not a valid YOLO segmentation model

    Examples:
        # Create batch processor with model path
        batch_processor = create_batch_processor(model="/path/to/model.pt")

        # Create with custom confidence threshold
        batch_processor = create_batch_processor(
            model="/path/to/model.pt",
            confidence_threshold=0.7
        )
    """
    processor = create_processor(model=model, confidence_threshold=confidence_threshold)
    return BatchProcessor(processor)


def create_processor(
    model: Union[str, YOLO], confidence_threshold: float = 0.3
) -> BackgroundReplacementProcessor:
    """Factory function to create a processor instance.

    Args:
        model: Either a YOLO model instance or path to model file (required)
        confidence_threshold: Confidence threshold for segmentation

    Returns:
        BackgroundReplacementProcessor instance

    Raises:
        ValueError: If model parameter is invalid or confidence threshold is out of range
        FileNotFoundError: If model path doesn't exist
        RuntimeError: If model file is not a valid YOLO segmentation model

    Examples:
        # Use custom model path
        processor = create_processor("/path/to/my/model.pt")

        # Use existing YOLO model
        my_model = YOLO("yolov8n-seg.pt")
        processor = create_processor(my_model)

        # Custom model with confidence threshold
        processor = create_processor("/path/to/model.pt", confidence_threshold=0.7)
    """
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError(
            f"Confidence threshold must be between 0.0 and 1.0, got {confidence_threshold}"
        )

    return BackgroundReplacementProcessor(
        model=model, confidence_threshold=confidence_threshold
    )


def load_model(model_path: str) -> YOLO:
    """Utility function to load and validate a YOLO model.

    Args:
        model_path: Path to the model file

    Returns:
        Loaded YOLO model instance

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If file extension is not supported by YOLO
        RuntimeError: If model cannot be loaded or is not a valid YOLO model

    Example:
        model = load_model("/path/to/model.pt")
        processor = create_processor(model)
    """
    # Validate file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.isfile(model_path):
        raise ValueError(f"Model path is not a file: {model_path}")

    # Check for valid YOLO model extensions
    valid_extensions = {".pt", ".onnx", ".engine", ".torchscript", ".mlmodel"}
    file_extension = os.path.splitext(model_path)[1].lower()

    if file_extension not in valid_extensions:
        raise ValueError(
            f"Invalid model file extension '{file_extension}'. "
            f"YOLO models should have one of these extensions: {', '.join(sorted(valid_extensions))}"
        )

    # Try to load the model
    try:
        model = YOLO(model_path)

        # Basic validation - try to access model properties
        if not hasattr(model, "model"):
            raise RuntimeError("Loaded model does not appear to be a valid YOLO model")

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model from '{model_path}': {e}")
