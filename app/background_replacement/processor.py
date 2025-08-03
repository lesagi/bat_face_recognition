"""
Background replacement processor for bat face processing.

This module provides background replacement functionality using the generic ImageProcessor
for bat face images with YOLO segmentation models.
"""

import numpy as np
from typing import Callable, Union
from ultralytics import YOLO

from image_processor.processor import ImageProcessor


class BackgroundReplacementProcessor:
    """Background replacement processor for bat face images.

    This class provides a focused interface for background replacement operations,
    while delegating generic image processing to the underlying ImageProcessor.
    For generic segmentation operations, use ImageProcessor directly.
    """

    def __init__(self, model: Union[str, YOLO], confidence_threshold: float = 0.3):
        """Initialize the processor with a YOLO segmentation model.

        Args:
            model: Either a YOLO model instance or path to model file (required)
            confidence_threshold: Confidence threshold for segmentation predictions

        Raises:
            ValueError: If model parameter is invalid
            FileNotFoundError: If model path doesn't exist
            RuntimeError: If model file is not a valid YOLO model
        """
        self.image_processor = ImageProcessor(model, confidence_threshold)

    @property
    def model(self):
        """Access to the underlying YOLO model."""
        return self.image_processor.model

    @property
    def confidence_threshold(self):
        """Access to the confidence threshold."""
        return self.image_processor.confidence_threshold

    def _apply_background_processing(
        self,
        original_image: np.ndarray,
        mask: np.ndarray,
        background_generator: Callable[[int, int], np.ndarray],
    ) -> np.ndarray:
        """Private method to apply background replacement processing.

        Args:
            original_image: Original image array
            mask: Boolean mask for the bat
            background_generator: Function that takes (height, width) and returns background image

        Returns:
            Processed image with background replaced
        """
        height, width = original_image.shape[:2]

        # Create background
        background = background_generator(height, width)

        # Create 3-channel mask
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Extract bat from original image
        bat_roi = original_image * mask_3d

        # Place bat on background
        result_image = background.copy()
        result_image[mask_3d] = bat_roi[mask_3d]

        return result_image

    def apply_background(
        self,
        image_path: str,
        output_path: str,
        background_generator: Callable[[int, int], np.ndarray],
        suffix: str = "",
    ) -> bool:
        """Apply custom background to a bat image using segmentation.

        Args:
            image_path: Path to input image
            output_path: Directory to save output image
            background_generator: Function that takes (height, width) and returns background image
            suffix: Optional suffix for output filename

        Returns:
            True if successful, False otherwise, None if no bat detected
        """
        return self.image_processor.process_image(
            image_path=image_path,
            output_path=output_path,
            processing_function=self._apply_background_processing,
            suffix=suffix,
            background_generator=background_generator,
        )

    def apply_background_batch(
        self,
        input_directory: str,
        output_directory: str,
        background_generator: Callable[[int, int], np.ndarray],
        file_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
        suffix: str = "",
    ) -> dict:
        """Apply background replacement to a batch of images.

        Args:
            input_directory: Directory containing input images
            output_directory: Directory to save processed images
            background_generator: Function that takes (height, width) and returns background image
            file_extensions: Tuple of valid file extensions to process
            suffix: Optional suffix for output filenames

        Returns:
            Dict with processing results: {'successful': int, 'failed': int, 'no_detection': int}
        """
        return self.image_processor.process_batch(
            input_directory=input_directory,
            output_directory=output_directory,
            processing_function=self._apply_background_processing,
            file_extensions=file_extensions,
            suffix=suffix,
            background_generator=background_generator,
        )
