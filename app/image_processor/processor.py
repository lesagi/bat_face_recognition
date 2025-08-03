"""
Enhanced image processor for both model-based and plain image processing.

This module provides a comprehensive image processing pipeline that can handle:
- Model-based processing (requires YOLO segmentation/detection models)
- Plain image processing (no model required)
- Automatic detection of processing type based on function signatures
- Batch processing and complex pipelines
"""

import os
import inspect
from typing import Union, Optional, Tuple, List, Callable

import cv2
import numpy as np
from ultralytics import YOLO

from utils.image_utils import strip_filename_from_path
from .transforms import ImageTransforms


class ImageProcessor:
    """Comprehensive image processor for both model-based and plain image processing."""

    def __init__(
        self, model: Union[str, YOLO, None] = None, confidence_threshold: float = 0.3
    ):
        """Initialize the image processor with optional YOLO segmentation model.

        Args:
            model: Path to YOLO model file, YOLO model instance, or None for plain processing
            confidence_threshold: Minimum confidence threshold for detections (0.0-1.0)
        """
        self.model = None
        self.confidence_threshold = confidence_threshold

        if model is not None:
            if isinstance(model, str):
                self._validate_model_path(model)
                try:
                    self.model = YOLO(model)
                    self._validate_segmentation_model(self.model, model)
                except Exception as e:
                    raise RuntimeError(f"Failed to load YOLO model from {model}: {e}")
            elif isinstance(model, YOLO):
                self._validate_segmentation_model(model, "provided YOLO instance")
                self.model = model
            else:
                raise ValueError(
                    "Model must be a path to YOLO model file (.pt) or YOLO instance"
                )

    def _validate_model_path(self, model_path: str) -> None:
        """Validate that the model path exists and has correct extension.

        Args:
            model_path: Path to the model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file has incorrect extension
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not model_path.endswith((".pt", ".onnx")):
            raise ValueError(
                f"Model file must have .pt or .onnx extension, got: {model_path}"
            )

    def _validate_segmentation_model(self, model: YOLO, model_source: str) -> None:
        """Validate that the YOLO model supports segmentation.

        Args:
            model: YOLO model instance
            model_source: Description of model source for error messages

        Raises:
            ValueError: If model doesn't support segmentation
        """
        # Test the model with a small dummy image to check capabilities
        dummy_image = np.ones((100, 100, 3), dtype=np.uint8)

        try:
            results = model.predict(dummy_image, verbose=False)
            if not hasattr(results[0], "masks") or results[0].masks is None:
                # Check if it's a detection model that can still be used
                if hasattr(results[0], "boxes"):
                    print(
                        f"Warning: {model_source} appears to be a detection model, "
                        "not segmentation. Some features may be limited."
                    )
                else:
                    raise ValueError(
                        f"Model from {model_source} doesn't support detection or segmentation"
                    )
        except Exception as e:
            raise ValueError(f"Failed to validate model from {model_source}: {e}")

    # =============================================================================
    # CORE BUSINESS LOGIC (Essential for ImageProcessor functionality)
    # =============================================================================

    def _load_image_array(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Load image as numpy array from path or return existing array.

        Args:
            image: Input image as numpy array or path to image file

        Returns:
            Image as numpy array

        Raises:
            FileNotFoundError: If image path doesn't exist
            ValueError: If image cannot be loaded
        """
        if isinstance(image, str):
            # Load image from path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")

            loaded_image = cv2.imread(image)
            if loaded_image is None:
                raise ValueError(f"Could not load image from path: {image}")

            return loaded_image
        elif isinstance(image, np.ndarray):
            # Use provided numpy array directly
            return image
        else:
            raise ValueError("Image must be a file path (str) or numpy array")

    def segment_image(
        self, image: Union[str, np.ndarray]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Segment image using YOLO model and return original image with mask.

        Args:
            image: Input image as numpy array or path to image file

        Returns:
            Tuple of (original_image, mask) if object detected, None otherwise
        """
        if self.model is None:
            raise RuntimeError(
                "No segmentation model loaded. Cannot perform segmentation."
            )

        # Handle both image path and numpy array inputs
        image_array = self._load_image_array(image)

        try:
            # Run segmentation
            results = self.model.predict(image_array, conf=self.confidence_threshold)[0]

            if not results or not results.masks:
                return None

            # Get the first mask
            mask = results.masks.data[0].cpu().numpy()

            # Resize mask to match image dimensions if needed
            if mask.shape != image_array.shape[:2]:
                mask = cv2.resize(mask, (image_array.shape[1], image_array.shape[0]))

            # Convert to binary mask
            mask = (mask > 0.5).astype(np.uint8)

            return image_array, mask

        except Exception as e:
            raise RuntimeError(f"Segmentation failed: {e}")

    def process_with_model(
        self,
        image_path: str,
        output_path: str,
        processing_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        suffix: str = "",
        **processing_kwargs,
    ) -> bool:
        """Process an image using segmentation model and a custom processing function.

        Args:
            image_path: Path to input image
            output_path: Directory to save output image
            processing_function: Function that takes (original_image, mask) and returns processed image
            suffix: Optional suffix for output filename
            **processing_kwargs: Additional arguments to pass to the processing function

        Returns:
            True if successful, False otherwise, None if no object detected
        """
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Use process_without_model() for plain processing."
            )

        try:
            # Step 1: Segment the image
            segmentation_result = self.segment_image(image_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Failed to load image {image_path}: {e}")
            return False
        except RuntimeError as e:
            print(f"Segmentation error for {image_path}: {e}")
            return False

        if segmentation_result is None:
            print(f"No object detected in image: {image_path}")
            return None

        original_image, mask = segmentation_result

        try:
            # Step 2: Apply custom processing function
            processed_image = processing_function(
                original_image, mask, **processing_kwargs
            )
        except Exception as e:
            print(f"Processing function failed for {image_path}: {e}")
            return False

        # Step 3: Save processed image
        return self._save_processed_image(
            processed_image, image_path, output_path, suffix
        )

    def process_without_model(
        self,
        image_path: str,
        output_path: str,
        processing_function: Callable[[np.ndarray], np.ndarray],
        suffix: str = "",
        **processing_kwargs,
    ) -> bool:
        """Process an image using plain processing function (no model required).

        Args:
            image_path: Path to input image
            output_path: Directory to save output image
            processing_function: Function that takes image array and returns processed image
            suffix: Optional suffix for output filename
            **processing_kwargs: Additional arguments to pass to the processing function

        Returns:
            True if successful, False otherwise
        """
        try:
            # Step 1: Load the image
            original_image = self._load_image_array(image_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Failed to load image {image_path}: {e}")
            return False

        try:
            # Step 2: Apply custom processing function
            processed_image = processing_function(original_image, **processing_kwargs)
        except Exception as e:
            print(f"Processing function failed for {image_path}: {e}")
            return False

        # Step 3: Save processed image
        return self._save_processed_image(
            processed_image, image_path, output_path, suffix
        )

    # =============================================================================
    # AUTO-DETECTION AND PIPELINE PROCESSING
    # =============================================================================

    def _detect_processing_type(self, processing_function: Callable) -> str:
        """Automatically detect if function expects model-based or plain processing.

        Args:
            processing_function: Function to inspect

        Returns:
            'model' if function expects (image, mask) parameters
            'plain' if function expects (image) parameter only
        """
        try:
            sig = inspect.signature(processing_function)
            params = list(sig.parameters.values())

            # Filter to only positional parameters (excluding **kwargs, *args)
            positional_params = [
                p
                for p in params
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]

            if len(positional_params) < 2:
                return "plain"

            # Check if second parameter is likely a mask by name or type annotation
            second_param = positional_params[1]

            # Check parameter name for mask-related keywords
            mask_keywords = {"mask", "segmentation", "prediction"}
            if any(keyword in second_param.name.lower() for keyword in mask_keywords):
                return "model"

            # Check type annotation - if it's np.ndarray, likely a mask
            if second_param.annotation == np.ndarray:
                return "model"

            # If we have specific known mask-based functions, identify them explicitly
            mask_functions = {
                "apply_background_replacement",
                "create_blurred_background",
                "create_color_background",
                "crop_square_around_segmentation",
            }
            if processing_function.__name__ in mask_functions:
                return "model"

            # Default to plain for configuration parameters
            return "plain"

        except Exception:
            # If inspection fails, default to plain processing
            return "plain"

    def process_auto(
        self,
        image_path: str,
        output_path: str,
        processing_function: Callable,
        suffix: str = "",
        **processing_kwargs,
    ) -> bool:
        """Automatically process image based on function signature.

        Args:
            image_path: Path to input image
            output_path: Directory to save output image
            processing_function: Function that takes either (image) or (image, mask)
            suffix: Optional suffix for output filename
            **processing_kwargs: Additional arguments to pass to the processing function

        Returns:
            True if successful, False otherwise, None if no object detected (model-based only)
        """
        processing_type = self._detect_processing_type(processing_function)

        if processing_type == "model":
            if self.model is None:
                raise RuntimeError(
                    "Processing function expects model-based input (image, mask) but no model is loaded. "
                    "Initialize with ImageProcessor(model_path) or use a function that takes only (image)."
                )
            return self.process_with_model(
                image_path,
                output_path,
                processing_function,
                suffix,
                **processing_kwargs,
            )
        else:
            return self.process_without_model(
                image_path,
                output_path,
                processing_function,
                suffix,
                **processing_kwargs,
            )

    def process_pipeline(
        self,
        image_path: str,
        output_path: str,
        processing_steps: List[Tuple[Callable, dict]],
        suffix: str = "",
    ) -> bool:
        """Process image through a pipeline of multiple processing steps with auto-detection.

        Args:
            image_path: Path to input image
            output_path: Directory to save output image
            processing_steps: List of (function, kwargs) tuples
            suffix: Optional suffix for output filename

        Returns:
            True if successful, False otherwise

        Example:
            steps = [
                (detect_and_crop_function, {}),  # Auto-detected as model-based
                (ImageTransforms.resize_image, {'target_size': (224, 224)}),  # Auto-detected as plain
                (ImageTransforms.normalize_image, {'scale': 255.0})  # Auto-detected as plain
            ]
        """
        try:
            current_image = self._load_image_array(image_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Failed to load image {image_path}: {e}")
            return False

        for processing_function, kwargs in processing_steps:
            try:
                processing_type = self._detect_processing_type(processing_function)

                if processing_type == "model":
                    if self.model is None:
                        raise RuntimeError(
                            f"Processing step {processing_function.__name__} requires model "
                            "but no model is loaded"
                        )
                    # For model steps, we need segmentation results
                    segmentation_result = self.segment_image(current_image)
                    if segmentation_result is None:
                        print(
                            f"No object detected for model step: {processing_function.__name__}"
                        )
                        return None
                    current_image, mask = segmentation_result
                    current_image = processing_function(current_image, mask, **kwargs)
                else:
                    # Plain processing
                    current_image = processing_function(current_image, **kwargs)

            except Exception as e:
                print(f"Processing step {processing_function.__name__} failed: {e}")
                return False

        # Save final result
        return self._save_processed_image(
            current_image, image_path, output_path, suffix
        )

    # =============================================================================
    # BATCH PROCESSING METHODS
    # =============================================================================

    def process_batch(
        self,
        input_directory: str,
        output_directory: str,
        processing_function: Callable,
        file_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
        suffix: str = "",
        **processing_kwargs,
    ) -> dict:
        """Process a batch of images with automatic model/plain detection.

        Args:
            input_directory: Directory containing input images
            output_directory: Directory to save processed images
            processing_function: Processing function (auto-detects if model-based or plain)
            file_extensions: Tuple of valid file extensions to process
            suffix: Optional suffix for output filenames
            **processing_kwargs: Additional arguments to pass to the processing function

        Returns:
            Dict with processing results: {'successful': int, 'failed': int, 'no_detection': int}
        """
        if not os.path.exists(input_directory):
            raise FileNotFoundError(f"Input directory not found: {input_directory}")

        results = {"successful": 0, "failed": 0, "no_detection": 0}

        # Auto-detect processing type
        processing_type = self._detect_processing_type(processing_function)

        if processing_type == "model" and self.model is None:
            raise RuntimeError(
                f"Processing function {processing_function.__name__} expects model-based input "
                "but no model is loaded. Initialize with ImageProcessor(model_path) or use a "
                "function that takes only (image)."
            )

        # Get all image files in the directory
        image_files = [
            f
            for f in os.listdir(input_directory)
            if f.lower().endswith(file_extensions)
        ]

        if not image_files:
            print(f"No image files found in {input_directory}")
            return results

        print(
            f"Processing {len(image_files)} images using {processing_type} processing..."
        )

        for image_file in image_files:
            image_path = os.path.join(input_directory, image_file)

            if processing_type == "model":
                result = self.process_with_model(
                    image_path=image_path,
                    output_path=output_directory,
                    processing_function=processing_function,
                    suffix=suffix,
                    **processing_kwargs,
                )
            else:
                result = self.process_without_model(
                    image_path=image_path,
                    output_path=output_directory,
                    processing_function=processing_function,
                    suffix=suffix,
                    **processing_kwargs,
                )

            if result is True:
                results["successful"] += 1
            elif result is False:
                results["failed"] += 1
            elif result is None:
                results["no_detection"] += 1

        print(f"Batch processing complete: {results}")
        return results

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def _save_processed_image(
        self,
        processed_image: np.ndarray,
        original_path: str,
        output_path: str,
        suffix: str = "",
    ) -> bool:
        """Save processed image to output path.

        Args:
            processed_image: Processed image array
            original_path: Original image path (for filename)
            output_path: Output directory
            suffix: Optional suffix for filename

        Returns:
            True if successful, False otherwise
        """
        filename = strip_filename_from_path(original_path)
        os.makedirs(output_path, exist_ok=True)
        output_filename = f"{filename}{suffix}.png"
        output_full_path = os.path.join(output_path, output_filename)

        success = cv2.imwrite(output_full_path, processed_image)
        if success:
            print(f"Saved: {output_full_path}")
        else:
            print(f"Failed to save: {output_full_path}")
        return success

    # =============================================================================
    # PROPERTIES FOR EXTERNAL ACCESS
    # =============================================================================

    @property
    def confidence_threshold(self) -> float:
        """Get the current confidence threshold."""
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        """Set the confidence threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        self._confidence_threshold = value

    # =============================================================================
    # BACKWARD COMPATIBILITY ALIASES
    # =============================================================================

    def process_image(self, *args, **kwargs):
        """Backward compatibility alias - now uses auto-detection."""
        return self.process_auto(*args, **kwargs)
