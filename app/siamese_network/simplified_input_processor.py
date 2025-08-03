#!/usr/bin/env python3
"""
Simplified Siamese Network Image Processor

This module implements a simplified preprocessing pipeline with only 2 steps:
1. Face Alignment (using YOLO pose model)
2. Face Centering (using YOLO pose model) 

No segmentation step for testing purposes.
"""

import os
import sys
from typing import Optional
import numpy as np
import cv2

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from image_processor import ImageProcessor, ImageTransforms


class SimplifiedSiamesePreprocessingPipeline:
    """Simplified preprocessing pipeline with only face alignment and centering using YOLO pose."""

    def __init__(
        self,
        pose_model_path: str,
        target_size: int = 224,
        confidence_threshold: float = 0.5,
        segmentation_model_path: str = None,  # Not used, kept for compatibility
    ):
        """Initialize the simplified preprocessing pipeline.

        Args:
            pose_model_path: Path to YOLO pose model (.pt file) - REQUIRED
            target_size: Target size for face centering
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            segmentation_model_path: Not used in this simplified version
        """
        if not pose_model_path:
            raise ValueError("Pose model path is required")

        # We keep the processor for potential future use, but don't require it
        self.processor = None
        if segmentation_model_path:
            self.processor = ImageProcessor(
                model=segmentation_model_path, confidence_threshold=confidence_threshold
            )
        self.pose_model_path = pose_model_path
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold

        # Define simplified pipeline steps (no segmentation for now)
        self.pipeline_steps = [
            # Step 1: Face alignment using YOLO pose
            (
                ImageTransforms.align_face_landmarks,
                {
                    "face_detector_type": "yolo_pose",
                    "yolo_model_path": pose_model_path,
                    "confidence_threshold": confidence_threshold,
                },
            ),
            # Step 2: Face centering using YOLO pose
            (
                ImageTransforms.center_face_landmarks,
                {
                    "target_face_size": target_size,
                    "face_detector_type": "yolo_pose",
                    "yolo_model_path": pose_model_path,
                    "confidence_threshold": confidence_threshold,
                },
            ),
        ]

    def process_image(self, image_path: str) -> Optional[np.ndarray]:
        """Process a single image through the simplified preprocessing pipeline.

        Args:
            image_path: Path to input image

        Returns:
            Processed image or None if processing fails
        """
        try:
            print(f"🔄 Processing: {os.path.basename(image_path)}")

            # Load image
            import cv2

            current_image = cv2.imread(image_path)
            if current_image is None:
                print(f"❌ Could not load image: {image_path}")
                return None
            current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
            print(f"📏 Loaded image shape: {current_image.shape}")

            # Process through pipeline steps (all are plain image processing now)
            for step_idx, (processing_function, kwargs) in enumerate(
                self.pipeline_steps
            ):
                step_name = processing_function.__name__
                print(f"   Step {step_idx + 1}: {step_name}")

                # All steps are plain image processing (alignment, centering)
                processed_result = processing_function(current_image, **kwargs)

                if processed_result is not None:
                    current_image = processed_result
                    print(
                        f"      ✅ {step_name} successful, shape: {current_image.shape}"
                    )
                else:
                    print(f"      ❌ {step_name} failed - no landmarks detected")
                    return None

            print(f"✅ Processing completed! Final shape: {current_image.shape}")
            return current_image

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def process_batch(self, image_paths: list, output_dir: Optional[str] = None):
        """Process multiple images through the simplified pipeline.

        Args:
            image_paths: List of image file paths
            output_dir: Optional directory to save processed images

        Returns:
            List of (filename, processed_array) tuples for successful processing
        """
        results = []
        successful = 0

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"🚀 Processing {len(image_paths)} images through simplified pipeline...")

        for i, image_path in enumerate(image_paths):
            print(f"\n--- Image {i+1}/{len(image_paths)} ---")

            result = self.process_image(image_path)

            if result is not None:
                filename = os.path.basename(image_path)
                results.append((filename, result))
                successful += 1

                # Save result if output directory specified
                if output_dir:
                    # Convert to uint8 for saving if needed
                    if result.dtype == np.float32 or result.dtype == np.float64:
                        if result.min() >= 0 and result.max() <= 1:
                            save_image = (result * 255).astype(np.uint8)
                        else:
                            save_image = result.astype(np.uint8)
                    else:
                        save_image = result

                    output_path = os.path.join(output_dir, f"simplified_{filename}")
                    cv2.imwrite(
                        output_path, cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
                    )
                    print(f"💾 Saved: {output_path}")

        print(
            f"\n📊 Batch processing summary: {successful}/{len(image_paths)} successful"
        )
        return results
