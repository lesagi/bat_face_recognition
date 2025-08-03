#!/usr/bin/env python3
"""
Siamese Network Advanced Image Processor

This module implements the complete advanced image processing pipeline for preparing images
for the siamese network model. This processor includes segmentation-based background 
replacement and cropping for optimal siamese network training.

Pipeline steps:
1. YOLO Segmentation → 2. Background Replacement → 3. Square Cropping → 4. Face Alignment → 5. Face Centering → 6. Normalize to [0,1]

Features:
- YOLO-based bat face segmentation
- Configurable background replacement using BackgroundGenerator
- Smart square cropping around detected bat faces with buffer
- Multiple face alignment options: OpenCV, MediaPipe, or YOLO Pose
- YOLO Pose integration for accurate bat face landmark detection
- Standardized 224x224 output for siamese network
- Batch and single image processing support

Note: A YOLO segmentation model is REQUIRED for this pipeline.
Optional: A YOLO pose model can be used for more accurate face alignment.

Usage:
    # Basic usage with OpenCV face alignment
    python input_processor.py --input image.jpg --output processed.jpg --model face_segmentation.pt
    
    # With YOLO pose alignment for better accuracy
    python input_processor.py --input image.jpg --output processed.jpg --model face_segmentation.pt --pose_model face_pose.pt --use_yolo_pose
    
    # Batch processing with YOLO pose alignment
    python input_processor.py --input_dir /path/to/images/ --output_dir /path/to/processed/ --model face_segmentation.pt --pose_model face_pose.pt --use_yolo_pose
"""

import argparse
import os
import sys

from typing import List, Optional, Tuple, Union
import numpy as np
import cv2

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from image_processor import ImageProcessor, ImageTransforms
from background_generation.background_generator import BackgroundGenerator
from utils.image_utils import is_img_file


class SiamesePreprocessingPipeline:
    """Advanced preprocessing pipeline for siamese network with segmentation-based processing.

    This pipeline uses YOLO segmentation to detect bat faces, replaces backgrounds with
    generated content, and performs smart cropping for optimal siamese network performance.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        background_generator: callable = BackgroundGenerator.blur,
        pose_model_path: Optional[str] = None,
        use_yolo_pose_alignment: bool = False,
    ):
        """Initialize the advanced preprocessing pipeline.

        Args:
            model_path: Path to YOLO segmentation model (.pt file) - REQUIRED
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            background_generator: Background generator function from BackgroundGenerator class
            pose_model_path: Optional path to YOLO pose model (.pt file) for face alignment
            use_yolo_pose_alignment: If True, use YOLO pose model for face alignment instead of OpenCV
        """
        if not model_path:
            raise ValueError(
                "Model path is required for advanced siamese preprocessing pipeline"
            )

        self.processor = ImageProcessor(
            model=model_path, confidence_threshold=confidence_threshold
        )
        self.target_size = 224  # Standard input size for siamese network
        self.background_generator = background_generator
        self.pose_model_path = pose_model_path
        self.use_yolo_pose_alignment = use_yolo_pose_alignment

        # Validate pose model path if YOLO pose alignment is requested
        if use_yolo_pose_alignment and not pose_model_path:
            raise ValueError(
                "pose_model_path is required when use_yolo_pose_alignment=True"
            )

        # Determine face detector type and parameters for alignment
        if use_yolo_pose_alignment:
            face_detector_type = "yolo_pose"
            alignment_kwargs = {
                "face_detector_type": face_detector_type,
                "yolo_model_path": pose_model_path,
            }
            centering_kwargs = {
                "target_face_size": self.target_size,
                "face_detector_type": face_detector_type,  # Use same detector for centering
                "yolo_model_path": (
                    pose_model_path if face_detector_type == "yolo_pose" else None
                ),
            }
        else:
            face_detector_type = "opencv"
            alignment_kwargs = {"face_detector_type": face_detector_type}
            centering_kwargs = {
                "target_face_size": self.target_size,
                "face_detector_type": face_detector_type,
            }

        # Define advanced siamese pipeline with segmentation and background replacement
        self.advanced_pipeline_steps = [
            # Step 1: Replace background with generated background (requires segmentation mask)
            (
                ImageTransforms.apply_background_replacement,
                {"background_source": self.background_generator},
            ),
            # Step 2: Crop square around segmented region with buffer (requires segmentation mask)
            (ImageTransforms.crop_square_around_segmentation, {"buffer_factor": 1.5}),
            # Step 3: Face alignment (rotate to make eyes horizontal) - plain transform
            (ImageTransforms.align_face_landmarks, alignment_kwargs),
            # Step 4: Center face based on landmarks for consistent positioning - plain transform
            (ImageTransforms.center_face_landmarks, centering_kwargs),
            # Step 5: Normalize to [0,1] range - plain transform
            (ImageTransforms.normalize_image, {"scale": 255.0}),
        ]

    def preprocess_single_image(self, image_path: str) -> Optional[np.ndarray]:
        """Process a single image through the complete advanced preprocessing pipeline.

        Args:
            image_path: Path to input image

        Returns:
            Preprocessed image ready for siamese network (224x224, normalized [0,1])
            Returns None if processing fails or no bat face detected
        """
        try:
            # Load image
            current_image = self.processor._load_image_array(image_path)

            # Process through advanced pipeline steps
            for step_idx, (processing_function, kwargs) in enumerate(
                self.advanced_pipeline_steps
            ):
                processing_type = self.processor._detect_processing_type(
                    processing_function
                )

                if processing_type == "model":
                    # Get segmentation mask (model is guaranteed to be loaded)
                    segmentation_result = self.processor.segment_image(current_image)
                    if segmentation_result is None:
                        print(f"⚠️  No bat face detected in: {image_path}")
                        return None  # No object detected
                    current_image, mask = segmentation_result
                    processed_result = processing_function(
                        current_image, mask, **kwargs
                    )
                    if processed_result is not None:
                        current_image = processed_result
                    # If processing_function returns None, keep current_image unchanged
                else:
                    # Plain processing (resize, normalize, etc.)
                    processed_result = processing_function(current_image, **kwargs)

                    # Special handling for face alignment and centering - REQUIRED for processing
                    if (
                        processing_function.__name__
                        in ["align_face_landmarks", "center_face_landmarks"]
                    ) and processed_result is None:
                        print(
                            f"❌ {processing_function.__name__} failed for {image_path} - skipping image (face landmarks required)"
                        )
                        return None  # Skip this image entirely
                    elif processed_result is not None:
                        current_image = processed_result
                    else:
                        # For other functions that return None, this is an error
                        print(
                            f"❌ Processing step {processing_function.__name__} failed for {image_path}"
                        )
                        return None

            return current_image

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None

    def preprocess_batch(self, input_dir: str, output_dir: Optional[str] = None):
        """Process multiple images through the advanced pipeline with recursive directory support.

        Args:
            input_dir: Directory containing input images (supports nested directories)
            output_dir: Optional directory to save processed images

        Returns:
            List of (relative_path, processed_array) tuples for successful processing
        """
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")

        results = []
        successful = 0
        total_files = 0

        # Count total image files first
        for dirpath, dirnames, filenames in os.walk(input_dir):
            for filename in filenames:
                if is_img_file(filename):
                    total_files += 1

        if total_files == 0:
            print(f"⚠️  No image files found in {input_dir}")
            return []

        print(
            f"🔄 Processing {total_files} images with advanced pipeline using {self.background_generator.__name__} backgrounds..."
        )

        # Process all images recursively
        for dirpath, dirnames, filenames in os.walk(input_dir):
            # Create corresponding output directory structure
            relative_path = os.path.relpath(dirpath, input_dir)
            current_output_dir = (
                os.path.join(output_dir, relative_path)
                if output_dir and relative_path != "."
                else output_dir
            )

            for filename in filenames:
                if is_img_file(filename):
                    image_path = os.path.join(dirpath, filename)
                    relative_image_path = os.path.relpath(image_path, input_dir)
                    print(f"Processing: {relative_image_path}...")

                    result = self.preprocess_single_image(image_path)

                    if result is not None:
                        results.append((relative_image_path, result))
                        successful += 1

                        # Save processed image if output directory specified
                        if current_output_dir:
                            os.makedirs(current_output_dir, exist_ok=True)
                            # Convert back to uint8 for saving
                            output_image = (result * 255).astype(np.uint8)
                            output_file = os.path.join(
                                current_output_dir, f"processed_{filename}"
                            )
                            cv2.imwrite(output_file, output_image)

        print(f"✅ Successfully processed {successful}/{total_files} images")
        if successful < total_files:
            print(
                f"⚠️  {total_files - successful} images failed (likely no bat faces detected)"
            )
        return results


def main():
    """Main command-line interface for advanced siamese network image processing."""
    parser = argparse.ArgumentParser(
        description="Siamese Network Advanced Image Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image with blur background
  python image_adjuster_processor.py --input image.jpg --output processed.jpg --model face_segmentation.pt
  
  # Process directory with gradient backgrounds  
  python image_adjuster_processor.py --input_dir /path/to/images/ --output_dir /path/to/processed/ --model face_segmentation.pt --background gradient
        """,
    )

    # Input options
    parser.add_argument("--input", type=str, help="Single input image path")
    parser.add_argument("--input_dir", type=str, help="Input directory with images")
    # Output options
    parser.add_argument("--output", type=str, help="Output path for single image")
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for batch processing"
    )

    # Model options (required)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO segmentation model (.pt file) - REQUIRED",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Detection confidence threshold"
    )

    # Background options
    parser.add_argument(
        "--background",
        type=str,
        default="blur",
        choices=["blur", "noise", "gradient", "picsum", "solid_color"],
        help="Background generator type",
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.input, args.input_dir]):
        parser.error("Must specify either --input or --input_dir")

    # Select background generator
    background_generators = {
        "blur": BackgroundGenerator.blur,
        "noise": BackgroundGenerator.noise,
        "gradient": BackgroundGenerator.gradient,
        "picsum": BackgroundGenerator.picsum,
        "solid_color": BackgroundGenerator.solid_color,
    }

    background_gen = background_generators[args.background]

    # Initialize advanced pipeline
    pipeline = SiamesePreprocessingPipeline(
        model_path=args.model,
        confidence_threshold=args.confidence,
        background_generator=background_gen,
    )

    try:
        if args.input:
            # Single image processing
            print(f"🔄 Processing single image: {args.input}")
            result = pipeline.preprocess_single_image(args.input)

            if result is not None:
                print("✅ Processing successful!")
                print(f"   Output shape: {result.shape}, dtype: {result.dtype}")
                print(f"   Value range: [{result.min():.3f}, {result.max():.3f}]")

                if args.output:
                    # Save processed image
                    output_image = (result * 255).astype(np.uint8)
                    cv2.imwrite(args.output, output_image)
                    print(f"💾 Saved processed image to: {args.output}")
                else:
                    print("💡 Use --output to save the processed image")
            else:
                print("❌ Processing failed - no bat face detected or processing error")

        elif args.input_dir:
            # Batch processing
            print(f"🔄 Processing batch: {args.input_dir}")
            results = pipeline.preprocess_batch(args.input_dir, args.output_dir)
            print(f"✅ Batch processing complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
