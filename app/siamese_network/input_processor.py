#!/usr/bin/env python3
"""
Siamese Network Advanced Image Processor

This module implements the complete advanced image processing pipeline for preparing images
for the siamese network model. This processor includes segmentation-based background 
replacement and cropping for optimal siamese network training.

Pipeline steps:
1. Face Alignment → 2. YOLO Segmentation → 3. Square Cropping → 4. Background Replacement → 5. Resize to Target → 6. Normalize to [0,1]

Features:
- YOLO-based bat face segmentation
- Configurable background replacement using BackgroundGenerator
- Smart square cropping around detected bat faces with buffer
- Multiple face alignment options: OpenCV, MediaPipe, or YOLO Pose
- YOLO Pose integration for accurate bat face landmark detection
- Standardized 224x224 output for siamese network
- Batch and single image processing support

Note: Both YOLO segmentation and YOLO pose models are REQUIRED for this pipeline.
The segmentation model is used for bat face detection and background replacement.
The pose model is used for accurate face alignment and landmark detection.

Usage:
    # Process single image
    python input_processor.py process-single --input image.jpg --output processed.jpg --segmentation-model face_segmentation.pt --pose-model face_pose.pt
    
    # Process batch of images
    python input_processor.py process-batch --input-dir /path/to/images/ --output-dir /path/to/processed/ --segmentation-model face_segmentation.pt --pose-model face_pose.pt
    
    # Get help
    python input_processor.py --help
    python input_processor.py process-single --help
    python input_processor.py process-batch --help
"""

import os
import sys

from typing import List, Optional, Tuple, Union
import numpy as np
import cv2
import click

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from image_processor import ImageTransforms
from background_generation.background_generator import BackgroundGenerator
from utils.image_utils import is_img_file
from config.loader import load_config


class SiamesePreprocessingPipeline:
    """Advanced preprocessing pipeline for siamese network with segmentation and pose-based processing.

    This pipeline uses YOLO segmentation to detect bat faces and YOLO pose for face alignment.
    It replaces backgrounds with generated content and performs smart cropping for optimal 
    siamese network performance.
    """

    def __init__(self):
        # Load configuration
        config = load_config()
        siamese_dp = config.siamese_network.training.get('data_preprocessing', {})
        
        # Margin ratio for cropping
        face_outer_margin_ratio = siamese_dp.get('face_outer_margin_ratio', 0.5)

        # Background generator type from config
        background_generators = {
            "blur": BackgroundGenerator.blur,
            "noise": BackgroundGenerator.noise,
            "gradient": BackgroundGenerator.gradient,
            "picsum": BackgroundGenerator.picsum,
            "solid_color": BackgroundGenerator.solid_color,
        }
        bg_type = siamese_dp.get('background', {}).get('type', 'blur')
        self.background_generator = background_generators.get(bg_type, BackgroundGenerator.blur)

        self.face_outer_margin_ratio = face_outer_margin_ratio
        self.target_size = siamese_dp.get('target_size', 224)
        self.normalize_scale = siamese_dp.get('scale_factor', 255.0)

    def preprocess_single_image(self, image_path: str, debug: bool = False, output_dir: str = None) -> Optional[np.ndarray]:
        try:
            # Load image
            image_array = cv2.imread(image_path)
            if image_array is None:
                print(f"❌ Failed to load image: {image_path}")
                return None
            
            current_image = image_array
            
            # Setup debug output and base filename
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            
            if debug:
                if not output_dir:
                    raise Exception("Debug flag is True but no debug directory provided. Please specify --debug-dir.")
                
                os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: Face alignment using YOLO pose landmarks
            # Align face using YOLO pose landmarks (model loaded from config automatically)
            aligned_image = ImageTransforms.align_face_landmarks(
                current_image,
                face_detector_type="yolo_pose",
                debug=debug,
                debug_dir=output_dir,
                base_filename=base_filename
            )
            
            if aligned_image is not None:
                current_image = aligned_image
            
            # Step 2: Get segmentation mask from aligned/original image
            # Get segmentation mask from aligned/original image
            mask = ImageTransforms.segment_image(
                current_image,
                debug=debug,
                debug_dir=output_dir,
                base_filename=base_filename
            )
            
            if mask is None:
                print(f"⚠️  Segmentation failed for: {image_path} - no bat face detected")
                return None
            
            # Step 3: Crop square around segmented region using the mask
            cropped_image = ImageTransforms.crop_square_around_segmentation_mask(
                current_image,
                mask,
                margin_ratio=self.face_outer_margin_ratio,
                debug=debug,
                debug_dir=output_dir,
                base_filename=base_filename
            )
            
            if cropped_image is None:
                print(f"⚠️  Cropping failed for: {image_path}")
                return None
            
            current_image = cropped_image

            # Step 4: Try to get mask for cropped image (resize first to help detection)
            # Resize cropped image to a size that works better for segmentation
            temp_size = 640
            h, w = current_image.shape[:2]
            if h != temp_size or w != temp_size:
                temp_image = cv2.resize(current_image, (temp_size, temp_size))
            else:
                temp_image = current_image
            
            cropped_mask = ImageTransforms.segment_image(
                temp_image,
                debug=debug,
                debug_dir=output_dir,
                base_filename=base_filename
            )
            
            if cropped_mask is not None:
                # Resize mask back to original cropped image size
                if h != temp_size or w != temp_size:
                    cropped_mask = cv2.resize(cropped_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    cropped_mask = (cropped_mask > 0).astype(np.uint8)
            else:
                print("⚠️ No mask found for cropped image")

            # Step 5: Replace background using the cropped mask (if available)
            # if debug:
            #     print("🔍 Step 5: Replacing background...")
            
            # if cropped_mask is not None:
            #     background_replaced = ImageTransforms.apply_background_replacement(
            #         current_image,
            #         cropped_mask,
            #         self.background_generator,
            #     )
                
            #     if background_replaced is not None:
            #         current_image = background_replaced
            #         if debug:
            #             print("🔍 Background replacement successful!")
                        
            #             # Save background replaced image
            #             debug_path = os.path.join(output_dir, f"{base_filename}_step5_background_replaced.jpg")
            #             cv2.imwrite(debug_path, background_replaced)
            #             print(f"🔍 Saved debug background replaced image: {debug_path}")
            #     else:
            #         print("⚠️ Background replacement failed, continuing without...")
            # else:
            #     print("⚠️ No mask available, skipping background replacement...")

            # Step 6: Resize to target size
            resized_image = ImageTransforms.resize_square_image(
                current_image, 
                self.target_size, 
                interpolation="bilinear",
                debug=debug,
                debug_dir=output_dir,
                base_filename=base_filename
            )
            
            if resized_image is None:
                print(f"⚠️  Resizing failed for: {image_path}")
                return None
            
            current_image = resized_image

            # Step 7: Normalize to [0,1] range
            normalized_image = ImageTransforms.normalize_image(
                current_image, 
                scale=self.normalize_scale,
                debug=debug,
                debug_dir=output_dir,
                base_filename=base_filename
            )
            
            if normalized_image is None:
                print(f"⚠️  Normalization failed for: {image_path}")
                return None
            
            return normalized_image

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def preprocess_batch(self, input_dir: str, output_dir: str, debug: bool = False, debug_dir: str = None):
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")

        results = []
        successful = 0
        total_files = 0
        failed_files = []

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
        print(f"🔧 Config: margin_ratio={self.face_outer_margin_ratio}, target_size={self.target_size}, normalize_scale={self.normalize_scale}")

        # Process all images recursively
        for dirpath, dirnames, filenames in os.walk(input_dir):
            # Create corresponding output directory structure
            relative_path = os.path.relpath(dirpath, input_dir)
            current_output_dir = (
                os.path.join(output_dir, relative_path)
                if relative_path != "."
                else output_dir
            )

            for filename in filenames:
                if is_img_file(filename):
                    image_path = os.path.join(dirpath, filename)
                    relative_image_path = os.path.relpath(image_path, input_dir)
                    print(f"\n📷 Processing: {relative_image_path}...")

                    # Create debug directory for this image if debug is enabled
                    if debug:
                        if debug_dir:
                            # Create subdirectory for each image to avoid conflicts
                            image_debug_dir = os.path.join(debug_dir, os.path.splitext(filename)[0])
                            os.makedirs(image_debug_dir, exist_ok=True)
                        else:
                            # Fallback to default behavior
                            image_debug_dir = os.path.join(output_dir, "debug_steps", os.path.splitext(filename)[0])
                            os.makedirs(image_debug_dir, exist_ok=True)
                    else:
                        image_debug_dir = None
                    
                    result = self.preprocess_single_image(image_path, debug=debug, output_dir=image_debug_dir)

                    if result is not None:
                        results.append((relative_image_path, result))
                        successful += 1

                        # Save processed image
                        os.makedirs(current_output_dir, exist_ok=True)
                        # Convert back to uint8 for saving
                        output_image = (result * 255).astype(np.uint8)
                        output_file = os.path.join(
                            current_output_dir, f"processed_{filename}"
                        )
                        success = cv2.imwrite(output_file, output_image)
                        if success:
                            print(f"✅ Saved: {output_file}")
                        else:
                            print(f"❌ Failed to save: {output_file}")
                    else:
                        failed_files.append(relative_image_path)
                        print(f"❌ Failed to process: {relative_image_path}")

        print(f"\n🎯 BATCH PROCESSING COMPLETE!")
        print(f"✅ Successfully processed: {successful}/{total_files} images")
        if successful < total_files:
            print(f"⚠️  Failed: {total_files - successful} images")
            if failed_files:
                print("Failed files:")
                for failed_file in failed_files[:10]:  # Show first 10 failed files
                    print(f"  - {failed_file}")
                if len(failed_files) > 10:
                    print(f"  ... and {len(failed_files) - 10} more")
        
        return results


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Siamese Network Advanced Image Processor
    
    Advanced preprocessing pipeline for siamese network with segmentation and pose-based processing.
    This pipeline uses YOLO segmentation to detect bat faces and YOLO pose for face alignment.
    """
    pass


@cli.command()
@click.option("--input", "-i", type=click.Path(exists=True, file_okay=True, dir_okay=False), 
              required=True, help="Single input image path")
@click.option("--output", "-o", type=click.Path(file_okay=True, dir_okay=False), 
              required=True, help="Output path for single image")
@click.option("--debug", "-d", is_flag=True, help="Enable debug output")
@click.option("--debug-dir", type=click.Path(file_okay=False, dir_okay=True), 
              help="Directory to save debug step images (optional when using --debug)")
def process_single(input, output, debug, debug_dir):
    """Process a single image through the advanced preprocessing pipeline."""
    
    # Set default debug directory if debug is enabled but no debug_dir provided
    if debug and not debug_dir:
        debug_dir = os.path.join(os.path.dirname(output), "debug_steps")
        print(f"🔧 Using default debug directory: {debug_dir}")
    
    # Initialize advanced pipeline
    pipeline = SiamesePreprocessingPipeline()
    
    # thresholds and background come from config
    
    try:
        print(f"🔄 Processing single image: {input}")
        result = pipeline.preprocess_single_image(input, debug=debug, output_dir=debug_dir)
        
        if result is not None:
            print("✅ Processing successful!")
            print(f"   Output shape: {result.shape}, dtype: {result.dtype}")
            print(f"   Value range: [{result.min():.3f}, {result.max():.3f}]")
            
            # Save processed image
            output_image = (result * 255).astype(np.uint8)
            success = cv2.imwrite(output, output_image)
            if success:
                print(f"💾 Saved processed image to: {output}")
            else:
                print(f"❌ Failed to save image to: {output}")
        else:
            print("❌ Processing failed - no bat face detected or processing error")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option("--input-dir", "-i", type=click.Path(exists=True, file_okay=False, dir_okay=True), 
              required=True, help="Input directory with images")
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, dir_okay=True), 
              required=True, help="Output directory for batch processing")
@click.option("--debug", "-d", is_flag=True, help="Enable debug output")
@click.option("--debug-dir", type=click.Path(file_okay=False, dir_okay=True), 
              help="Directory to save debug step images (optional when using --debug)")
def process_batch(input_dir, output_dir, debug, debug_dir):
    """Process multiple images through the advanced preprocessing pipeline."""
    
    # Set default debug directory if debug is enabled but no debug_dir provided
    if debug and not debug_dir:
        debug_dir = os.path.join(output_dir, "debug_steps")
        print(f"🔧 Using default debug directory: {debug_dir}")
    
    # Initialize advanced pipeline
    pipeline = SiamesePreprocessingPipeline()
    
    # thresholds and background come from config
    
    try:
        print(f"🔄 Processing batch: {input_dir}")
        results = pipeline.preprocess_batch(input_dir, output_dir, debug=debug, debug_dir=debug_dir)
        print(f"✅ Batch processing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
