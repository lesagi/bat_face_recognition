#!/usr/bin/env python3
"""
Test the complete 6-step pipeline on 3 randomly selected images
with a different segmentation model.
"""

import os
import sys
import cv2
import numpy as np

# Add app directory to path
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)

from image_processor.transforms import ImageTransforms
from image_processor import ImageProcessor
from background_generation.background_generator import BackgroundGenerator


def test_pipeline_on_image(image_path: str, config: dict, image_index: int):
    """Test the complete 6-step pipeline on a single image."""

    image_name = os.path.basename(image_path)
    print(f"\n{'='*80}")
    print(f"🧪 TESTING IMAGE {image_index}: {image_name}")
    print(f"{'='*80}")

    try:
        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"❌ Failed to load image: {image_path}")
            return False

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        current_image = original_image.copy()
        print(f"📁 Loaded: {image_path}, {original_image.shape}")

        # Create output subdirectory for this image
        image_output_dir = os.path.join(
            config["output_dir"], f"image_{image_index}_{image_name.split('.')[0]}"
        )
        os.makedirs(image_output_dir, exist_ok=True)

        # Step 1: Face Alignment (try multiple confidence thresholds)
        print(f"\n🔄 Step 1: Face alignment using YOLO pose...")
        aligned_image = None

        # Try different confidence thresholds for pose detection (face recognition images work better)
        for confidence in [0.3, 0.2, 0.1, 0.4, 0.5]:
            print(f"   🔍 Trying pose confidence threshold: {confidence}")
            test_aligned = ImageTransforms.align_face_landmarks(
                original_image=current_image,
                face_detector_type="yolo_pose",
                yolo_model_path=config["pose_model_path"],
                confidence_threshold=confidence,
            )

            if test_aligned is not None:
                print(f"   ✅ Face alignment successful with confidence {confidence}")
                aligned_image = test_aligned
                break
            else:
                print(f"   ❌ Failed with confidence {confidence}")

        if aligned_image is None:
            print(
                f"❌ Face alignment failed with all confidence levels - continuing with original image"
            )
            aligned_image = current_image
        else:
            print(f"✅ Step 1 complete: {aligned_image.shape}")
            current_image = aligned_image

        # Save Step 1 result
        step1_output = os.path.join(image_output_dir, "step1_aligned.jpg")
        cv2.imwrite(step1_output, cv2.cvtColor(aligned_image, cv2.COLOR_RGB2BGR))
        print(f"💾 Alignment saved: {step1_output}")

        # Step 2: Segmentation
        print(f"\n🔄 Step 2: Running segmentation...")
        processor = ImageProcessor(model=config["segmentation_model_path"])
        segmentation_result = processor.segment_image(current_image)

        if segmentation_result is None:
            print(f"❌ Segmentation failed for {image_name}")
            return False

        segmented_image, mask = segmentation_result
        print(f"✅ Step 2 complete: mask has {np.sum(mask > 0)} points")

        # Save Step 2 result
        step2_output = os.path.join(image_output_dir, "step2_mask.jpg")
        cv2.imwrite(step2_output, (mask * 255).astype(np.uint8))
        print(f"💾 Mask saved: {step2_output}")

        # Step 3: Square Cropping
        print(f"\n🔄 Step 3: Cropping square around segmentation with 20% margin...")
        cropped = ImageTransforms.crop_square_around_segmentation_mask(
            original_image=current_image,
            segmentation_mask=mask,
            margin_ratio=0.2,
            debug=False,
        )

        if cropped is None:
            print(f"❌ Segmentation cropping failed for {image_name}")
            return False

        print(f"✅ Step 3 complete: {cropped.shape}")
        current_image = cropped

        # Save Step 3 result
        step3_output = os.path.join(image_output_dir, "step3_cropped.jpg")
        cv2.imwrite(step3_output, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        print(f"💾 Crop saved: {step3_output}")

        # Step 4: Background Replacement
        print(
            f"\n🔄 Step 4: Replacing background with Picsum using exact segmentation..."
        )
        try:
            cropped_mask = None

            # Try re-segmentation with multiple confidence levels
            print(f"🔄 Attempting re-segmentation on cropped image...")
            original_confidence = processor.confidence_threshold

            for confidence in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
                processor.confidence_threshold = confidence
                print(f"   🔍 Trying confidence threshold: {confidence}")

                cropped_segmentation_result = processor.segment_image(current_image)
                if cropped_segmentation_result is not None:
                    _, cropped_mask = cropped_segmentation_result
                    foreground_pixels = np.sum(cropped_mask > 0)
                    if foreground_pixels > 1000:
                        print(
                            f"   ✅ Re-segmentation successful with confidence {confidence}: {foreground_pixels} foreground pixels"
                        )
                        break
                    else:
                        cropped_mask = None
                else:
                    print(f"   ❌ Failed with confidence {confidence}")

            # Restore original confidence
            processor.confidence_threshold = original_confidence

            # Fallback: Map original mask to cropped coordinates
            if cropped_mask is None:
                print(
                    f"🔄 Re-segmentation failed, mapping original mask to cropped coordinates..."
                )

                mask_bbox_y, mask_bbox_x = np.where(mask > 0)
                if len(mask_bbox_y) > 0 and len(mask_bbox_x) > 0:
                    min_y, max_y = mask_bbox_y.min(), mask_bbox_y.max()
                    min_x, max_x = mask_bbox_x.min(), mask_bbox_x.max()

                    mask_height = max_y - min_y + 1
                    mask_width = max_x - min_x + 1
                    mask_center_y = (min_y + max_y) / 2
                    mask_center_x = (min_x + max_x) / 2

                    max_dim = max(mask_height, mask_width)
                    margin = int(max_dim * 0.2)
                    square_size = max_dim + 2 * margin

                    half_size = square_size // 2
                    crop_y1 = max(0, int(mask_center_y - half_size))
                    crop_x1 = max(0, int(mask_center_x - half_size))
                    crop_y2 = min(mask.shape[0], crop_y1 + square_size)
                    crop_x2 = min(mask.shape[1], crop_x1 + square_size)

                    if crop_y2 - crop_y1 < square_size:
                        crop_y1 = max(0, crop_y2 - square_size)
                    if crop_x2 - crop_x1 < square_size:
                        crop_x1 = max(0, crop_x2 - square_size)

                    cropped_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]

                    foreground_pixels = np.sum(cropped_mask > 0)
                    print(
                        f"   ✅ Mapped original mask: {foreground_pixels} foreground pixels"
                    )

                    if cropped_mask.shape[:2] != current_image.shape[:2]:
                        print(f"   ⚠️  Resizing mask to match image...")
                        cropped_mask = cv2.resize(
                            cropped_mask.astype(np.uint8),
                            (current_image.shape[1], current_image.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )

            # Apply background replacement
            if cropped_mask is not None and np.sum(cropped_mask > 0) > 0:
                foreground_pixels = np.sum(cropped_mask > 0)
                total_pixels = cropped_mask.size
                print(
                    f"✅ Using exact segmentation mask: {foreground_pixels}/{total_pixels} pixels ({100*foreground_pixels/total_pixels:.1f}% bat)"
                )

                # Save the mask
                mask_output = os.path.join(
                    image_output_dir, "step4_exact_segmentation_mask.jpg"
                )
                cv2.imwrite(mask_output, (cropped_mask * 255).astype(np.uint8))
                print(f"💾 Exact segmentation mask saved: {mask_output}")

                bg_replaced = ImageTransforms.apply_background_replacement(
                    original_image=current_image,
                    mask=cropped_mask,
                    background_source=BackgroundGenerator.picsum,
                )

                if bg_replaced is not None:
                    print(
                        f"✅ Step 4 complete with exact segmentation mask: {bg_replaced.shape}"
                    )
                    current_image = bg_replaced

                    # Save Step 4 result
                    step4_output = os.path.join(
                        image_output_dir, "step4_background_replaced.jpg"
                    )
                    cv2.imwrite(
                        step4_output, cv2.cvtColor(bg_replaced, cv2.COLOR_RGB2BGR)
                    )
                    print(f"💾 Background replacement saved: {step4_output}")
                else:
                    print(f"❌ Background replacement failed")
            else:
                print(f"❌ No valid segmentation mask found")

        except Exception as e:
            print(f"❌ Background replacement error: {e}")

        # Step 5: Resize
        print(f"\n🔄 Step 5: Resizing to 224x224...")
        resized = ImageTransforms.resize_square_image(
            original_image=current_image, target_size=224, interpolation="bilinear"
        )

        print(f"✅ Step 5 complete: {resized.shape}")
        current_image = resized

        # Save Step 5 result
        step5_output = os.path.join(image_output_dir, "step5_resized.jpg")
        cv2.imwrite(step5_output, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        print(f"💾 Resize saved: {step5_output}")

        # Step 6: Normalize
        print(f"\n🔄 Step 6: Normalizing to [0,1] range...")
        normalized = ImageTransforms.normalize_image(
            original_image=current_image, scale=255.0
        )

        print(f"✅ Step 6 complete: {normalized.shape}")
        print(f"   📊 Value range: [{normalized.min():.3f}, {normalized.max():.3f}]")

        # Save Step 6 result (convert back to uint8)
        final_uint8 = (normalized * 255).astype(np.uint8)
        step6_output = os.path.join(image_output_dir, "step6_final_normalized.jpg")
        cv2.imwrite(step6_output, cv2.cvtColor(final_uint8, cv2.COLOR_RGB2BGR))
        print(f"💾 Final result saved: {step6_output}")

        print(f"\n🎉 Pipeline successful for {image_name}!")
        print(f"   📏 Original: {original_image.shape}")
        print(f"   🔄 Aligned: {aligned_image.shape}")
        print(f"   📦 Cropped: {cropped.shape}")
        print(f"   📏 Resized: {resized.shape}")
        print(
            f"   🎯 Final: {normalized.shape}, range: [{normalized.min():.3f}, {normalized.max():.3f}]"
        )
        print(f"   📁 Results saved in: {image_output_dir}")

        return True

    except Exception as e:
        print(f"❌ Pipeline failed for {image_name}: {e}")
        return False


def main():
    """Test the pipeline on 3 randomly selected images."""

    # Configuration
    config = {
        "pose_model_path": "/Users/MAC/Documents/bat_face_rec/face_annotation_eyes_nose/runs/weights/best.pt",
        #"segmentation_model_path": "/Users/MAC/Documents/bat_face_rec/face_segmentation_mauritius/chosen_model/best.pt", #Mauritius model
        "segmentation_model_path": "/Users/MAC/Documents/bat_face_rec/face_rec_rousettus_#1/runs/model/siamesemodelv2_v80", #Rousettus model
        "images_dir": "",
        "output_dir": "/Users/MAC/Documents/bat_face_rec/pipeline_test_face_recognition_image_output",
    }

    # Selected image for testing (face recognition dataset)
    selected_images = [
        "/Users/MAC/Documents/bat_face_rec/face_recognition-original_background_20_06_24/data/training/20230831_143044/20230831_143044.1382.jpg",
        "/Users/MAC/Documents/bat_face_rec/face_recognition-original_background_20_06_24/data/training/20230828_185848/20230828_185848.1218.png",
        "/Users/MAC/Documents/bat_face_rec/face_recognition-original_background_20_06_24/data/training/20230831_155428/20230831_155428.1103.jpg",
        "/Users/MAC/Documents/bat_face_rec/face_rec_rousettus_#1/images/original/romi/IMG_20250518_152544.jpg"
        ]

    print(f"🧪 TESTING 6-STEP PIPELINE ON FACE RECOGNITION IMAGE")
    print(f"{'='*80}")
    print(f"🎯 Pose model: {config['pose_model_path']}")
    print(f"🔍 Segmentation model: {config['segmentation_model_path']}")
    print(f"📁 Images directory: {config['images_dir']}")
    print(f"📤 Output directory: {config['output_dir']}")
    print(f"🎯 Target images: {selected_images}")

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Test each image
    successful = 0
    failed = 0

    for i, image_name in enumerate(selected_images, 1):
        image_path = os.path.join(config["images_dir"], image_name)

        if os.path.exists(image_path):
            success = test_pipeline_on_image(image_path, config, i)
            if success:
                successful += 1
            else:
                failed += 1
        else:
            print(f"❌ Image not found: {image_path}")
            failed += 1

    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 PIPELINE TESTING COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Successful: {successful}/{len(selected_images)}")
    print(f"❌ Failed: {failed}/{len(selected_images)}")
    print(f"📁 Results saved in: {config['output_dir']}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
