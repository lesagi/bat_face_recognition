#!/usr/bin/env python3
"""
Test script for the Simplified Siamese Preprocessing Pipeline

This script tests the simplified pipeline with only 2 steps:
1. Face Alignment (using YOLO pose model)
2. Face Centering (using YOLO pose model) 

No segmentation step for testing purposes.
"""

import os
import sys
from pathlib import Path


def test_simplified_pipeline():
    """Test the simplified preprocessing pipeline."""
    print("🔧 Testing Simplified Siamese Preprocessing Pipeline")
    print("=" * 60)

    # Add app directory to path
    app_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, app_dir)

    from siamese_preprocessing.input_processor import (
        SiamesePreprocessingPipeline,
    )
    from background_generation.background_generator import BackgroundGenerator

    # Configuration
    config = {
        "images_dir": "/Users/MAC/Documents/bat_face_rec/face_recognition-with_bg_31_03_25/data",
        "pose_model_path": "/Users/MAC/Documents/bat_face_rec/face_annotation_eyes_nose/runs/weights/best.pt",
        "segmentation_model_path": "/Users/MAC/Documents/bat_face_rec/face_segmentation_mauritius/chosen_model/best.pt",
        "output_dir": "/Users/MAC/Documents/bat_face_rec/face_recognition-with_bg_31_03_25/processed_data",
        "target_size": 224,
        "confidence_threshold": 0.5,
    }

    print("📋 Configuration:")
    print(f"   📁 Images: {config['images_dir']}")
    print(f"   🎯 Pose model: {config['pose_model_path']}")
    print(f"   🔍 Segmentation model: {config['segmentation_model_path']}")
    print(f"   📤 Output: {config['output_dir']}")
    print(f"   📏 Target size: {config['target_size']}")
    print(f"   🎚️  Confidence: {config['confidence_threshold']}")

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Check if models exist
    print(f"\n🔍 Checking models...")
    if not os.path.exists(config["pose_model_path"]):
        print(f"❌ Pose model not found: {config['pose_model_path']}")
        return
    else:
        print(f"✅ Pose model found")

    if not os.path.exists(config["segmentation_model_path"]):
        print(f"❌ Segmentation model not found: {config['segmentation_model_path']}")
        return
    else:
        print(f"✅ Segmentation model found")

    # Initialize pipeline
    print(f"\n🔧 Initializing SiamesePreprocessingPipeline...")
    try:
        pipeline = SiamesePreprocessingPipeline(
            pose_model_path=config["pose_model_path"],
            confidence_threshold=config["confidence_threshold"],
            model_path=config[
                "segmentation_model_path"
            ],
            background_generator=BackgroundGenerator.picsum,
            use_yolo_pose_alignment=True,
        )
        print("✅ Pipeline initialized successfully!")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Find test images
    print(f"\n🔍 Finding test images...")
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    test_images = []

    for root, dirs, files in os.walk(config["images_dir"]):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                test_images.append(os.path.join(root, file))
                if len(test_images) >= 5:  # Test with 5 images
                    break
        if len(test_images) >= 5:
            break

    if not test_images:
        print("❌ No test images found!")
        return

    print(f"✅ Found {len(test_images)} test images:")
    for i, img_path in enumerate(test_images):
        print(f"   {i+1}. {os.path.basename(img_path)}")

    # Test pipeline processing
    print(f"\n🚀 Testing simplified pipeline processing...")
    print("=" * 60)

    results = pipeline.preprocess_batch(test_images, config["output_dir"])

    # Summary
    print(f"\n" + "=" * 60)
    print(f"📊 SIMPLIFIED PIPELINE TEST SUMMARY")
    print(f"=" * 60)

    successful = len(results)
    total = len(test_images)

    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")

    if successful > 0:
        print(f"\n🎯 Successful Results:")
        for filename, result in results:
            print(f"   • {filename}: {result.shape} | dtype: {result.dtype}")
            if hasattr(result, "min"):
                print(f"     📈 Value range: [{result.min():.3f}, {result.max():.3f}]")

    if total - successful > 0:
        print(f"\n❌ Failed: {total - successful} images")
        print("   Check the processing logs above for details")

    print(f"\n📁 Results saved in: {config['output_dir']}")

    # Show output files
    try:
        output_files = [
            f for f in os.listdir(config["output_dir"]) if f.endswith(".jpg")
        ]
        if output_files:
            print(f"\n📷 Output files created:")
            for f in output_files:
                file_path = os.path.join(config["output_dir"], f)
                file_size = os.path.getsize(file_path)
                print(f"   • {f} ({file_size:,} bytes)")
    except Exception as e:
        print(f"⚠️  Could not list output files: {e}")


def test_single_image():
    """Test the pipeline on a single known good image."""
    print(f"\n" + "=" * 60)
    print(f"🧪 TESTING SINGLE IMAGE (Known Good)")
    print(f"=" * 60)

    # Add app directory to path
    app_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, app_dir)

    from siamese_preprocessing.input_processor import (
        SiamesePreprocessingPipeline,
    )
    from background_generation.background_generator import BackgroundGenerator

    # Single image test configuration
    single_config = {
        "test_image": "/Users/MAC/Documents/bat_face_rec/face_rec_rousettus_#1/images/original/babyis/IMG_20250518_151847.jpg",
        "pose_model_path": "/Users/MAC/Documents/bat_face_rec/face_annotation_eyes_nose/best.pt",
        "segmentation_model_path": "/Users/MAC/Documents/bat_face_rec/app/yolo_segmentation_trainer/rousesttus/training_results/runs/segment/bat_face_seg/weights/best.pt",
        "output_path": "/Users/MAC/Documents/bat_face_rec/single_test_result.jpg",
        "target_size": 224,
        "confidence_threshold": 0.3,
    }

    print(f"🖼️  Test image: {os.path.basename(single_config['test_image'])}")

    # Check if test image exists
    if not os.path.exists(single_config["test_image"]):
        print(f"❌ Test image not found: {single_config['test_image']}")
        return

    # Initialize pipeline
    try:
        pipeline = SiamesePreprocessingPipeline(
            pose_model_path=single_config["pose_model_path"],
            confidence_threshold=single_config["confidence_threshold"],
            model_path=single_config[
                "segmentation_model_path"
            ],  # Optional, not used
        )
        print("✅ Pipeline initialized for single image test")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return

    # Process single image
    print(f"\n🔄 Processing single image...")
    result = pipeline.process_image(single_config["test_image"])

    if result is not None:
        print(f"\n✅ Single image test SUCCESSFUL!")
        print(f"   📏 Final shape: {result.shape}")
        print(f"   📊 Data type: {result.dtype}")
        if hasattr(result, "min"):
            print(f"   📈 Value range: [{result.min():.3f}, {result.max():.3f}]")

        # Save result
        try:
            import cv2

            if result.dtype == np.float32 or result.dtype == np.float64:
                if result.min() >= 0 and result.max() <= 1:
                    save_image = (result * 255).astype(np.uint8)
                else:
                    save_image = result.astype(np.uint8)
            else:
                save_image = result

            cv2.imwrite(
                single_config["output_path"],
                cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR),
            )
            print(f"   💾 Saved: {single_config['output_path']}")
        except Exception as e:
            print(f"   ⚠️  Could not save result: {e}")
    else:
        print(f"❌ Single image test FAILED")


if __name__ == "__main__":
    print("🧪 SIMPLIFIED SIAMESE PREPROCESSING PIPELINE TEST")
    print("=" * 70)

    # Import numpy here for the single image test
    import numpy as np

    # Test 1: Batch processing
    test_simplified_pipeline()

    # Test 2: Single image processing
    test_single_image()

    print(f"\n🎉 Testing completed!")
    print("=" * 70)
