"""
Examples demonstrating the enhanced ImageProcessor capabilities.

This module shows how to use ImageProcessor for various scenarios:
- Plain image processing (no model required)
- Model-based processing (with YOLO segmentation)
- Siamese network preprocessing
- Custom processing pipelines
- Background replacement
"""

import numpy as np
import cv2
from .processor import ImageProcessor
from .transforms import ImageTransforms


def example_siamese_preprocessing():
    """Example: Preprocess images for siamese network."""
    print("📸 Example: Siamese Network Preprocessing")

    # Create processor without model (plain processing only)
    processor = ImageProcessor()

    # Method 1: Direct siamese preprocessing
    def process_for_siamese(image_path, output_path):
        # This replicates the main.py preprocessing flow
        img = processor._load_image_array(image_path)
        processed = ImageTransforms.preprocess_for_siamese(img, target_size=224)
        # Note: processed is normalized [0,1], need to convert back for saving
        save_ready = (processed * 255).astype(np.uint8)
        processor._save_processed_image(save_ready, image_path, output_path, "_siamese")

    # Method 2: Step-by-step processing
    def process_step_by_step(image_path, output_path):
        # Load image
        img = processor._load_image_array(image_path)

        # Resize to 224x224 (siamese network input size)
        resized = ImageTransforms.resize_image(img, (224, 224))

        # Normalize to [0, 1]
        normalized = ImageTransforms.normalize_image(resized, scale=255.0)

        # Convert back for saving
        save_ready = (normalized * 255).astype(np.uint8)
        processor._save_processed_image(
            save_ready, image_path, output_path, "_step_by_step"
        )

    return process_for_siamese, process_step_by_step


def example_model_based_processing():
    """Example: Processing with YOLO model (detection + cropping)."""
    print("🎯 Example: Model-based Processing")

    # Note: Replace with actual model path
    # processor = ImageProcessor("path/to/your/model.pt")

    def detect_and_crop_then_preprocess(image_path, output_path):
        """Detect bat face, crop to square, then preprocess for siamese."""
        # This would work with actual model:
        #
        # # Step 1: Detect and crop to square
        # cropped = processor.detect_and_crop_square(image_path)
        # if cropped is None:
        #     print("No detection found")
        #     return False
        #
        # # Step 2: Preprocess cropped image for siamese
        # preprocessed = processor.preprocess_for_siamese(cropped, target_size=224)
        #
        # # Step 3: Save result
        # save_ready = (preprocessed * 255).astype(np.uint8)
        # return processor._save_processed_image(save_ready, image_path, output_path, "_detected_cropped")

        print("   (Requires actual YOLO model file)")
        return True

    return detect_and_crop_then_preprocess


def example_background_replacement():
    """Example: Background replacement using existing functionality."""
    print("🎨 Example: Background Replacement")

    # Note: Replace with actual model path
    # processor = ImageProcessor("path/to/segmentation/model.pt")

    def replace_background_with_blur(original_image, mask):
        """Custom background function - creates blurred background."""
        height, width = original_image.shape[:2]

        # Create random image and blur it
        random_img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        blurred_bg = cv2.GaussianBlur(random_img, (15, 15), 0)

        # Apply mask
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        result = blurred_bg.copy()
        result[mask_3d] = original_image[mask_3d]

        return result

    def process_with_background_replacement(image_path, output_path):
        """Process image with background replacement."""
        # This would work with actual model:
        # return processor.process_with_model(
        #     image_path=image_path,
        #     output_path=output_path,
        #     processing_function=replace_background_with_blur,
        #     suffix="_blur_bg"
        # )

        print("   (Requires actual YOLO segmentation model)")
        return True

    return process_with_background_replacement


def example_processing_pipeline():
    """Example: Complex processing pipeline."""
    print("🔄 Example: Processing Pipeline")

    # Create processor that can handle both model and plain operations
    # processor = ImageProcessor("path/to/model.pt")  # With model
    processor = ImageProcessor()  # Without model for this example

    def create_siamese_pipeline():
        """Create a pipeline for siamese network preprocessing."""

        # Define a function that uses parameters from the dict
        def custom_resize(
            image, target_width=224, target_height=224, interpolation="area"
        ):
            """Custom resize function that gets parameters from the pipeline dict."""
            print(
                f"   🔹 Resizing to {target_width}x{target_height} using {interpolation}"
            )
            return ImageTransforms.resize_image(
                image, (target_width, target_height), interpolation
            )

        # Define processing steps - AUTO-DETECTION based on parameter count!
        processing_steps = [
            # Step 1: Detect and crop (2 params = auto-detected as model-based)
            # (lambda img, mask: detect_and_crop_logic(img, mask), {}),
            # Step 2: Custom resize with parameters from dict (1 param = auto-detected as plain)
            (
                custom_resize,
                {
                    "target_width": 512,  # ← These parameters come from this dict
                    "target_height": 512,  # ← and get passed to custom_resize() as **kwargs
                    "interpolation": "cubic",
                },
            ),
            # Step 3: Normalize with lambda (1 param = auto-detected as plain)
            (lambda img: ImageTransforms.normalize_image(img, scale=255.0), {}),
        ]

        print("   ✨ AUTO-DETECTION Pipeline: detect → resize → normalize")
        print("   📋 Step 2 shows how dict parameters work:")
        print(
            "      custom_resize() gets: target_width=512, target_height=512, interpolation='cubic'"
        )
        print("   📋 Step 3 uses lambda with empty dict {} (no parameters)")

        # Note: This would work with actual model:
        # return processor.process_pipeline(
        #     image_path="input/image.jpg",
        #     output_path="output/",
        #     processing_steps=processing_steps,
        #     suffix="_pipeline"
        # )

        return processing_steps

    return create_siamese_pipeline


def example_batch_processing():
    """Example: Batch processing with different modes."""
    print("📦 Example: Batch Processing")

    processor = ImageProcessor()

    def batch_siamese_preprocessing():
        """Batch process images for siamese network."""

        # Define simple preprocessing function
        def siamese_preprocess(image_array):
            # Resize to 224x224
            resized = cv2.resize(image_array, (224, 224))
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            # Convert back for saving
            return (normalized * 255).astype(np.uint8)

        # Batch process with AUTO-DETECTION
        # results = processor.process_batch(
        #     input_directory="input/images/",
        #     output_directory="output/siamese_ready/",
        #     processing_function=siamese_preprocess,  # Auto-detected as plain (1 param)
        #     suffix="_siamese_ready"
        # )

        print("   ✨ AUTO-DETECTED batch processing: resize + normalize for siamese")
        return siamese_preprocess

    def batch_background_replacement():
        """Batch process with background replacement."""

        # This would work with segmentation model:
        # def bg_replacement(original_image, mask):
        #     # Your background replacement logic
        #     return processed_image
        #
        # results = processor.process_batch(
        #     input_directory="input/images/",
        #     output_directory="output/bg_replaced/",
        #     processing_function=bg_replacement,  # Auto-detected as model (2 params)
        #     suffix="_bg_replaced"
        # )

        print("   ✨ AUTO-DETECTED batch processing: background replacement with model")
        return True

    return batch_siamese_preprocessing, batch_background_replacement


def main():
    """Run all examples."""
    print("🚀 Enhanced ImageProcessor Examples\n")

    # Example 1: Siamese Network Preprocessing
    siamese_funcs = example_siamese_preprocessing()
    print("   ✅ Siamese preprocessing methods created\n")

    # Example 2: Model-based Processing
    model_func = example_model_based_processing()
    print("   ✅ Model-based processing method created\n")

    # Example 3: Background Replacement
    bg_func = example_background_replacement()
    print("   ✅ Background replacement method created\n")

    # Example 4: Processing Pipeline
    pipeline_func = example_processing_pipeline()
    print("   ✅ Processing pipeline created\n")

    # Example 5: Batch Processing
    batch_funcs = example_batch_processing()
    print("   ✅ Batch processing methods created\n")

    print("🎯 Key Capabilities:")
    print("   ✨ AUTO-DETECTION: 1 param = plain, 2+ params = model-based")
    print("   • Plain processing: No model needed")
    print("   • Model processing: YOLO detection/segmentation")
    print("   • Siamese preprocessing: Built-in resize + normalize")
    print("   • Custom pipelines: Chain multiple operations")
    print("   • Batch processing: Process entire directories")
    print("   • Backward compatibility: Existing code still works")

    print("\n💡 Usage:")
    print("   processor = ImageProcessor()  # Plain processing")
    print("   processor = ImageProcessor('model.pt')  # With model")
    print("   processor.preprocess_for_siamese(image)  # Direct siamese prep")
    print("   processor.detect_and_crop_square(image)  # Model-based crop")


if __name__ == "__main__":
    main()
