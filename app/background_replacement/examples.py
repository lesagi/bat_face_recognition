"""
Comprehensive examples for using the background replacement module.

This file demonstrates various ways to use the background replacement functionality,
including model management, background replacement, and batch processing.
"""

import cv2
import os
from background_replacement import create_processor, create_batch_processor, load_model
from background_generation import BackgroundPresets, BackgroundGenerator


def example_basic_usage():
    """Basic usage example with different background types."""
    print("=== Basic Usage Example ===")

    # Example 1: Use explicit model path
    try:
        processor = create_processor(model="/path/to/your/model.pt")
        print("✓ Created processor with explicit model")
    except Exception as e:
        print(f"✗ Error creating processor: {e}")
        return

    # Apply different background types
    backgrounds = [
        ("white", BackgroundPresets.solid_color(color=(255, 255, 255))),
        ("black", BackgroundPresets.solid_color(color=(0, 0, 0))),
        ("noise", BackgroundPresets.noise(intensity=50)),
        ("gradient", BackgroundPresets.gradient()),
    ]

    for bg_name, bg_generator in backgrounds:
        try:
            success = processor.apply_background(
                image_path="sample_image.jpg",
                output_path="output/",
                background_generator=bg_generator,
                suffix=f"_{bg_name}",
            )
            if success:
                print(f"✓ Applied {bg_name} background")
            else:
                print(f"✗ Failed to apply {bg_name} background")
        except Exception as e:
            print(f"✗ Error with {bg_name} background: {e}")


def example_model_management():
    """Example of different ways to manage models."""
    print("\n=== Model Management Example ===")

    # Method 1: Load model first, then create processor
    try:
        model = load_model("/path/to/model.pt")
        processor = create_processor(model)
        print("✓ Created processor with pre-loaded model")
    except Exception as e:
        print(f"✗ Error with pre-loaded model: {e}")

    # Method 2: Pass model path directly
    try:
        processor = create_processor(model="/path/to/model.pt")
        print("✓ Created processor with model path")
    except Exception as e:
        print(f"✗ Error with model path: {e}")

    # Method 3: Use YOLO instance directly
    try:
        from ultralytics import YOLO

        yolo_model = YOLO("yolov8n-seg.pt")
        processor = create_processor(yolo_model)
        print("✓ Created processor with YOLO instance")
    except Exception as e:
        print(f"✗ Error with YOLO instance: {e}")


def example_model_validation():
    """Example showing model validation scenarios."""
    print("\n=== Model Validation Example ===")

    # Test different model file extensions
    test_models = [
        "/path/to/model.pt",  # Valid PyTorch model
        "/path/to/model.onnx",  # Valid ONNX model
        "/path/to/model.engine",  # Valid TensorRT model
        "/path/to/invalid.txt",  # Invalid extension
        "/nonexistent/path.pt",  # Non-existent file
    ]

    for model_path in test_models:
        try:
            processor = create_processor(model_path)
            print(f"✓ Valid model: {model_path}")
        except FileNotFoundError:
            print(f"✗ File not found: {model_path}")
        except ValueError as e:
            print(f"✗ Invalid model file {model_path}: {e}")
        except RuntimeError as e:
            print(f"✗ Model loading failed {model_path}: {e}")
        except Exception as e:
            print(f"✗ Unexpected error with {model_path}: {e}")


def example_confidence_threshold():
    """Example of working with confidence thresholds."""
    print("\n=== Confidence Threshold Example ===")

    try:
        # Create processor with custom confidence threshold
        processor = create_processor(
            model="/path/to/model.pt", confidence_threshold=0.7
        )
        print("✓ Created processor with confidence threshold 0.7")

        # Test different confidence values during creation
        confidence_levels = [0.3, 0.5, 0.7, 0.9]
        for conf in confidence_levels:
            try:
                test_processor = create_processor(
                    "/path/to/model.pt", confidence_threshold=conf
                )
                print(f"✓ Created processor with confidence threshold {conf}")
            except Exception as e:
                print(f"✗ Error with confidence {conf}: {e}")

        # Test invalid confidence threshold
        try:
            invalid_processor = create_processor(
                "/path/to/model.pt", confidence_threshold=1.5
            )
        except ValueError as e:
            print(f"✓ Correctly caught invalid threshold: {e}")

    except Exception as e:
        print(f"✗ Error in confidence threshold example: {e}")


def example_direct_segmentation():
    """Example of using segmentation directly without background replacement."""
    print("\n=== Direct Segmentation Example ===")

    try:
        processor = create_processor("/path/to/model.pt")

        # Load an image
        image = cv2.imread("sample_image.jpg")
        if image is None:
            print("✗ Could not load sample image")
            return

        # Perform segmentation
        result = processor.segment_image(image)

        if result is not None:
            original_image, mask = result
            print("✓ Bat detected and segmented")
            print(f"✓ Mask shape: {mask.shape}")
            print(f"✓ Mask covers {mask.sum()} pixels")

            # Save segmentation mask
            mask_image = (mask * 255).astype("uint8")
            cv2.imwrite("output/segmentation_mask.png", mask_image)
            print("✓ Saved segmentation mask")

        else:
            print("✗ No bat detected in image")

    except Exception as e:
        print(f"✗ Error in direct segmentation: {e}")


def example_batch_processing():
    """Example of batch processing multiple images."""
    print("\n=== Batch Processing Example ===")

    # Create batch processor with explicit model
    try:
        batch_processor = create_batch_processor(model="/path/to/model.pt")
        print("✓ Created batch processor with explicit model")
    except Exception as e:
        print(f"✗ Error creating batch processor with model: {e}")
        return

    # Method 2: Create batch processor with custom confidence threshold
    try:
        batch_processor = create_batch_processor(
            model="/path/to/model.pt", confidence_threshold=0.7
        )
        print("✓ Created batch processor with custom confidence threshold")
    except Exception as e:
        print(f"✗ Error creating batch processor with custom threshold: {e}")
        return

    # Process directory
    try:
        processed_count = batch_processor.process_directory(
            input_dir="input_images/",
            output_dir="output_images/",
            background_generator=BackgroundPresets.solid_color(),
            suffix="_processed",
        )
        print(f"✓ Processed {processed_count} images in directory")
    except Exception as e:
        print(f"✗ Error processing directory: {e}")

    # Process specific image list
    image_list = ["image1.jpg", "image2.jpg", "image3.jpg"]
    try:
        processed_count = batch_processor.process_image_list(
            image_paths=image_list,
            output_dir="output/",
            background_generator=BackgroundPresets.noise(),
            suffix="_noise",
        )
        print(f"✓ Processed {processed_count} images from list")
    except Exception as e:
        print(f"✗ Error processing image list: {e}")


def example_custom_backgrounds():
    """Example of creating custom background generators."""
    print("\n=== Custom Backgrounds Example ===")

    try:
        processor = create_processor("/path/to/model.pt")
    except Exception as e:
        print(f"✗ Error creating processor: {e}")
        return

    # Custom solid color background
    def custom_orange_background(height, width):
        return BackgroundGenerator.solid_color(height, width, (0, 165, 255))  # BGR

    # Custom gradient background
    def custom_gradient_background(height, width):
        return BackgroundGenerator.gradient(
            height,
            width,
            start_color=(50, 50, 50),
            end_color=(200, 200, 200),
            direction="horizontal",
        )

    # Custom noise background
    def custom_noise_background(height, width):
        return BackgroundGenerator.noise(height, width, intensity=100)

    # Apply custom backgrounds
    custom_backgrounds = [
        ("orange", custom_orange_background),
        ("custom_gradient", custom_gradient_background),
        ("custom_noise", custom_noise_background),
    ]

    for bg_name, bg_generator in custom_backgrounds:
        try:
            success = processor.apply_background(
                image_path="sample_image.jpg",
                output_path="output/",
                background_generator=bg_generator,
                suffix=f"_{bg_name}",
            )
            if success:
                print(f"✓ Applied {bg_name} background")
            else:
                print(f"✗ Failed to apply {bg_name} background")
        except Exception as e:
            print(f"✗ Error with {bg_name} background: {e}")


def example_advanced_usage():
    """Example of more advanced usage patterns."""
    print("\n=== Advanced Usage Examples ===")

    # Example of processing with different confidence levels
    confidence_levels = [0.3, 0.5, 0.7, 0.9]
    image_path = "/path/to/your/bat/image.jpg"
    output_dir = "/path/to/output/directory"

    if os.path.exists(image_path):
        for confidence in confidence_levels:
            print(f"Processing with confidence threshold: {confidence}")
            # Create processor with specific confidence for each test
            try:
                custom_processor = create_processor(
                    "/path/to/model.pt", confidence_threshold=confidence
                )
            except Exception as e:
                print(f"   Failed to create processor: {e}")
                continue

            success = custom_processor.apply_background(
                image_path,
                output_dir,
                BackgroundPresets.solid_color(),
                f"_conf_{confidence}",
            )
            if success:
                print(f"✓ Processed with confidence {confidence}")
            else:
                print(f"✗ Failed with confidence {confidence}")

    # Example of using different processors for different steps
    print("\nUsing different processors with different models...")
    if os.path.exists(image_path):
        # Use different processors for different steps
        try:
            processors = [
                create_processor(model="yolov8n-seg.pt", confidence_threshold=0.5),
                create_processor(model="yolov8n-seg.pt", confidence_threshold=0.7),
                create_processor(model="yolov8n-seg.pt", confidence_threshold=0.9),
            ]
        except Exception as e:
            print(f"Failed to create processors for different steps: {e}")
            return

        background_chain = [
            ("step1_green", BackgroundPresets.solid_color(color=(0, 255, 0))),
            ("step2_blur", BackgroundPresets.blur()),
            ("step3_gradient", BackgroundPresets.gradient()),
        ]

        for i, (step_name, bg_generator) in enumerate(background_chain):
            processor = processors[i % len(processors)]
            success = processor.apply_background(
                image_path, output_dir, bg_generator, f"_{step_name}"
            )
            print(
                f"{'✓' if success else '✗'} {step_name} (confidence: {processor.confidence_threshold})"
            )


def example_error_handling():
    """Example of comprehensive error handling."""
    print("\n=== Error Handling Example ===")

    # Test different error scenarios
    error_scenarios = [
        ("Invalid model path", lambda: create_processor("/invalid/path.pt")),
        ("Invalid file extension", lambda: create_processor("/path/to/model.txt")),
        (
            "Invalid confidence threshold",
            lambda: create_processor("/path/to/model.pt", confidence_threshold=2.0),
        ),
        (
            "No model parameter",
            lambda: create_batch_processor(),
        ),  # Missing required model parameter
    ]

    for scenario_name, scenario_func in error_scenarios:
        try:
            result = scenario_func()
            print(f"✗ {scenario_name}: Expected error but got result")
        except FileNotFoundError as e:
            print(f"✓ {scenario_name}: Caught FileNotFoundError - {e}")
        except ValueError as e:
            print(f"✓ {scenario_name}: Caught ValueError - {e}")
        except RuntimeError as e:
            print(f"✓ {scenario_name}: Caught RuntimeError - {e}")
        except Exception as e:
            print(f"? {scenario_name}: Caught unexpected error - {e}")


def run_all_examples():
    """Run all examples."""
    print("Running all segmentation examples...\n")

    # Create output directory
    os.makedirs("output", exist_ok=True)

    example_basic_usage()
    example_model_management()
    example_model_validation()
    example_confidence_threshold()
    example_direct_segmentation()
    example_batch_processing()
    example_custom_backgrounds()
    example_advanced_usage()
    example_error_handling()

    print("\n=== All Examples Complete ===")
    print(
        "Note: Some examples may show errors if sample images or models are not available."
    )
    print("This is expected and demonstrates the error handling capabilities.")


if __name__ == "__main__":
    run_all_examples()
