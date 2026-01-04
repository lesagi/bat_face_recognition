#!/usr/bin/env python3
"""
Run script for generating saliency maps using trained Siamese models.

This script loads a trained Siamese model and generates saliency maps for images
in a specified input directory. The input directory should contain subdirectories
with images organized by class/individual.

Usage:
    python run_saliency_maps.py --input_dir /path/to/input --model_path /path/to/model
    python run_saliency_maps.py --input_dir /path/to/input --model_path /path/to/model --method integrated_gradients
    python run_saliency_maps.py --input_dir /path/to/input --model_path /path/to/model --method comparison
"""

import argparse
import os
import sys
import time
import tensorflow as tf

# Add the app directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siamese_core.network import L1Dist
from visualization.saliency import (
    SiameseModelSaliencyMapCreator,
    MeanSaliencyMapCreator,
)


def detect_model_input_size(model):
    """
    Auto-detect the expected input size from the model architecture.

    Args:
        model: Loaded Keras model

    Returns:
        int: Expected input size (height/width, assuming square images)
    """
    try:
        # Get the input shape from the model
        input_shape = model.input_shape

        # Handle both single input and multiple inputs (Siamese networks have 2 inputs)
        if isinstance(input_shape, list):
            # Take the first input shape (both should be the same for Siamese)
            shape = input_shape[0]
        else:
            shape = input_shape

        # Extract height (assuming square images: [batch, height, width, channels])
        if len(shape) >= 3:
            height = shape[1]  # shape is typically (None, height, width, channels)
            width = shape[2]

            if height == width:
                return height
            else:
                print(
                    f"Warning: Non-square input detected ({height}x{width}), using height: {height}"
                )
                return height
        else:
            raise ValueError(f"Unexpected input shape: {shape}")

    except Exception as e:
        print(f"Warning: Could not auto-detect input size: {e}")
        print("Defaulting to 224x224. Use --input_size to specify manually.")
        return 224


def load_model(model_path):
    """
    Load a trained Siamese model from the specified path.

    Args:
        model_path (str): Path to the trained model directory

    Returns:
        tuple: (tf.keras.Model, int) - Loaded Siamese model and detected input size
    """
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "L1Dist": L1Dist,
                "BinaryCrossentropy": tf.losses.BinaryCrossentropy,
            },
        )
        print(f"Successfully loaded model from: {model_path}")

        # Auto-detect input size
        detected_size = detect_model_input_size(model)
        print(f"Detected model input size: {detected_size}x{detected_size}")

        return model, detected_size
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise


def validate_input_directory(input_dir):
    """
    Validate that the input directory exists and contains images organized by individual.

    Args:
        input_dir (str): Path to the input directory

    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return False

    # Check if directory contains subdirectories (nested structure)
    subdirs = [
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    ]

    if subdirs:
        print(f"Found {len(subdirs)} subdirectories (nested structure): {subdirs}")
        return True
    
    # Check for flat structure with double-dash naming
    all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    
    if image_files and any('--' in f for f in image_files):
        # Extract unique individual names
        individual_names = set()
        for filename in image_files:
            if '--' in filename:
                individual_name = filename.split('--')[0]
                individual_names.add(individual_name)
        
        print(f"Found {len(image_files)} image files in flat structure")
        print(f"Detected {len(individual_names)} unique individuals: {sorted(individual_names)}")
        return True
    
    # Check for any image files (fallback)
    if image_files:
        print(f"Found {len(image_files)} image files in flat structure")
        return True

    print(f"Error: No valid image files or subdirectories found in {input_dir}")
    print(
        "Input directory should contain either:\n"
        "1. Subdirectories with images organized by class/individual, or\n"
        "2. Image files with 'individual--image.jpg' naming convention"
    )
    return False


def generate_saliency_maps_from_config(
    model_path,
    input_dir,
    output_dir,
    method=None,
    sample_size=None,
    fast_mode=None,
    input_size=None,
    integration_steps=None,
    smoothing_samples=None,
    nesting=None,
    no_smoothing=False,
    max_images=None,
    base_method=None,
    use_mean_image_background=False,
):
    """
    Generate saliency maps using configuration from config.yml.
    
    Args:
        model_path (str): Path to trained Siamese model directory
        input_dir (str): Path to input directory containing images
        output_dir (str): Path to output directory for saliency maps
        method (str, optional): Saliency computation method. If None, reads from config.
        sample_size (int, optional): Number of images to sample. If None, reads from config.
        fast_mode (bool, optional): Use faster parameters. If None, reads from config.
        input_size (int, optional): Input image size. If None, auto-detects from model.
        integration_steps (int, optional): Integration steps. If None, reads from config.
        smoothing_samples (int, optional): Smoothing samples. If None, reads from config.
        nesting (str, optional): Nested subdirectory name
        no_smoothing (bool): Disable Gaussian smoothing
        max_images (int, optional): Max images for mean_saliency method
        base_method (str, optional): Base method for mean_saliency. If None, reads from config.
        use_mean_image_background (bool): Use mean image background for mean_saliency
    
    Returns:
        str: Path to output file
    """
    # Import config loader
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, app_dir)
    from app.config.loader import load_config
    
    cfg = load_config()
    saliency_config = cfg.siamese_network.saliency_maps
    
    # Get config values with fallback to function parameters
    method = method or saliency_config.get("method", "integrated_gradients")
    sample_size = sample_size if sample_size is not None else saliency_config.get("sample_size", 25)
    fast_mode = fast_mode if fast_mode is not None else saliency_config.get("fast_mode", False)
    integration_steps = integration_steps if integration_steps is not None else saliency_config.get("integration_steps")
    smoothing_samples = smoothing_samples if smoothing_samples is not None else saliency_config.get("smoothing_samples")
    base_method = base_method or saliency_config.get("base_method", "guided_gradients")
    use_mean_image_background = use_mean_image_background or saliency_config.get("use_mean_image_background", False)
    
    # Validate input directory
    if not validate_input_directory(input_dir):
        raise ValueError(f"Invalid input directory: {input_dir}")
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Load the model
    model, detected_input_size = load_model(model_path)
    
    # Use provided input_size or auto-detected size
    final_input_size = input_size if input_size else detected_input_size
    print(f"📏 Final input size: {final_input_size}x{final_input_size}")
    
    # Create saliency map generator
    print(f"Creating saliency maps for images in: {input_dir}")
    print(f"Using model: {model_path}")
    print(f"Method: {method}")
    print(f"Output directory: {output_dir}")
    if nesting:
        print(f"Nested subdirectory: {nesting}")
    if fast_mode:
        print("🚀 Fast mode enabled - using reduced parameters for speed")
    
    saliency_creator = SiameseModelSaliencyMapCreator(
        model=model,
        input_dir_path=input_dir,
        output_dir_path=output_dir,
        nesting=nesting,
        sample_size=sample_size,
        fast_mode=fast_mode,
        input_size=final_input_size,
        integration_steps=integration_steps,
        smoothing_samples=smoothing_samples,
    )
    
    # Show detected structure information
    if saliency_creator.subdirs:
        if any('--' in subdir for subdir in saliency_creator.subdirs):
            print(f"📁 Detected flat structure with {len(saliency_creator.subdirs)} unique individuals")
        else:
            print(f"📁 Detected nested structure with {len(saliency_creator.subdirs)} subdirectories")
    else:
        print(f"📁 Detected flat structure with {saliency_creator.num_images} image files")
    
    print(f"📊 Found {saliency_creator.num_images} total images")
    print(f"🎯 Will process {saliency_creator.actual_sample_size} images for saliency maps")
    
    # Generate saliency maps
    start_time = time.time()
    
    if method == "standard":
        print("Generating standard saliency maps...")
        saliency_creator.compute_saliency_map()
        print(f"Standard saliency maps generated successfully!")
        print(f"Output file: {saliency_creator.output_file_path}")
        output_file = saliency_creator.output_file_path
    elif method == "mean_saliency":
        print("Generating mean saliency map across all images...")
        
        # Create mean saliency creator
        mean_creator = MeanSaliencyMapCreator(
            model=model,
            input_dir_path=input_dir,
            output_dir_path=output_dir,
            nesting=nesting,
            input_size=final_input_size,
            integration_steps=integration_steps,
            smoothing_samples=smoothing_samples,
            fast_mode=fast_mode,
        )
        
        # Show detected structure information for mean saliency
        print(f"📊 Found {len(mean_creator.all_images)} total images for mean computation")
        if max_images:
            print(f"🎯 Will process up to {max_images} images for mean saliency")
        
        print(f"Using {base_method} as base method for mean computation")
        
        output_file = mean_creator.compute_mean_saliency_map(
            method=base_method,
            max_images=max_images,
            use_mean_image_background=use_mean_image_background,
        )
        print(f"Mean saliency map generated successfully!")
        print(f"Output file: {output_file}")
    else:
        print(f"Generating advanced saliency maps using {method}...")
        smoothing = not no_smoothing
        output_file = saliency_creator.compute_advanced_saliency_maps(
            method=method, smoothing=smoothing
        )
        print(f"Advanced saliency maps generated successfully!")
        print(f"Output file: {output_file}")
    
    elapsed_time = time.time() - start_time
    print(f"⏱️  Total processing time: {elapsed_time:.1f} seconds")
    
    return output_file


def main():
    """
    CLI entry point for run_saliency_maps (kept for backward compatibility).
    """
    parser = argparse.ArgumentParser(
        description="Generate saliency maps for Siamese network analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with nested directory structure
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model
    
    # Basic usage with flat structure (individual--image.jpg naming)
    python run_saliency_maps.py --input_dir /path/to/flat_data --model_path /path/to/model
    
    # Advanced method: Integrated Gradients (recommended for detailed analysis)
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --method integrated_gradients
    
    # Guided Gradients (good for localization)
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --method guided_gradients
    
    # Smoothed Gradients (reduces noise)
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --method smoothed_gradients
    
    # Comparison of all methods
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --method comparison
    
    # With custom output directory and no smoothing
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --method integrated_gradients --output_dir /path/to/output --no_smoothing
    
    # Force specific input size (overrides auto-detection)
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --method guided_gradients --input_size 105
    
    # Custom integration steps for more detailed Integrated Gradients
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --method integrated_gradients --integration_steps 50
    
    # Custom smoothing samples for cleaner Smoothed Gradients
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --method smoothed_gradients --smoothing_samples 20
    
    # Mean saliency across all images (shows global attention pattern)
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --method mean_saliency
    
    # Mean saliency with limited images for faster processing
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --method mean_saliency --max_images 50
    
    # Mean saliency using integrated gradients as base method
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --method mean_saliency --base_method integrated_gradients --integration_steps 30
    
    # Mean saliency with mean image background (shows saliency overlaid on averaged input images)
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --method mean_saliency --use_mean_image_background
    
    # Mean saliency with mean image background and custom parameters
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --method mean_saliency --use_mean_image_background --base_method integrated_gradients --max_images 100
    
    # With nested subdirectory structure
    python run_saliency_maps.py --input_dir /path/to/data --model_path /path/to/model --nesting validation --method integrated_gradients
    
    # With flat structure and nested subdirectories
    python run_saliency_maps.py --input_dir /path/to/flat_data --model_path /path/to/model --nesting validation --method integrated_gradients
        """,
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to input directory containing images organized by individual (supports both nested subdirectories and flat structure with 'individual--image.jpg' naming)",
    )

    parser.add_argument(
        "--model_path", required=True, help="Path to trained Siamese model directory"
    )

    parser.add_argument(
        "--output_dir",
        help="Path to output directory for saliency maps (default: same as input_dir)",
    )

    parser.add_argument(
        "--nesting",
        help="Nested subdirectory name within input_dir (e.g., 'validation', 'test')",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=25,
        help="Number of images to sample for saliency map generation (default: 25)",
    )

    parser.add_argument(
        "--method",
        choices=[
            "standard",
            "integrated_gradients",
            "guided_gradients",
            "smoothed_gradients",
            "comparison",
            "mean_saliency",
        ],
        default="standard",
        help="Saliency computation method (default: standard)",
    )

    parser.add_argument(
        "--no_smoothing",
        action="store_true",
        help="Disable Gaussian smoothing for advanced methods",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster parameters for quick testing (fewer steps/samples)",
    )

    parser.add_argument(
        "--input_size",
        type=int,
        help="Input image size for the model (e.g., 105, 224). Auto-detects from model if not specified.",
    )

    parser.add_argument(
        "--integration_steps",
        type=int,
        help="Number of steps for Integrated Gradients method (default: 20, or 10 in fast mode)",
    )

    parser.add_argument(
        "--smoothing_samples",
        type=int,
        help="Number of noise samples for Smoothed Gradients method (default: 10, or 5 in fast mode)",
    )

    parser.add_argument(
        "--max_images",
        type=int,
        help="Maximum number of images to process for mean_saliency method (default: all images)",
    )

    parser.add_argument(
        "--base_method",
        choices=[
            "standard",
            "integrated_gradients",
            "guided_gradients",
            "smoothed_gradients",
        ],
        default="guided_gradients",
        help="Base saliency method to use for mean_saliency computation (default: guided_gradients)",
    )

    parser.add_argument(
        "--use_mean_image_background",
        action="store_true",
        help="For mean_saliency method: compute and use mean of original images as background for saliency visualization",
    )

    args = parser.parse_args()

    # Validate arguments
    if not validate_input_directory(args.input_dir):
        return 1

    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        return 1

    # Load the model
    try:
        model, detected_input_size = load_model(args.model_path)

        # Use provided input_size or auto-detected size
        final_input_size = args.input_size if args.input_size else detected_input_size
        print(f"📏 Final input size: {final_input_size}x{final_input_size}")

    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1

    # Set output directory
    output_dir = args.output_dir or args.input_dir

    # Create saliency map generator
    try:
        print(f"Creating saliency maps for images in: {args.input_dir}")
        print(f"Using model: {args.model_path}")
        print(f"Method: {args.method}")
        print(f"Output directory: {output_dir}")
        if args.nesting:
            print(f"Nested subdirectory: {args.nesting}")
        if args.fast:
            print("🚀 Fast mode enabled - using reduced parameters for speed")

        saliency_creator = SiameseModelSaliencyMapCreator(
            model=model,
            input_dir_path=args.input_dir,
            output_dir_path=output_dir,
            nesting=args.nesting,
            sample_size=args.sample_size,
            fast_mode=args.fast,
            input_size=final_input_size,
            integration_steps=args.integration_steps,
            smoothing_samples=args.smoothing_samples,
        )

        # Show detected structure information
        if saliency_creator.subdirs:
            if any('--' in subdir for subdir in saliency_creator.subdirs):
                print(f"📁 Detected flat structure with {len(saliency_creator.subdirs)} unique individuals")
            else:
                print(f"📁 Detected nested structure with {len(saliency_creator.subdirs)} subdirectories")
        else:
            print(f"📁 Detected flat structure with {saliency_creator.num_images} image files")
        
        print(f"📊 Found {saliency_creator.num_images} total images")
        print(f"🎯 Will process {saliency_creator.actual_sample_size} images for saliency maps")

        # Generate saliency maps
        start_time = time.time()

        if args.method == "standard":
            print("Generating standard saliency maps...")
            saliency_creator.compute_saliency_map()
            print(f"Standard saliency maps generated successfully!")
            print(f"Output file: {saliency_creator.output_file_path}")
        elif args.method == "mean_saliency":
            print("Generating mean saliency map across all images...")

            # Create mean saliency creator
            mean_creator = MeanSaliencyMapCreator(
                model=model,
                input_dir_path=args.input_dir,
                output_dir_path=output_dir,
                nesting=args.nesting,
                input_size=final_input_size,
                integration_steps=args.integration_steps,
                smoothing_samples=args.smoothing_samples,
                fast_mode=args.fast,
            )

            # Show detected structure information for mean saliency
            print(f"📊 Found {len(mean_creator.all_images)} total images for mean computation")
            if args.max_images:
                print(f"🎯 Will process up to {args.max_images} images for mean saliency")

            # Determine which base method to use for mean computation
            base_method = args.base_method
            print(f"Using {base_method} as base method for mean computation")

            output_file = mean_creator.compute_mean_saliency_map(
                method=base_method,
                max_images=args.max_images,
                use_mean_image_background=args.use_mean_image_background,
            )
            print(f"Mean saliency map generated successfully!")
            print(f"Output file: {output_file}")
        else:
            print(f"Generating advanced saliency maps using {args.method}...")
            smoothing = not args.no_smoothing
            output_file = saliency_creator.compute_advanced_saliency_maps(
                method=args.method, smoothing=smoothing
            )
            print(f"Advanced saliency maps generated successfully!")
            print(f"Output file: {output_file}")

        elapsed_time = time.time() - start_time
        print(f"⏱️  Total processing time: {elapsed_time:.1f} seconds")

        # Provide method-specific insights
        if args.method == "integrated_gradients":
            print("\n🔍 Integrated Gradients Analysis:")
            print("   - Provides smooth, interpretable attribution")
            print("   - Shows gradual feature importance")
            print("   - Good for understanding model decisions")
            if args.integration_steps:
                print(f"   - Using custom {args.integration_steps} integration steps")

        elif args.method == "guided_gradients":
            print("\n🔍 Guided Gradients Analysis:")
            print("   - Focuses on positive contributions")
            print("   - Better localization of important features")
            print("   - Reduces noise in visualization")

        elif args.method == "smoothed_gradients":
            print("\n🔍 Smoothed Gradients Analysis:")
            print("   - Reduces noise through averaging")
            print("   - More stable attributions")
            print("   - Cleaner visualizations")
            if args.smoothing_samples:
                print(f"   - Using custom {args.smoothing_samples} smoothing samples")

        elif args.method == "comparison":
            print("\n🔍 Comparison Analysis:")
            print("   - Shows all methods side-by-side")
            print("   - Helps identify consistent patterns")
            print("   - Best for comprehensive analysis")
            if args.integration_steps or args.smoothing_samples:
                print("   - Using custom parameters for:")
                if args.integration_steps:
                    print(
                        f"     • Integrated Gradients: {args.integration_steps} steps"
                    )
                if args.smoothing_samples:
                    print(
                        f"     • Smoothed Gradients: {args.smoothing_samples} samples"
                    )

        elif args.method == "mean_saliency":
            print("\n🔍 Mean Saliency Analysis:")
            print("   - Computes average attention pattern across all images")
            print("   - Shows global model behavior patterns")
            print("   - Reveals consistent attention regions")
            print("   - Useful for understanding dataset-wide model focus")
            if args.max_images:
                print(f"   - Limited to {args.max_images} images for processing")
            if args.base_method != "guided_gradients":
                print(
                    f"   - Using {args.base_method} as base method for mean computation"
                )
                if args.integration_steps:
                    print(
                        f"   - Using custom {args.integration_steps} integration steps"
                    )
            if args.use_mean_image_background:
                print("   - Computing mean of original images as background")
                print(
                    "   - Overlaying saliency heatmap on mean image for better visualization"
                )
                print("   - Saving both mean image and saliency data separately")

    except Exception as e:
        print(f"Error generating saliency maps: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
