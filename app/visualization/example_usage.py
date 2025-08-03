#!/usr/bin/env python3
"""
Example usage script for saliency maps generation.

This script demonstrates how to use the saliency maps functionality with the available
data and models in the project. It provides ready-to-run examples using the existing
directory structure.
"""

import os
import sys
import subprocess
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    # Go up to the project root (app/visualization -> app -> project_root)
    return current_file.parent.parent.parent


def run_saliency_example():
    """Run saliency map generation with available data."""

    project_root = get_project_root()

    # Available models (choose one)
    available_models = [
        "face_rec_rousettus_#1/runs/model/siamesemodelv2_v80",
        "face_rec_rousettus_#1/runs/model/siamesemodelv2_v60",
        "face_rec_rousettus_#1/runs/model/siamesemodelv2_v40",
        "face_rec_rousettus_#1/runs/model/siamesemodelv2_v20",
        "face_rec_rousettus_#1/runs/model/siamesemodelv2_v10",
        "face_rec_rousettus_#1/runs/model/siamesemodelv2_v5",
    ]

    # Available input directories
    available_input_dirs = [
        "face_recognition-with_bg_31_03_25/data/unseen",
        "face_recognition-with_bg_31_03_25/data/validation",
        "face_recognition-no_background_05_07_24/data/unseen",
        "face_recognition-no_background_05_07_24/data/validation",
        "bats_images_24-05-25",
    ]

    print("=== Saliency Maps Example Usage ===\n")

    # Check what's available
    print("Checking available models...")
    valid_models = []
    for model_path in available_models:
        full_path = project_root / model_path
        if full_path.exists():
            valid_models.append(model_path)
            print(f"✓ Found: {model_path}")
        else:
            print(f"✗ Missing: {model_path}")

    print(f"\nChecking available input directories...")
    valid_input_dirs = []
    for input_dir in available_input_dirs:
        full_path = project_root / input_dir
        if full_path.exists():
            valid_input_dirs.append(input_dir)
            print(f"✓ Found: {input_dir}")
        else:
            print(f"✗ Missing: {input_dir}")

    if not valid_models:
        print("\n❌ No trained models found. Please train a model first.")
        return

    if not valid_input_dirs:
        print("\n❌ No input directories found. Please prepare your data first.")
        return

    # Use the first available model and input directory
    model_path = project_root / valid_models[0]
    input_dir = project_root / valid_input_dirs[0]

    print(f"\n🚀 Running saliency map generation with:")
    print(f"   Model: {model_path}")
    print(f"   Input: {input_dir}")

    # Create output directory
    output_dir = project_root / "saliency_output"
    output_dir.mkdir(exist_ok=True)

    # Run the saliency script
    script_path = Path(__file__).parent / "run_saliency_maps.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--input_dir",
        str(input_dir),
        "--model_path",
        str(model_path),
        "--output_dir",
        str(output_dir),
        "--sample_size",
        "10",  # Use a small sample for demo
    ]

    print(f"\nRunning command:")
    print(" ".join(cmd))
    print("\n" + "=" * 50)

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "=" * 50)
        print("✅ Saliency maps generated successfully!")
        print(f"📁 Output saved to: {output_dir}")

        # List output files
        output_files = list(output_dir.glob("*.pdf"))
        if output_files:
            print("\n📄 Generated files:")
            for file in output_files:
                print(f"   - {file.name}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running saliency maps: {e}")
        print("Please check the error messages above.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


def show_available_options():
    """Show available models and input directories."""

    project_root = get_project_root()

    print("=== Available Options ===\n")

    print("📊 Available Models:")
    model_base = project_root / "face_rec_rousettus_#1/runs/model"
    if model_base.exists():
        for model_dir in model_base.iterdir():
            if model_dir.is_dir():
                print(f"   - {model_dir.name}")
    else:
        print("   No models found in face_rec_rousettus_#1/runs/model")

    print("\n📁 Available Input Directories:")
    input_dirs = [
        "face_recognition-with_bg_31_03_25/data/unseen",
        "face_recognition-with_bg_31_03_25/data/validation",
        "face_recognition-no_background_05_07_24/data/unseen",
        "face_recognition-no_background_05_07_24/data/validation",
        "bats_images_24-05-25",
    ]

    for input_dir in input_dirs:
        full_path = project_root / input_dir
        if full_path.exists():
            # Count subdirectories
            subdirs = [d for d in full_path.iterdir() if d.is_dir()]
            print(f"   - {input_dir} ({len(subdirs)} subdirectories)")
        else:
            print(f"   - {input_dir} (not found)")


def main():
    """Main function to run examples."""

    print("Saliency Maps Example Script")
    print("=" * 40)

    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        show_available_options()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(
            """
Usage:
    python example_usage.py           # Run example with available data
    python example_usage.py --list    # Show available options
    python example_usage.py --help    # Show this help
    
This script will:
1. Check for available trained models
2. Check for available input directories
3. Run saliency map generation with a small sample
4. Save results to saliency_output directory
        """
        )
        return

    run_saliency_example()


if __name__ == "__main__":
    main()
