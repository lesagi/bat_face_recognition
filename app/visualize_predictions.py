#!/usr/bin/env python3
"""
Standalone script to visualize segmentation and pose predictions on images.

Usage:
    # Process single image
    python visualize_predictions.py --input image.jpg --output output_dir/
    
    # Process directory of images
    python visualize_predictions.py --input input_dir/ --output output_dir/
    
    # Enable debug output
    python visualize_predictions.py --input input_dir/ --output output_dir/ --debug
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add app directory to path for imports
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from image_processor.transforms import ImageTransforms
from utils.image_utils import is_img_file


def process_single_image(input_path: str, output_dir: str, debug: bool = False) -> bool:
    """Process a single image and save the annotated result."""
    try:
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            print(f"❌ Failed to load image: {input_path}")
            return False
        
        if debug:
            print(f"🔄 Processing: {os.path.basename(input_path)}")
        
        # Run the prediction visualization
        annotated = ImageTransforms.draw_all_predictions(image, debug=debug)
        
        if annotated is None:
            print(f"❌ Failed to process: {input_path}")
            return False
        
        # Generate output filename
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{input_name}_annotated.jpg")
        
        # Save annotated image
        cv2.imwrite(output_path, annotated)
        
        if debug:
            print(f"✅ Saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {input_path}: {e}")
        return False


def process_directory(input_dir: str, output_dir: str, debug: bool = False) -> None:
    """Process all images in a directory recursively."""
    if not os.path.exists(input_dir):
        print(f"❌ Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Count total image files
    total_files = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if is_img_file(file):
                total_files += 1
    
    if total_files == 0:
        print(f"⚠️  No image files found in {input_dir}")
        return
    
    print(f"🔄 Processing {total_files} images...")
    
    successful = 0
    
    # Process all images recursively
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding output subdirectory
        relative_path = os.path.relpath(root, input_dir)
        current_output_dir = os.path.join(output_dir, relative_path) if relative_path != "." else output_dir
        os.makedirs(current_output_dir, exist_ok=True)
        
        for file in files:
            if is_img_file(file):
                input_path = os.path.join(root, file)
                if process_single_image(input_path, current_output_dir, debug):
                    successful += 1
    
    print(f"✅ Successfully processed {successful}/{total_files} images")
    print(f"📁 Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize segmentation and pose predictions on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Input image file or directory"
    )
    
    parser.add_argument(
        "--output", "-o", 
        required=True,
        help="Output directory for annotated images"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    input_path = args.input
    output_dir = args.output
    
    if not os.path.exists(input_path):
        print(f"❌ Input path does not exist: {input_path}")
        sys.exit(1)
    
    if os.path.isfile(input_path):
        # Process single image
        print(f"🖼️  Processing single image: {input_path}")
        os.makedirs(output_dir, exist_ok=True)
        success = process_single_image(input_path, output_dir, args.debug)
        if success:
            print(f"✅ Image processed successfully")
            print(f"📁 Result saved to: {output_dir}")
        else:
            print("❌ Failed to process image")
            sys.exit(1)
    
    elif os.path.isdir(input_path):
        # Process directory
        print(f"📁 Processing directory: {input_path}")
        process_directory(input_path, output_dir, args.debug)
    
    else:
        print(f"❌ Input path is neither a file nor directory: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
