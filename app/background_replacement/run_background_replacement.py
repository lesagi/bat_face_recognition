#!/usr/bin/env python3
"""
Simple background replacement script for nested directories.
"""

import sys
import os

# Add the app directory to Python path FIRST
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

# Now we can import the modules
from background_replacement import create_batch_processor, create_processor
from background_generation import BackgroundPresets

# Create batch processor with your trained model
batch_processor = create_batch_processor(
    model="/Users/MAC/Documents/bat_face_rec/app/yolo_segmentation_trainer/runs/segment/bat_face_seg/weights/best.pt",
    confidence_threshold=0.2,
)

print("✓ Created batch processor with your trained model")
print("Processing with random Picsum backgrounds...")

# Process all images in a directory (note: correct parameter names are input_dir/output_dir)
processed_count = batch_processor.process_directory(
    input_dir="/Users/MAC/Documents/bat_face_rec/face_rec_rousettus_#1/background_replacement/original",
    output_dir="/Users/MAC/Documents/bat_face_rec/face_rec_rousettus_#1/background_replacement/processed_picsum",
    background_generator=BackgroundPresets.picsum(
        max_retries=3,  # Try 3 times before giving up
        timeout=10,  # 10 second timeout per request
        fallback_to_generated=True,
    ),
    suffix="_picsum_bg",
)

print(f"✓ Successfully processed {processed_count} images with Picsum backgrounds!")
print("Check the 'processed_picsum' directory for results.")
