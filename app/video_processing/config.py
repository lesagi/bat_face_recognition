"""
Video processing configuration settings.
"""

import os

# Base directory path
BASE_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Video processing paths
VIDEOS_BASE_DIR = os.path.join(BASE_DIR_PATH, "raw_data", "videos")
VIDEOS_DIR = os.path.join(VIDEOS_BASE_DIR, "input")
PROCESSED_VIDEOS_BASE_DIR = os.path.join(VIDEOS_BASE_DIR, "processed")
PROCESSED_VIDEOS_SUB_DIR = os.path.join(PROCESSED_VIDEOS_BASE_DIR, "segmentation")

# Processing limits
VIDEOS_PROCESSING_LIMIT = 10  # Maximum number of videos to process at once 