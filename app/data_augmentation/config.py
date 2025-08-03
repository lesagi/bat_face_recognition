"""Augmentation configuration settings."""

import os

# Base directory path
BASE_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Augmentation paths
AUGMENTATION_PATHS = {
    'BASE': '',
    'INPUT': '',
    'OUTPUT': '',
}

# Augmentation types
AUGMENTATION_TYPES = [
    'brightness',
]

# Augmentation factor
AUGMENTATION_FACTOR = 10
