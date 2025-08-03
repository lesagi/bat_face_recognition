"""
Siamese network path configuration settings.
"""

import os

# Base directory path
BASE_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Siamese network paths
SIAMESE_PATHS = {
    "BASE": os.path.join(BASE_DIR_PATH, "face_recognition"),
    "TRAINING": {
        "DATA": os.path.join(BASE_DIR_PATH, "face_recognition", "data"),
        "OUTPUT": os.path.join(BASE_DIR_PATH, "face_recognition", "output"),
        "CHECKPOINTS": os.path.join(BASE_DIR_PATH, "face_recognition", "checkpoints"),
        "MODEL": os.path.join(BASE_DIR_PATH, "face_recognition", "model")
    }
} 