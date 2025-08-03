"""
Path configuration settings for the application.
"""

import os

# Base directory path
BASE_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Face recognition paths
FACE_RECOGNITION_PATHS = {
    "BASE": os.path.join(BASE_DIR_PATH, "face_recognition"),
    "TRAINING": {
        "DATA": os.path.join(BASE_DIR_PATH, "face_recognition", "data"),
        "OUTPUT": os.path.join(BASE_DIR_PATH, "face_recognition", "output"),
        "CHECKPOINTS": os.path.join(BASE_DIR_PATH, "face_recognition", "checkpoints"),
        "MODEL": os.path.join(BASE_DIR_PATH, "face_recognition", "model"),
    },
    "AUGMENTATION": {
        "INPUT": os.path.join(BASE_DIR_PATH, "face_recognition", "data", "original"),
        "OUTPUT": os.path.join(BASE_DIR_PATH, "face_recognition", "data", "augmented"),
    },
}

# Face segmentation paths
FACE_SEGMENTATION_PATHS = {
    "BASE": os.path.join(BASE_DIR_PATH, "face_segmentation"),
    "MODEL": os.path.join(
        BASE_DIR_PATH, "face_segmentation", "chosen_model", "best.pt"
    ),
    "CONFIG": os.path.join(BASE_DIR_PATH, "face_segmentation", "config.yaml"),
    "DATA": os.path.join(BASE_DIR_PATH, "face_segmentation", "data"),
    "RUNS": os.path.join(BASE_DIR_PATH, "face_segmentation", "runs"),
}

# Background replacement paths (commented out until needed)
# SEGMENTATION_PATHS = {
#     "BASE": os.path.join(BASE_DIR_PATH, "background_replacement"),
#     "MODEL": os.path.join(BASE_DIR_PATH, "background_replacement", "model", "best.pt"),
#     "DATA": os.path.join(BASE_DIR_PATH, "background_replacement", "data"),
#     "CONFIG": os.path.join(BASE_DIR_PATH, "background_replacement", "config.yaml")
# }

# Video processing paths
VIDEO_PATHS = {
    "BASE": os.path.join(BASE_DIR_PATH, "video_processing"),
    "INPUT": os.path.join(BASE_DIR_PATH, "raw_data", "videos"),
    "OUTPUT": os.path.join(BASE_DIR_PATH, "video_processing", "output"),
}

# Data augmentation paths
AUGMENTATION_PATHS = {
    "BASE": os.path.join(BASE_DIR_PATH, "data_augmentation"),
    "INPUT": os.path.join(BASE_DIR_PATH, "face_recognition", "data", "original"),
    "OUTPUT": os.path.join(BASE_DIR_PATH, "face_recognition", "data", "augmented"),
}
