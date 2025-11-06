import os
from datetime import datetime

# General Config
BASE_DIR_PATH = os.path.abspath('..')
GPU_ENABLED = True

# Get the current timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


# Define OUTPUT folder
def create_output_dir_name(x):
    return f'output_{x}_{timestamp}' if x else f'output_{timestamp}'


# ------------------------ TRAINING CONFIG --------------------------

# The SIAMESE NETWORK training data directory
FACES_SNAPSHOTS_BASE_DIR_PATH = os.path.join(BASE_DIR_PATH, 'face_recognition-with_bg_31_03_25')
SIAMESE_TRAINING_DATA_BASE_DIR = os.path.join(FACES_SNAPSHOTS_BASE_DIR_PATH, 'data', 'training')

# The SIAMESE NETWORK training output directory
SIAMESE_TRAINING_OUTPUT_DIR = os.path.join(BASE_DIR_PATH, create_output_dir_name("siamese"), 'train')
SIAMESE_TRAINING_CHECKPOINT_DIR = os.path.join(SIAMESE_TRAINING_OUTPUT_DIR, 'checkpoints')
SIAMESE_TRAINING_MODEL_OUTPUT_PATH = os.path.join(SIAMESE_TRAINING_OUTPUT_DIR, 'model')
FACE_RECOGNITION_TRAINING_DATA_MAX_SIZE_LIMIT = 2000
SIAMESE_INPUT_EDGE_LENGTH = 105
SIAMESE_TRAIN_EPOCHS = 40


# ------------------------ EVALUATION CONFIG --------------------------
TRAINED_MODEL_DIR_NAME = 'output_siamese_20250331_191022'

# The SIAMESE NETWORK saved model directory
TRAINED_SIAMESE_MODEL_BASE_DIR_PATH = os.path.join(BASE_DIR_PATH, TRAINED_MODEL_DIR_NAME)
TRAINED_SIAMESE_MODEL_PATH = None # os.path.join(TRAINED_SIAMESE_MODEL_BASE_DIR_PATH, 'train', 'model', 'siamesemodelv2_v2')
SIAMESE_OUTPUT_EVALUATIONS_DIR = os.path.join(TRAINED_SIAMESE_MODEL_BASE_DIR_PATH, 'evaluations')

# Saliency Map Config
INPUT_DIR_FOR_SALIENCY_MAPS = FACES_SNAPSHOTS_BASE_DIR_PATH
OUTPUT_DIR_FOR_SALIENCY_MAPS = os.path.join(SIAMESE_OUTPUT_EVALUATIONS_DIR, f'saliency_maps_{TRAINED_MODEL_DIR_NAME}')

# Face Segmentation Config
FACE_SEGMENTATION_BASE_PATH = os.path.join(BASE_DIR_PATH, 'face_segmentation')
FACE_SEGMENTATION_DATA_PATH = os.path.join(FACE_SEGMENTATION_BASE_PATH, 'data')
FACE_SEGMENTATION_CONFIG_PATH = os.path.join(FACE_SEGMENTATION_BASE_PATH, 'config.yaml')
FACE_SEGMENTATION_EPOCHS = 50

# Video Processing Config
VIDEOS_BASE_DIR = os.path.join(BASE_DIR_PATH, 'raw_data', 'videos')
VIDEOS_DIR = os.path.join(VIDEOS_BASE_DIR, 'train')
VIDEOS_PROCESSING_LIMIT = 1
PROCESSED_VIDEOS_BASE_DIR = os.path.join(VIDEOS_BASE_DIR, 'processed')
PROCESSED_VIDEOS_SUB_DIR = 'segmentation'
VIDEO_PROCESSING_MODEL_PATH = os.path.join(FACE_SEGMENTATION_BASE_PATH, 'chosen_model')

# Augmentation Config
AUGMENTED_IMAGES_BASE_DIR_INPUT = os.path.join(SIAMESE_TRAINING_DATA_BASE_DIR, 'training')
AUGMENTED_IMAGES_BASE_DIR_OUTPUT = os.path.join(SIAMESE_TRAINING_DATA_BASE_DIR, 'augmented')
AUGMENTED_IMAGES_SUB_DIR = None
