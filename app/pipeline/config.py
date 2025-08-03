"""
Configuration settings for the ML pipeline.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
PIPELINE_DIR = BASE_DIR / "pipeline"
DATA_DIR = PIPELINE_DIR / "data"
MODELS_DIR = PIPELINE_DIR / "models"
TRAINING_DIR = PIPELINE_DIR / "training"
EVALUATION_DIR = PIPELINE_DIR / "evaluation"
DEPLOYMENT_DIR = PIPELINE_DIR / "deployment"

# MLflow settings
MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "bat_face_recognition"

# Data settings
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = DATA_DIR / "train"
TEST_DATA_DIR = DATA_DIR / "test"
VAL_DATA_DIR = DATA_DIR / "val"

# Model settings
MODEL_CONFIG = {
    "input_size": 224,
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001,
    "margin": 1.0
}

# Training settings
TRAINING_CONFIG = {
    "early_stopping_patience": 5,
    "validation_split": 0.2,
    "random_seed": 42
}

# Evaluation settings
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1_score"],
    "confusion_matrix": True
}

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, TRAINING_DIR, EVALUATION_DIR, DEPLOYMENT_DIR,
                 RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_DATA_DIR, TEST_DATA_DIR, VAL_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 