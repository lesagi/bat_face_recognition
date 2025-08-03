"""
Utility functions for the ML pipeline.
"""

import os
import mlflow
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime

def log_artifacts(artifacts_dir, artifact_path=None):
    """Log artifacts to MLflow"""
    if os.path.exists(artifacts_dir):
        mlflow.log_artifacts(artifacts_dir, artifact_path)

def save_dict_to_json(data, filepath):
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_dict_from_json(filepath):
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_timestamp():
    """Get current timestamp in a formatted string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_gpu():
    """Configure GPU settings"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPU found, using CPU")

def log_model_architecture(model, run_id):
    """Log model architecture to MLflow"""
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    summary_str = "\n".join(model_summary)
    
    # Save summary to file
    summary_path = Path("model_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_str)
    
    # Log to MLflow
    mlflow.log_artifact(str(summary_path), f"model_architecture_{run_id}")
    
    # Clean up
    summary_path.unlink()

def calculate_metrics(y_true, y_pred):
    """Calculate common classification metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1_score": f1_score(y_true, y_pred, average='weighted')
    }
    
    return metrics

def plot_training_history(history, save_path):
    """Plot and save training history"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 