from ultralytics import YOLO
import os
import sys

# Add the app directory to the path to import config
app_dir = os.path.join(os.path.dirname(__file__), '..')
project_root = os.path.join(app_dir, '..')
sys.path.insert(0, app_dir)
sys.path.insert(0, project_root)

from app.config.loader import load_config
from app.config.yolo_segmentation import YOLOSegmentationConfig

# Load configuration
config = load_config()
yolo_config = config.yolo_segmentation

# Define training parameters
model = YOLO("yolov8n-seg.pt")  # You can change to 'yolov8s-seg.pt', etc.

model.train(
    data=yolo_config.data_path,
    imgsz=640,  # Input image size - determines the resolution images are resized to during training
    epochs=100,  # Number of training epochs
    batch=8,  # Batch size (adjust based on your GPU)
    name=yolo_config.name,  # Output folder name in runs/segment/
    task="segment",  # Ensure task type is segmentation
    save=True,  # Save checkpoints
    save_period=10,  # Save every 10 epochs
    patience=20,  # Early stopping patience
    verbose=True,
    project=f"runs_{yolo_config.name}",  # Output directory
    exist_ok=True,  # Overwrite if run exists
)
