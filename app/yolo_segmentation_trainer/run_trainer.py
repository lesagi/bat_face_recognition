# from ultralytics import YOLO
# import os

# print('✅ Starting 50-epoch bat segmentation training...')
# model = YOLO('yolov8n-seg.pt')

# # Your requested 50-epoch training
# results = model.train(
#     data=os.path.join(os.path.dirname(__file__), 'rousesttus', 'data', 'data.yaml'),
#     epochs=50,
#     batch=8,
#     patience=25,
#     project='rousesttus',
#     name='training_results',
#     verbose=True,
#     plots=True,
#     save_period=10,
#     device='cpu'  # Explicit CPU to avoid any GPU issues
# )

# print('🎉 Training completed successfully!')
# print(f'✅ Best model saved to: training_final/bat_segmentation_50epochs/weights/best.pt')
# print(f'✅ Last model saved to: training_final/bat_segmentation_50epochs/weights/last.pt')

from ultralytics import YOLO
import os

# Path to your data.yaml
DATA_YAML_PATH = "data/data.yaml"

# Define training parameters
model = YOLO("yolov8n-seg.pt")  # You can change to 'yolov8s-seg.pt', etc.

model.train(
    data=os.path.join(os.path.dirname(__file__), "rousesttus", "data", "data.yaml"),
    imgsz=640,  # Input image size
    epochs=100,  # Number of training epochs
    batch=8,  # Batch size (adjust based on your GPU)
    name="bat_face_seg",  # Output folder name in runs/segment/
    task="segment",  # Ensure task type is segmentation
    save=True,  # Save checkpoints
    save_period=10,  # Save every 10 epochs
    patience=20,  # Early stopping patience
    verbose=True,
    project="runs/segment",  # Output directory
    exist_ok=True,  # Overwrite if run exists
)
