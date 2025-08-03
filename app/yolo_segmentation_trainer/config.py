"""
Configuration settings for YOLO segmentation training.
"""

# Default training parameters
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMAGE_SIZE = 640
DEFAULT_PATIENCE = 50
DEFAULT_WORKERS = 8
DEFAULT_CONFIDENCE = 0.25

# Supported YOLO base models
SUPPORTED_BASE_MODELS = [
    'yolov8n-seg.pt',    # Nano - fastest, lowest accuracy
    'yolov8s-seg.pt',    # Small 
    'yolov8m-seg.pt',    # Medium
    'yolov8l-seg.pt',    # Large
    'yolov8x-seg.pt',    # Extra Large - slowest, highest accuracy
]

# Data preparation settings
DEFAULT_MIN_CONTOUR_AREA = 200
SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# Training output structure
TRAINING_OUTPUTS = {
    'WEIGHTS_DIR': 'weights',
    'BEST_MODEL': 'best.pt',
    'LAST_MODEL': 'last.pt',
    'RESULTS_FILE': 'results.csv',
    'CONFUSION_MATRIX': 'confusion_matrix.png',
    'VALIDATION_BATCH': 'val_batch*.jpg'
} 