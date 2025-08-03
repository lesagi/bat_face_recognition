# YOLO Segmentation Trainer Module

This module provides a comprehensive training framework for YOLO segmentation models specifically designed for bat face segmentation tasks.

## Features

- **Data Preparation**: Convert binary masks to YOLO segmentation labels
- **Model Training**: Train YOLO segmentation models with configurable parameters
- **Validation**: Validate trained models and get performance metrics
- **Prediction**: Run inference on sample images
- **Error Handling**: Comprehensive error handling with specific exceptions

## Installation Requirements

```bash
pip install ultralytics opencv-python numpy
```

## Quick Start

### Basic Training Workflow

```python
from yolo_segmentation_trainer import YoloSegmentationTrainer

# Initialize trainer with dataset config
trainer = YoloSegmentationTrainer(
    config_path="path/to/dataset.yaml",
    epochs=100,
    base_model="yolov8s-seg.pt"
)

# Prepare labels from binary masks (if needed)
from yolo_segmentation_trainer import YoloTrainingUtils
label_count = YoloTrainingUtils.prepare_labels_from_masks(
    mask_dir="path/to/masks/",
    output_dir="path/to/labels/"
)

# Train the model
best_model_path = trainer.train(
    save_dir="training_results/",
    batch_size=16,
    image_size=640
)

# Validate the trained model
metrics = trainer.validate()
print(f"Validation metrics: {metrics}")

# Test prediction on sample image
result = trainer.predict_sample("path/to/test_image.jpg")
if result:
    print(f"Found {len(result['masks'])} objects")
```

## API Reference

### YoloSegmentationTrainer Class

#### Constructor

```python
YoloSegmentationTrainer(config_path, epochs=80, base_model='yolov8n-seg.pt')
```

**Parameters:**
- `config_path` (str): Path to YOLO dataset configuration file (required)
- `epochs` (int): Number of training epochs (default: 80)
- `base_model` (str): Base YOLO model to use (default: 'yolov8n-seg.pt')

**Raises:**
- `FileNotFoundError`: If config_path doesn't exist
- `ValueError`: If epochs is not positive

#### Methods

##### train()

Train the YOLO segmentation model.

```python
trainer.train(save_dir=None, patience=50, batch_size=16, image_size=640, workers=8)
```

**Parameters:**
- `save_dir` (str, optional): Directory to save training results
- `patience` (int): Early stopping patience (default: 50)
- `batch_size` (int): Training batch size (default: 16)
- `image_size` (int): Input image size (default: 640)
- `workers` (int): Number of data loading workers (default: 8)

**Returns:** Path to the best trained model

##### validate()

Validate the trained model.

```python
trainer.validate(model_path=None)
```

**Parameters:**
- `model_path` (str, optional): Path to model file (uses trained model if not specified)

**Returns:** Dictionary containing validation metrics

##### predict_sample()

Run prediction on a sample image.

```python
trainer.predict_sample(image_path, model_path=None, confidence=0.25)
```

**Parameters:**
- `image_path` (str): Path to image file
- `model_path` (str, optional): Path to model file (uses trained model if not specified)
- `confidence` (float): Confidence threshold for predictions (default: 0.25)

**Returns:** Prediction results dictionary or None if no predictions

##### get_model_info()

Get information about the current model.

```python
trainer.get_model_info()
```

**Returns:** Dictionary containing model information

## Utility Methods

The module also provides standalone utility methods through the `YoloTrainingUtils` class for data preparation tasks. These methods can be used independently of the main trainer class.

### YoloTrainingUtils Class

#### prepare_labels_from_masks()

Convert binary mask images to YOLO segmentation label format. This is a standalone version of the method also available in the trainer class.

```python
from yolo_segmentation_trainer import YoloTrainingUtils

# Convert masks to YOLO labels
label_count = YoloTrainingUtils.prepare_labels_from_masks(
    mask_dir="path/to/binary/masks/",
    output_dir="path/to/yolo/labels/",
    min_contour_area=200
)
print(f"Created {label_count} label files")
```

**Parameters:**
- `mask_dir` (str): Directory containing binary mask images
- `output_dir` (str): Directory to save YOLO label files  
- `min_contour_area` (int): Minimum contour area to include (default: 200)

**Returns:** Number of label files created

**Raises:**
- `FileNotFoundError`: If mask_dir doesn't exist

#### cleanup_images_without_labels()

Remove images that don't have corresponding label files. Useful for cleaning datasets before training to ensure perfect image-label correspondence.

**Expected Directory Structure:**

This method expects a specific directory structure pattern:

```
# Image directory structure
image_base_dir/
├── subject1/                    # Subject/class subdirectories
│   ├── IMG_001.jpg             # Image files
│   ├── IMG_002.jpg
│   └── IMG_003.jpg
├── subject2/
│   ├── IMG_001.jpg
│   └── IMG_002.jpg
└── subject3/
    └── IMG_001.jpg

# Label directory structure (CVAT export format)
label_base_dir/
├── subject1_annotations_TIMESTAMP_ultralytics yolo segmentation 1.0/
│   └── labels/
│       └── train/              # YOLO label files must be in labels/train/
│           ├── IMG_001.txt     # Matching label filenames (same basename)
│           └── IMG_002.txt
├── subject2_annotations_TIMESTAMP_ultralytics yolo segmentation 1.0/
│   └── labels/
│       └── train/
│           └── IMG_001.txt
└── subject3_annotations_TIMESTAMP_ultralytics yolo segmentation 1.0/
    └── labels/
        └── train/
            └── (no labels - IMG_001.jpg will be removed)
```

**Pattern Matching:**
- Image directories: Direct subdirectories under `image_base_dir`
- Label directories: Must match pattern `{subject_name}_annotations_*` under `label_base_dir`
- Label files: Must be located in `labels/train/` within each annotation directory
- File correspondence: Image `filename.jpg` matches label `filename.txt`

```python
from yolo_segmentation_trainer import YoloTrainingUtils

# Clean up dataset - remove images without labels
stats = YoloTrainingUtils.cleanup_images_without_labels(
    image_base_dir="path/to/image/directories/",
    label_base_dir="path/to/label/directories/",
    image_extensions=['.jpg', '.jpeg', '.png'],
    verbose=True
)

print(f"Total images removed: {stats['total_removed']}")
print("Details by directory:", stats['by_directory'])
```

**Parameters:**
- `image_base_dir` (str): Base directory containing image subdirectories
- `label_base_dir` (str): Base directory containing label annotation subdirectories
- `image_extensions` (List[str]): List of image file extensions to consider (default: `['.jpg', '.jpeg', '.png']`)
- `verbose` (bool): Whether to print detailed progress information (default: `True`)

**Returns:** Dictionary with cleanup statistics:
```python
{
    'total_removed': int,
    'by_directory': {
        'dir_name': {
            'removed': int,
            'original_images': int,
            'available_labels': int,
            'remaining_images': int
        }
    }
}
```

**Raises:**
- `FileNotFoundError`: If base directories don't exist

**Important Notes:**
- This method is specifically designed for CVAT annotation exports in YOLO format
- For other annotation tools or directory structures, you may need to adapt the method
- The method uses glob pattern matching: `{subject_name}_annotations_*` to find label directories
- Label files must be in the exact path: `labels/train/*.txt` within each annotation directory

#### create_yolo_dataset()

Create YOLO dataset structure with train/validation splits from existing directories. This method transforms your cleaned dataset into the standard YOLO format with proper train/validation splits.

**Expected Input Structure:**

This method expects the same CVAT directory structure as `cleanup_images_without_labels()`:

```
# Your dataset directory (e.g., rousesttus/)
dataset_dir/
├── original_images/             # Cleaned images (after cleanup)
│   ├── subject1/
│   │   ├── IMG_001.jpg
│   │   └── IMG_002.jpg
│   ├── subject2/
│   │   └── IMG_003.jpg
│   └── subject3/
│       └── IMG_004.jpg
└── cvat_labels/                 # CVAT annotation exports
    ├── subject1_annotations_TIMESTAMP_ultralytics yolo segmentation 1.0/
    │   └── labels/train/
    │       ├── IMG_001.txt
    │       └── IMG_002.txt
    ├── subject2_annotations_TIMESTAMP_ultralytics yolo segmentation 1.0/
    │   └── labels/train/
    │       └── IMG_003.txt
    └── subject3_annotations_TIMESTAMP_ultralytics yolo segmentation 1.0/
        └── labels/train/
            └── IMG_004.txt
```

**Created Output Structure:**

```
dataset_dir/
├── original_images/             # (unchanged)
├── cvat_labels/                 # (unchanged)
└── data/                        # ✨ New YOLO dataset
    ├── train/
    │   ├── images/
    │   │   ├── subject1_IMG_001.jpg    # Prefixed with subject name
    │   │   ├── subject2_IMG_003.jpg
    │   │   └── subject3_IMG_004.jpg
    │   └── labels/
    │       ├── subject1_IMG_001.txt
    │       ├── subject2_IMG_003.txt
    │       └── subject3_IMG_004.txt
    ├── val/
    │   ├── images/
    │   │   └── subject1_IMG_002.jpg
    │   └── labels/
    │       └── subject1_IMG_002.txt
    └── data.yaml                # YOLO configuration file
```

```python
from yolo_segmentation_trainer import YoloTrainingUtils

# Create YOLO dataset structure with train/val splits
stats = YoloTrainingUtils.create_yolo_dataset(
    image_base_dir="dataset_dir/original_images/",
    label_base_dir="dataset_dir/cvat_labels/",
    # output_dir is auto-detected to "dataset_dir/data/" 
    train_split=0.8,
    seed=42,
    copy_files=True,                             # Preserve original files
    verbose=True
)

print(f"Dataset created: {stats['dataset_path']}")
print(f"Config file: {stats['config_file']}")
print(f"Train files: {stats['train_count']}, Val files: {stats['val_count']}")
```

**Parameters:**
- `image_base_dir` (str): Base directory containing image subdirectories
- `label_base_dir` (str): Base directory containing label annotation subdirectories
- `output_dir` (str, optional): Output directory for YOLO dataset structure (auto-detected if None)
- `train_split` (float): Fraction of data to use for training (0.0-1.0, default: 0.8)
- `seed` (int): Random seed for reproducible splits (default: 42)
- `copy_files` (bool): If True, copy files; if False, move files (default: True)
- `image_extensions` (List[str]): List of image file extensions to consider (default: `['.jpg', '.jpeg', '.png']`)
- `verbose` (bool): Whether to print detailed progress information (default: `True`)

**Returns:** Dictionary with dataset statistics:
```python
{
    'dataset_path': str,           # Path to created dataset
    'config_file': str,            # Path to data.yaml file
    'total_files': int,            # Total files processed
    'train_count': int,            # Number of training files
    'val_count': int,              # Number of validation files
    'train_split': float,          # Train/val split used
    'by_subject': dict,            # Per-subject statistics
    'directories_created': dict    # Paths to created directories
}
```

**Raises:**
- `FileNotFoundError`: If base directories don't exist
- `ValueError`: If train_split is not between 0 and 1

**Key Features:**
- **Smart directory detection**: Automatically creates `data/` folder in the same directory as your images and labels
- **Automatic train/val split**: Configurable split ratio with reproducible results
- **Subject name prefixing**: Prevents filename conflicts across subjects
- **Perfect file matching**: Only processes image-label pairs that exist in both directories
- **Preserves originals**: Copy mode preserves your cleaned source data
- **Auto-generated config**: Creates ready-to-use `data.yaml` for YOLO training
- **Comprehensive reporting**: Detailed statistics by subject and overall

### Complete Data Preparation Workflow

Here's how to use all three utility methods together for complete dataset preparation:

```python
from yolo_segmentation_trainer import YoloTrainingUtils

# Step 1: Convert binary masks to YOLO labels (if needed)
print("Converting masks to YOLO labels...")
label_count = YoloTrainingUtils.prepare_labels_from_masks(
    mask_dir="dataset_dir/masks/",
    output_dir="dataset_dir/labels/",
    min_contour_area=150
)
print(f"Created {label_count} label files")

# Step 2: Clean up images without matching labels
print("\nCleaning up images without labels...")
cleanup_stats = YoloTrainingUtils.cleanup_images_without_labels(
    image_base_dir="dataset_dir/original_images/",
    label_base_dir="dataset_dir/cvat_labels/",
    verbose=True
)

print(f"Dataset cleanup complete:")
print(f"- Total images removed: {cleanup_stats['total_removed']}")
print(f"- Directories processed: {len(cleanup_stats['by_directory'])}")

# Step 3: Create YOLO dataset structure with train/val splits
print("\nCreating YOLO dataset structure...")
dataset_stats = YoloTrainingUtils.create_yolo_dataset(
    image_base_dir="dataset_dir/original_images/",
    label_base_dir="dataset_dir/cvat_labels/",
    # output_dir auto-detected to "dataset_dir/data/"
    train_split=0.8,
    seed=42,
    copy_files=True,
    verbose=True
)

print(f"YOLO dataset created:")
print(f"- Dataset path: {dataset_stats['dataset_path']}")
print(f"- Config file: {dataset_stats['config_file']}")
print(f"- Train files: {dataset_stats['train_count']}")
print(f"- Val files: {dataset_stats['val_count']}")

# Step 4: Now ready for training!
from yolo_segmentation_trainer import YoloSegmentationTrainer

trainer = YoloSegmentationTrainer(dataset_stats['config_file'])  # Use generated config
model_path = trainer.train()
```

**Real-world example with rousesttus dataset:**
```python
from yolo_segmentation_trainer import YoloTrainingUtils

# Complete workflow for bat face recognition dataset
base_dir = "yolo_segmentation_trainer/rousesttus"

# Step 1: Clean up mismatched images/labels
cleanup_stats = YoloTrainingUtils.cleanup_images_without_labels(
    image_base_dir=f"{base_dir}/original_images/",
    label_base_dir=f"{base_dir}/cvat_labels/",
    verbose=True
)

# Step 2: Create YOLO dataset
dataset_stats = YoloTrainingUtils.create_yolo_dataset(
    image_base_dir=f"{base_dir}/original_images/",
    label_base_dir=f"{base_dir}/cvat_labels/",
    # output_dir auto-detected to f"{base_dir}/data/"
    train_split=0.8,
    verbose=True
)

# Step 3: Train the model
trainer = YoloSegmentationTrainer(f"{base_dir}/data/data.yaml")
model_path = trainer.train()

print(f"Training complete! Model saved to: {model_path}")
```

**Final directory structure:**
```
yolo_segmentation_trainer/rousesttus/
├── original_images/             # Cleaned source images
├── cvat_labels/                 # CVAT annotation exports  
└── data/                        # Ready-to-use YOLO dataset
    ├── train/images/            # Training images
    ├── train/labels/            # Training labels
    ├── val/images/              # Validation images
    ├── val/labels/              # Validation labels
    └── data.yaml                # YOLO configuration
```

### Utility Methods vs Trainer Methods

| Method | Location | Use Case |
|--------|----------|----------|
| `YoloTrainingUtils.prepare_labels_from_masks()` | Standalone utility | Convert binary masks to YOLO labels |
| `YoloTrainingUtils.cleanup_images_without_labels()` | Standalone utility | Remove unmatched images, dataset cleaning |
| `YoloTrainingUtils.create_yolo_dataset()` | Standalone utility | Create train/val splits, YOLO format |
| `YoloSegmentationTrainer.prepare_labels_from_masks()` | *Removed* | Use utility method instead |

**Data Preparation Pipeline:**
1. **prepare_labels_from_masks()** - Convert masks to labels (if needed)
2. **cleanup_images_without_labels()** - Ensure perfect image-label matching  
3. **create_yolo_dataset()** - Create final YOLO dataset with splits
4. **YoloSegmentationTrainer** - Train the model

**Note:** The trainer class focuses purely on training. Use the utility class methods for all data preparation tasks to maintain clear separation of concerns.

## Configuration

### Supported Base Models

- `yolov8n-seg.pt` - Nano (fastest, lowest accuracy)
- `yolov8s-seg.pt` - Small
- `yolov8m-seg.pt` - Medium
- `yolov8l-seg.pt` - Large
- `yolov8x-seg.pt` - Extra Large (slowest, highest accuracy)

### Dataset Configuration

Create a YAML configuration file for your dataset:

```yaml
# dataset.yaml
train: path/to/train/images
val: path/to/val/images
test: path/to/test/images  # optional

nc: 1  # number of classes
names: ['bat_face']  # class names
```

## Advanced Usage

### Custom Training Parameters

```python
trainer = YoloSegmentationTrainer(
    config_path="dataset.yaml",
    epochs=200,
    base_model="yolov8m-seg.pt"
)

# Train with custom parameters
model_path = trainer.train(
    save_dir="custom_training/",
    patience=100,
    batch_size=8,  # Smaller batch for larger model
    image_size=1024,  # Higher resolution
    workers=4
)
```

### Batch Label Preparation

```python
from yolo_segmentation_trainer import YoloTrainingUtils

# Process multiple mask directories
mask_directories = [
    "masks/train/",
    "masks/val/",
    "masks/test/"
]

label_directories = [
    "labels/train/",
    "labels/val/",
    "labels/test/"
]

for mask_dir, label_dir in zip(mask_directories, label_directories):
    count = YoloTrainingUtils.prepare_labels_from_masks(mask_dir, label_dir)
    print(f"Processed {count} files from {mask_dir}")
```

### Model Comparison

```python
# Train different model sizes
models = ['yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt']
results = {}

for model in models:
    trainer = YoloSegmentationTrainer("dataset.yaml", base_model=model)
    model_path = trainer.train(save_dir=f"results_{model.split('-')[0]}/")
    metrics = trainer.validate()
    results[model] = metrics

# Compare results
for model, metrics in results.items():
    print(f"{model}: {metrics}")
```

## Error Handling

The module provides comprehensive error handling:

```python
try:
    trainer = YoloSegmentationTrainer("dataset.yaml")
    model_path = trainer.train()
    metrics = trainer.validate()
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except RuntimeError as e:
    print(f"Training/validation failed: {e}")
```

## Performance Tips

1. **Model Selection**: Choose model size based on your accuracy/speed requirements
2. **Batch Size**: Adjust based on available GPU memory
3. **Image Size**: Higher resolution improves accuracy but increases training time
4. **Early Stopping**: Use patience parameter to prevent overfitting
5. **Data Quality**: Ensure mask quality affects training performance significantly

## Integration

This module can be easily integrated with other parts of the bat face recognition pipeline:

```python
from yolo_segmentation_trainer import YoloSegmentationTrainer
from background_replacement import create_processor

# Train a new model
trainer = YoloSegmentationTrainer("dataset.yaml")
new_model_path = trainer.train()

# Use the trained model in background replacement pipeline
processor = create_processor(new_model_path)
``` 