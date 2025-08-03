# Background Replacement Module

This module provides comprehensive background replacement functionality for bat face processing, using segmentation models to identify objects and replace their backgrounds.

## Core Components

### Classes
- **`Processor`**: Main processor class for background replacement operations
- **`BatchProcessor`**: Handles batch processing of multiple images

### Factory Functions
- **`create_processor(model, confidence_threshold=0.3)`**: Create processor with explicit model
- **`create_processor_from_config(confidence_threshold=0.3)`**: Create processor using configured model path
- **`create_batch_processor(processor=None, model=None, confidence_threshold=0.3)`**: Create batch processor
- **`create_batch_processor_from_config(confidence_threshold=0.3)`**: Create batch processor using configured model path
- **`load_model(model_path)`**: Load and validate YOLO model from file

### Background Generation
Background generation functionality is provided by the separate `background_generation` module:
- **`BackgroundGenerator`**: Utility class for generating different background types
- **`BackgroundPresets`**: Pre-configured background generation functions

## Installation Requirements

```bash
pip install ultralytics opencv-python numpy
```

## Quick Start

### Basic Usage

```python
from background_replacement import create_processor
from background_generation import BackgroundPresets

# Create processor with custom model
processor = create_processor("/path/to/your/model.pt")

# Or use configured model from paths
processor = create_processor_from_config()

# Process single image with solid color background
success = processor.apply_background(
    image_path="input.jpg",
    output_path="output/",
    background_generator=BackgroundPresets.solid_color(color=(255, 255, 255))
)
```

### Batch Processing

```python
from background_replacement import create_batch_processor_from_config
from background_generation import BackgroundPresets

# Create batch processor
batch_processor = create_batch_processor_from_config()

# Process all images in a directory
batch_processor.process_directory(
    input_directory="input_images/",
    output_directory="output_images/",
    background_generator=BackgroundPresets.solid_color()
)
```

### Background Types

```python
from background_generation import BackgroundGenerator, BackgroundPresets

# Solid color backgrounds
white_bg = BackgroundPresets.solid_color(color=(255, 255, 255))
black_bg = BackgroundPresets.solid_color(color=(0, 0, 0))

# Gradient backgrounds
gradient_bg = BackgroundPresets.gradient(
    start_color=(100, 100, 100),
    end_color=(200, 200, 200)
)

# Noise backgrounds
noise_bg = BackgroundPresets.noise(intensity=50)

# Custom background generator
def custom_background(height, width):
    return BackgroundGenerator.solid_color(height, width, (128, 128, 128))
```

## Advanced Usage

### Model Management

```python
from background_replacement import create_processor, load_model
from ultralytics import YOLO

# Load model explicitly
model = load_model("/path/to/model.pt")
processor = create_processor(model)

# Or use YOLO instance directly
yolo_model = YOLO("yolov8n-seg.pt")
processor = create_processor(yolo_model)
```

### Confidence Threshold

```python
# Set confidence threshold during creation
processor = create_processor("/path/to/model.pt", confidence_threshold=0.7)

# Or use with configured model
processor = create_processor_from_config(confidence_threshold=0.8)
```

### Direct Segmentation

```python
import cv2
from background_replacement import create_processor_from_config

processor = create_processor_from_config()

# Load image
image = cv2.imread("input.jpg")

# Get segmentation result
result = processor.segment_image(image)
if result:
    original_image, mask = result
    print("Bat detected and segmented")
else:
    print("No bat detected")
```

## Error Handling

The module provides comprehensive error handling with specific exception types:

| Exception | Description | Common Causes |
|-----------|-------------|---------------|
| `FileNotFoundError` | Model file not found | Incorrect model path |
| `ValueError` | Invalid parameters | Wrong file extension, invalid confidence threshold |
| `RuntimeError` | Model loading/operation failed | Corrupted model, wrong model type |
| `ImportError` | Configuration not available | Missing paths configuration |

### Example Error Handling

```python
from background_replacement import create_processor

try:
    processor = create_processor("/path/to/model.pt")
except FileNotFoundError:
    print("Model file not found. Please check the path.")
except ValueError as e:
    print(f"Invalid model file: {e}")
except RuntimeError as e:
    print(f"Model loading failed: {e}")
```

## Batch Processing Features

### Directory Processing

```python
from background_replacement import create_batch_processor_from_config
from background_generation import BackgroundPresets

batch_processor = create_batch_processor_from_config()

# Process all images in directory
batch_processor.process_directory(
    input_directory="input/",
    output_directory="output/",
    background_generator=BackgroundPresets.solid_color()
)

# Process with custom suffix
batch_processor.process_directory(
    input_directory="input/",
    output_directory="output/",
    background_generator=BackgroundPresets.gradient(),
    suffix="_gradient"
)
```

### Image List Processing

```python
from background_generation import BackgroundPresets

image_list = ["image1.jpg", "image2.jpg", "image3.jpg"]

batch_processor.process_image_list(
    image_paths=image_list,
    output_directory="output/",
    background_generator=BackgroundPresets.noise()
)
```

## Configuration

The module uses configuration from `paths.py` for default model paths:

```python
# In paths.py
MODEL_USAGE_PATHS = {
    "ORIGINAL_IMAGES": "/path/to/your/segmentation/model.pt"
}
```

## Model Validation

The module automatically validates YOLO models:

### Supported Model Formats
- `.pt` (PyTorch)
- `.onnx` (ONNX)
- `.engine` (TensorRT)
- `.torchscript` (TorchScript)
- `.mlmodel` (CoreML)

### Validation Checks
- File existence and accessibility
- Valid YOLO model format
- Segmentation task capability (when detectable)

## Performance Tips

1. **Model Choice**: Use appropriate model size for your needs (nano, small, medium, large, extra-large)
2. **Confidence Threshold**: Adjust based on your accuracy requirements during processor creation
3. **Batch Processing**: Use batch processing for multiple images to improve efficiency
4. **Model Caching**: Reuse processor instances to avoid reloading models

## Migration Guide

### From Previous Versions

If you were using the old `BatFaceSegmentationProcessor` class:

```python
# Old way (deprecated)
from segmentation import BatFaceSegmentationProcessor
processor = BatFaceSegmentationProcessor()

# New way
from background_replacement import create_processor_from_config
processor = create_processor_from_config()
```

### Background Generation Module Separation

Background generation functionality has been moved to a separate module:

```python
# Old way
from segmentation import BackgroundGenerator, BackgroundPresets

# New way
from background_generation import BackgroundGenerator, BackgroundPresets
```

### Function Naming Changes

- Use `create_processor_from_config()` to read model from configuration
- Use `create_processor(model)` to specify model explicitly
- Use `create_batch_processor_from_config()` for batch processing with configured model

## Examples

See `examples.py` for comprehensive usage examples including:
- Basic background replacement with different backgrounds
- Batch processing scenarios
- Error handling patterns
- Model management examples
- Performance optimization tips
