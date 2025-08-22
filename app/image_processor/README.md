# Image Processor

A simple and efficient image processing pipeline for plain image transformations.

## Overview

The `ImageProcessor` class provides a clean interface for processing images with any transformation function. It handles image loading, processing, and saving without any model dependencies.

## Features

- **Single Image Processing**: Process one image with any transformation function
- **Batch Processing**: Process multiple images in a directory
- **Pipeline Processing**: Apply multiple transformations in sequence
- **Automatic File Management**: Handles input/output paths and file extensions
- **Error Handling**: Graceful error handling with informative messages

## Usage

### Basic Processing

```python
from image_processor import ImageProcessor
from image_processor.transforms import ImageTransforms

processor = ImageProcessor()

# Resize an image
processor.process(
    "input.jpg", 
    "output/", 
    ImageTransforms.resize_image, 
    target_size=(224, 224)
)
```

### Batch Processing

```python
# Process all images in a directory
results = processor.process_batch(
    input_directory="input_images/",
    output_directory="processed_images/",
    processing_function=ImageTransforms.resize_image,
    target_size=(224, 224)
)

print(f"Processed {results['success']} images successfully")
```

### Pipeline Processing

```python
# Apply multiple transformations
steps = [
    (ImageTransforms.resize_image, {"target_size": (224, 224)}),
    (ImageTransforms.normalize_image, {"scale": 255.0}),
    (ImageTransforms.adjust_brightness_contrast, {"brightness": 1.2, "contrast": 1.1})
]

processor.process_pipeline(
    "input.jpg",
    "output/",
    processing_steps=steps
)
```

## API Reference

### ImageProcessor

#### `process(image_path, output_path, processing_function, suffix="", **kwargs)`

Process a single image with a transformation function.

- **image_path**: Path to input image
- **output_path**: Directory to save processed image
- **processing_function**: Function that takes image array and returns processed image
- **suffix**: Optional suffix for output filename
- **kwargs**: Additional arguments for processing function

Returns: `True` if successful, `False` otherwise

#### `process_batch(input_directory, output_directory, processing_function, file_extensions=(.jpg, .jpeg, .png, .bmp, .tiff), suffix="", **kwargs)`

Process multiple images in a directory.

- **input_directory**: Directory containing input images
- **output_directory**: Directory to save processed images
- **processing_function**: Function to apply to each image
- **file_extensions**: Allowed file extensions
- **suffix**: Optional suffix for output filenames
- **kwargs**: Additional arguments for processing function

Returns: Dictionary with processing results

#### `process_pipeline(image_path, output_path, processing_steps, suffix="")`

Process an image through multiple transformation steps.

- **image_path**: Path to input image
- **output_path**: Directory to save processed image
- **processing_steps**: List of (function, kwargs) tuples
- **suffix**: Optional suffix for output filename

Returns: `True` if successful, `False` otherwise

## Available Transformations

All transformations are available in `ImageTransforms`:

- `resize_image`: Resize image to target dimensions
- `normalize_image`: Normalize pixel values
- `adjust_brightness_contrast`: Adjust brightness and contrast
- `apply_gamma_correction`: Apply gamma correction
- `apply_gaussian_blur`: Apply Gaussian blur
- `convert_color_space`: Convert between color spaces
- `center_crop_square`: Center crop to square
- `pad_to_square`: Pad image to square dimensions
- `resize_square_image`: Resize square image

## Model-Based Processing

For model-based processing (segmentation, face detection, etc.), use the `ImageTransforms` methods directly:

```python
from image_processor.transforms import ImageTransforms

# Face alignment with automatic model loading from config
result = ImageTransforms.align_face_landmarks(
    image, 
    face_detector_type="yolo_pose"
)

# Background replacement
result = ImageTransforms.apply_background_replacement(
    image, mask, background_generator
)
```

## Error Handling

The processor provides informative error messages:

- File not found errors
- Processing function errors
- Save errors
- Batch processing summaries

## File Management

- Automatically creates output directories
- Preserves original file extensions
- Supports custom suffixes
- Handles multiple image formats 