# Square Crop Plan for YOLO Segmentation Training Data

## Overview
This plan outlines the creation of square variants of existing YOLO segmentation training data by cropping around segmentation boundaries with a configurable buffer (1.1x multiplier).

## Current Data Structure Analysis
Based on the codebase examination, we have:
- **Source Data**: `data/interim/mauritius_images+labels/` and `data/interim/rous_images+labels/`
- **Format**: YOLO segmentation format with `.txt` label files and corresponding image files
- **Label Format**: YOLO segmentation format (class_id + normalized polygon coordinates)
- **Images**: Mixed formats (`.jpg`, `.png`) with varying dimensions
- **Flexibility**: Tool designed to work with any directory structure following the images+labels pattern

## Objectives
1. **Square Cropping**: Generate square crops centered around segmentation boundaries
2. **Buffer Management**: Apply 1.1x multiplier buffer around segmentation boundaries
3. **Label Transformation**: Properly transform YOLO segmentation labels to new coordinate system
4. **Quality Preservation**: Maintain segmentation accuracy and image quality
5. **Batch Processing**: Process entire datasets efficiently

## Technical Approach

### 1. Segmentation Boundary Detection
- Parse YOLO segmentation labels to extract polygon coordinates
- Convert normalized coordinates to absolute pixel coordinates
- Calculate bounding box around segmentation polygon
- Apply 1.1x buffer multiplier to bounding box dimensions

### 2. Square Crop Generation
- Ensure crop dimensions are square by **always** choosing the bigger size (max of width/height)
- Center crop around segmentation boundary
- Handle edge cases: if crop exceeds image borders, fill with non-solid random colors
- Output size determined automatically (not fixed to specific dimensions)
- Maintain aspect ratio considerations

### 3. Label Transformation
- Transform polygon coordinates from original image space to crop space
- Normalize new coordinates relative to crop dimensions
- Validate transformed coordinates (ensure they're within [0,1] range)
- Handle coordinate clipping if necessary

### 4. Image Processing
- Load images using OpenCV
- Apply square cropping with proper padding if needed
- Fill border areas with non-solid random colors when crop exceeds image boundaries
- Save cropped images in appropriate format
- Maintain image quality and metadata

## Implementation Plan

### Phase 1: Core Square Cropper Class
Create `SquareSegmentationCropper` class in `app/yolo_augmenter/`:
- **File**: `square_segmentation_cropper.py`
- **Responsibilities**:
  - Parse YOLO segmentation labels
  - Calculate optimal square crop dimensions
  - Apply buffer multiplier
  - Transform coordinates
  - Generate cropped images and labels

### Phase 2: Configuration Integration
- Extend existing configuration system
- Add square crop parameters (buffer multiplier, output size, etc.)
- Integrate with existing augmenter pipeline

### Phase 3: Batch Processing Pipeline
- Create batch processing script with CLI interface
- Handle any directory structure with images+labels format
- Auto-discovery of available datasets
- Progress tracking and error handling
- Output organization
- CLI argument parsing with sensible defaults

### Phase 4: Quality Assurance
- Validation of transformed labels
- Visual verification tools
- Statistics generation
- Error reporting

## File Structure
```
app/yolo_augmenter/
├── square_segmentation_cropper.py      # Main cropper class
├── square_crop_pipeline.py            # Batch processing script with CLI
├── square_crop_utils.py               # Utility functions
└── __main__.py                        # CLI entry point for module execution

data/processed/
├── mauritius_square_crops/            # Output for Mauritius dataset
│   ├── images/
│   └── labels/
└── rous_square_crops/                 # Output for Rousettus dataset
    ├── images/
    └── labels/
```

## Key Features

### 1. Buffer Management
- **Configurable Buffer**: 1.1x multiplier (configurable)
- **Smart Padding**: Handle edge cases near image boundaries with random color filling
- **Dynamic Sizing**: Output size determined automatically based on segmentation boundaries

### 2. Coordinate Transformation
- **Accurate Mapping**: Precise coordinate transformation
- **Validation**: Ensure coordinates remain within valid ranges
- **Error Handling**: Graceful handling of edge cases

### 3. Batch Processing
- **Progress Tracking**: Real-time progress updates
- **Error Recovery**: Continue processing on individual failures
- **Logging**: Comprehensive logging for debugging

### 4. Quality Control
- **Label Validation**: Verify transformed labels are correct
- **Image Quality**: Maintain original image quality
- **Statistics**: Generate processing statistics

## Configuration Parameters
```yaml
square_crop:
  buffer_multiplier: 1.1
  output_format: "jpg"
  quality: 95
  padding_strategy: "random_colors"  # fill border areas with random colors
```

## Usage Examples

### CLI Usage (Primary Interface)
```bash
# Basic usage with defaults
python -m app.yolo_augmenter.square_crop_pipeline

# Custom parameters
python -m app.yolo_augmenter.square_crop_pipeline \
    --input-dir data/interim/mauritius_images+labels \
    --output-dir data/processed/mauritius_square_crops \
    --buffer-multiplier 1.2 \
    --output-format png \
    --quality 90

# Process specific directory
python -m app.yolo_augmenter.square_crop_pipeline \
    --input-dir data/interim/mauritius_images+labels \
    --buffer-multiplier 1.1

# Process all detected datasets (auto-discovery)
python -m app.yolo_augmenter.square_crop_pipeline \
    --buffer-multiplier 1.1
```

### CLI Arguments with Defaults
```bash
--input-dir, -i      # Input directory (default: auto-detect from config)
--output-dir, -o     # Output directory (default: auto-generate in data/processed/)
--buffer-multiplier  # Buffer around segmentation (default: 1.1)
--output-format     # Output image format (default: jpg)
--quality           # Image quality for lossy formats (default: 95)
--auto-discover     # Auto-discover and process all available datasets (default: True)
--verbose           # Verbose output (default: False)
--dry-run           # Show what would be processed without doing it (default: False)
```

### Programmatic Usage
```python
from app.yolo_augmenter.square_segmentation_cropper import SquareSegmentationCropper

cropper = SquareSegmentationCropper(
    input_dir="data/interim/mauritius_images+labels",
    output_dir="data/processed/mauritius_square_crops",
    buffer_multiplier=1.1
)

cropper.process_dataset()
```

### Batch Processing
```python
from app.yolo_augmenter.square_crop_pipeline import SquareCropPipeline

pipeline = SquareCropPipeline()
pipeline.process_all_datasets()
```

## Expected Output
- **Square Images**: All output images will be square with consistent dimensions
- **Transformed Labels**: YOLO segmentation labels properly transformed to new coordinate system
- **Organized Structure**: Clean directory structure matching input organization
- **Processing Logs**: Detailed logs for verification and debugging

## Benefits
1. **Consistent Training**: Square images improve YOLO training consistency
2. **Better Performance**: Optimized crop sizes can improve model performance
3. **Data Augmentation**: Creates additional training variants
4. **Standardization**: Normalizes dataset format for better training

## Implementation Testing Strategy

### Testing Requirements
- **After each implementation step, create a test script to verify functionality**
- **Test script must pass all tests before proceeding to next phase**
- **Delete test file after successful completion and move to next phase**
- **If tests fail, solve issues until they pass before continuing**

### Test Scripts to Create
1. **Phase 1 Test**: `test_square_cropper_basic.py` - Test core cropping functionality
2. **Phase 2 Test**: `test_config_integration.py` - Test configuration system integration
3. **Phase 3 Test**: `test_batch_pipeline.py` - Test batch processing capabilities
4. **Phase 4 Test**: `test_quality_assurance.py` - Test validation and verification tools

## Next Steps
1. Implement core `SquareSegmentationCropper` class
   - Create test script: `test_square_cropper_basic.py`
   - Ensure all tests pass
   - Delete test file and proceed to next phase
2. Create configuration integration
   - Create test script: `test_config_integration.py`
   - Ensure all tests pass
   - Delete test file and proceed to next phase
3. Develop batch processing pipeline
   - Create test script: `test_batch_pipeline.py`
   - Ensure all tests pass
   - Delete test file and proceed to next phase
4. Add quality assurance tools
   - Create test script: `test_quality_assurance.py`
   - Ensure all tests pass
   - Delete test file and proceed to next phase
5. Test with sample data
6. Process full datasets

## Dependencies
- OpenCV (cv2)
- NumPy
- Pathlib
- Existing YOLO augmenter infrastructure
- Configuration management system
