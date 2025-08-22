# YOLO Segmentation Data Augmenter

A powerful data augmentation tool specifically designed for YOLO segmentation datasets. This module provides comprehensive augmentation capabilities while properly handling segmentation masks and bounding boxes.

## Features

- **Segmentation-aware augmentation**: Properly transforms both images and segmentation masks
- **YOLO format support**: Native support for YOLO segmentation label format
- **Configurable transforms**: Geometric, color, noise, and blur transformations
- **CLI interface**: Easy-to-use command-line interface
- **Config integration**: Uses app/config system when available
- **Batch processing**: Efficient processing of entire datasets
- **Custom naming**: Augmented files get `_a-{index}` postfix

## Installation

Required dependencies:
```bash
pip install opencv-python albumentations numpy pyyaml tqdm click
```

## Usage

### Command Line Interface

Basic usage:
```bash
python -m app.yolo_augmenter --input ./data/original --output ./data/augmented
```

With custom augmentation count:
```bash
python -m app.yolo_augmenter --input ./data/original --output ./data/augmented --count 20
```

Check dependencies:
```bash
python -m app.yolo_augmenter --check-deps
```

Force overwrite existing output:
```bash
python -m app.yolo_augmenter --input ./data/original --output ./data/augmented --force
```

### Programmatic Usage

```python
from app.yolo_augmenter import YoloSegmentationAugmenter

# Create augmenter
augmenter = YoloSegmentationAugmenter(
    source_dir="./data/original",
    target_dir="./data/augmented", 
    augmentations_per_image=10
)

# Run augmentation
results = augmenter.augment_dataset()
print(f"Augmentation results: {results}")
```

## Input Directory Structure

Your input directory should follow the standard YOLO dataset structure:

```
input_dir/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── val/ (optional)
    ├── images/
    │   └── ...
    └── labels/
        └── ...
```

## Output Structure

The augmented dataset will be created with the following structure:

```
output_dir/
├── train/
│   ├── images/
│   │   ├── image1_a-000.jpg
│   │   ├── image1_a-001.jpg
│   │   ├── image1_a-002.jpg
│   │   └── ...
│   └── labels/
│       ├── image1_a-000.txt
│       ├── image1_a-001.txt
│       ├── image1_a-002.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

## Augmentation Types

The augmenter applies the following transformations:

### Geometric Transformations
- **Rotation**: ±25 degrees
- **Scaling**: 0.8x to 1.2x
- **Translation**: ±10% of image size
- **Shearing**: ±8 degrees
- **Perspective**: Slight perspective distortion

### Color Transformations
- **Brightness/Contrast**: ±20%
- **Hue/Saturation**: Moderate shifts
- **RGB shifts**: Channel-wise color adjustments

### Noise & Blur
- **Gaussian noise**: Light noise addition
- **ISO noise**: Camera sensor noise simulation
- **Motion blur**: Slight motion blur effects
- **Sharpening**: Light sharpening effects

### Weather Effects
- **Shadows**: Subtle shadow effects (10% probability)

### Scale & Crop Variations (New!)
- **Tight Crops**: Simulate closer/tighter cropping (70-90% of image)
- **Loose Crops**: Simulate wider/looser cropping (40-70% of image)  
- **Multi-Scale Resize**: Train on different input sizes (480px, 640px, 800px)
- **Aspect Ratio**: Maintain roughly square ratios (0.9-1.1)

## Configuration

The module integrates with the app/config system. You can customize augmentation settings in `app/config/config.yml`:

```yaml
yolo_segmentation:
  augmentation:
    default_augmentations_per_image: 10
    postfix_format: "_a-{index:03d}"
    geometric_transforms:
      rotation_limit: 25
      scale_range: [0.8, 1.2]
      # ... more settings
```

## CLI Options

```
--input, -i     Input directory (required)
--output, -o    Output directory (required)  
--count, -c     Augmentations per image (1-1000, default: 10)
--verbose, -v   Enable verbose output
--check-deps    Check dependencies and exit
--force, -f     Overwrite output directory without prompting
--help, -h      Show help message
```

## Examples

### Basic Augmentation
```bash
python -m app.yolo_augmenter -i ./dataset/original -o ./dataset/augmented
```

### Heavy Augmentation
```bash
python -m app.yolo_augmenter -i ./dataset/original -o ./dataset/augmented -c 50
```

### With Verbose Output
```bash
python -m app.yolo_augmenter -i ./dataset/original -o ./dataset/augmented -v
```

### Force Overwrite
```bash
python -m app.yolo_augmenter -i ./dataset/original -o ./dataset/augmented --force
```

## File Naming

Augmented files follow this naming pattern:
- Original: `image1.jpg` → Augmented: `image1_a-000.jpg`, `image1_a-001.jpg`, etc.
- The index is zero-padded to 3 digits
- Original file extension is preserved

## Training with Augmented Data

After augmentation, you can train your YOLO model using:

```bash
yolo segment train data=./data/augmented/dataset.yaml model=yolov8n-seg.pt epochs=100
```

## Notes

- The augmenter preserves segmentation mask accuracy through proper coordinate transformations
- Bounding boxes are automatically recalculated from transformed segmentation masks
- Failed augmentations are skipped and reported
- The tool creates a `dataset.yaml` file compatible with YOLO training

### Scale & Crop Benefits

The new scale/crop transformations help create a more robust model by:
- **Training on various margins**: Helps model work with both tight and loose crops
- **Multi-scale training**: Improves detection at different object sizes
- **Real-world variation**: Simulates different cropping scenarios in production
- **Better generalization**: Reduces overfitting to specific crop margins

## Troubleshooting

### Common Issues

1. **"No images found"**: Check that your images have supported extensions (.jpg, .jpeg, .png, .bmp, .tiff)
2. **"No labels found"**: Ensure label files have the same name as image files but with .txt extension
3. **Import errors**: Install missing dependencies with `pip install opencv-python albumentations click`

### Performance Tips

- Use SSD storage for faster I/O
- Reduce augmentation count for faster processing
- Use smaller image sizes if memory is limited