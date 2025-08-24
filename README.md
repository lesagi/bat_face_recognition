# Bat Face Recognition Project

A comprehensive computer vision project for bat face detection, segmentation, and recognition using YOLO models and Siamese networks.

## 🚀 Features

### Core Functionality
- **Face Detection**: YOLO-based face detection with high accuracy
- **Face Segmentation**: Precise segmentation masks for bat faces
- **Face Recognition**: Siamese network-based face identification
- **Data Augmentation**: Advanced augmentation pipeline for YOLO segmentation
- **Square Crop Processing**: Intelligent square cropping around segmentation boundaries

### Square Crop Tool (New!)
- **Smart Cropping**: Automatically crops square regions around YOLO segmentation boundaries
- **Buffer Management**: Configurable buffer multiplier (default: 1.1x) around segmentation
- **Edge Case Handling**: Random color padding when crops exceed image boundaries
- **Dynamic Sizing**: Output dimensions determined automatically by segmentation boundaries
- **Batch Processing**: Process multiple datasets with CLI interface
- **Quality Assurance**: Comprehensive validation and verification tools

## 📁 Project Structure

```
bat_face_rec/
├── app/
│   ├── config/                 # Configuration files
│   ├── yolo_augmenter/        # YOLO augmentation tools
│   │   ├── segmentation_augmenter.py    # Data augmentation
│   │   ├── square_segmentation_cropper.py  # Square crop tool
│   │   ├── square_crop_pipeline.py      # Batch processing CLI
│   │   ├── square_crop_utils.py         # Quality assurance tools
│   │   └── README.md                    # Augmenter documentation
│   ├── visualization/          # Visualization tools
│   └── ...
├── data/
│   ├── interim/               # Intermediate datasets
│   └── processed/             # Processed datasets
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Conda (recommended)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd bat_face_rec

# Create and activate conda environment
conda create -n bat_face_rec python=3.8
conda activate bat_face_rec

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies
```bash
pip install opencv-python albumentations numpy pyyaml tqdm click
```

## 🎯 Square Crop Tool Usage

### Command Line Interface

The square crop tool provides an intuitive CLI for processing YOLO segmentation datasets:

#### Basic Usage
```bash
# Process all datasets with defaults
python -m app.yolo_augmenter.square_crop_pipeline

# Process specific directory
python -m app.yolo_augmenter.square_crop_pipeline \
    -i data/interim/mauritius_images+labels \
    -o data/processed/mauritius_square_crops

# Custom parameters
python -m app.yolo_augmenter.square_crop_pipeline \
    --buffer-multiplier 1.2 \
    --output-format png \
    --quality 90

# Dry run to preview operations
python -m app.yolo_augmenter.square_crop_pipeline --dry-run
```

#### CLI Options
```
-i, --input-dir           Input directory containing images and labels
-o, --output-dir          Output directory for processed data
--buffer-multiplier       Buffer around segmentation (default: 1.1)
--output-format           Output image format: jpg, png, bmp, tiff (default: jpg)
--quality                 Image quality for lossy formats (default: 95)
--all-datasets            Process all available datasets
--base-dir                Base directory to search for datasets (default: data/interim)
--output-base-dir         Base directory for output (default: data/processed)
-v, --verbose             Enable verbose output
--dry-run                 Show what would be processed without doing it
```

#### Programmatic Usage
```python
from app.yolo_augmenter.square_segmentation_cropper import SquareSegmentationCropper

# Initialize cropper
cropper = SquareSegmentationCropper(
    input_dir="data/interim/mauritius_images+labels",
    output_dir="data/processed/mauritius_square_crops",
    buffer_multiplier=1.1,
    output_format="jpg",
    quality=95,
    verbose=True
)

# Process dataset
results = cropper.process_dataset()
print(f"Processing results: {results}")
```

### Input Directory Structure

The tool expects the standard YOLO dataset structure:

```
input_dir/
├── images/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

### Output Structure

Processed data follows the same structure with `_cropped` suffix:

```
output_dir/
├── images/
│   ├── image1_cropped.jpg
│   ├── image2_cropped.png
│   └── ...
└── labels/
    ├── image1_cropped.txt
    ├── image2_cropped.txt
    └── ...
```

## 🔧 Configuration

The square crop tool integrates with the existing configuration system. Customize settings in `app/config/config.yml`:

```yaml
yolo_segmentation:
  square_crop:
    default_buffer_multiplier: 1.1
    default_output_format: "jpg"
    default_quality: 95
    default_padding_strategy: "random_colors"
    auto_discover_datasets: true
```

## 📊 Quality Assurance

The tool includes comprehensive quality assurance features:

### Validation Tools
- **Label Format Validation**: Ensures YOLO segmentation labels are properly formatted
- **Image Validation**: Checks image properties and format
- **Coordinate Consistency**: Verifies label coordinates match image dimensions
- **Processing Statistics**: Generates comprehensive processing metrics
- **Validation Reports**: Creates detailed quality reports

### Usage
```python
from app.yolo_augmenter.square_crop_utils import (
    create_validation_report,
    verify_processing_completeness
)

# Generate validation report
report = create_validation_report(input_dir, output_dir, detailed=True)
print(report)

# Verify processing completeness
verification = verify_processing_completeness(input_dir, output_dir)
print(f"Processing complete: {verification['complete']}")
```

## 🎨 Data Augmentation

The project includes a powerful YOLO segmentation augmenter:

### Features
- **Segmentation-aware**: Properly transforms both images and segmentation masks
- **Multiple Transform Types**: Geometric, color, noise, blur, and weather effects
- **Batch Processing**: Efficient processing of entire datasets
- **CLI Interface**: Easy-to-use command-line interface

### Usage
```bash
# Basic augmentation
python -m app.yolo_augmenter -i ./dataset/original -o ./dataset/augmented

# Heavy augmentation
python -m app.yolo_augmenter -i ./dataset/original -o ./dataset/augmented -c 50
```

## 🚀 Training Pipeline

### 1. Data Preparation
```bash
# Process raw data with square cropping
python -m app.yolo_augmenter.square_crop_pipeline \
    -i data/interim/mauritius_images+labels \
    -o data/processed/mauritius_square_crops

# Augment the processed data
python -m app.yolo_augmenter \
    -i data/processed/mauritius_square_crops \
    -o data/augmented/mauritius_augmented
```

### 2. Model Training
```bash
# Train YOLO segmentation model
yolo segment train data=data/augmented/mauritius_augmented/dataset.yaml \
    model=yolov8n-seg.pt epochs=100
```

## 📈 Performance

### Square Crop Tool
- **Processing Speed**: ~100-500 images/minute (depending on image size and hardware)
- **Memory Usage**: Efficient memory management for large datasets
- **Quality**: Maintains segmentation accuracy through proper coordinate transformation
- **Scalability**: Handles datasets of any size with batch processing

### Key Benefits
- **Square Outputs**: Always generates square images for consistent training
- **Smart Padding**: Handles edge cases with random color padding
- **Label Preservation**: Maintains YOLO segmentation label accuracy
- **Batch Processing**: Efficient processing of multiple datasets
- **Quality Validation**: Comprehensive quality assurance tools

## 🐛 Troubleshooting

### Common Issues

1. **"Input directory does not exist"**: Check the path and ensure the directory exists
2. **"No image files found"**: Verify images are in the `images/` subdirectory
3. **"No label files found"**: Ensure labels are in the `labels/` subdirectory with `.txt` extension
4. **Import errors**: Install missing dependencies with `pip install -r requirements.txt`

### Performance Tips

- Use SSD storage for faster I/O
- Process datasets in smaller batches if memory is limited
- Use `--dry-run` to preview operations before processing
- Enable verbose output with `-v` for detailed logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLO community for the segmentation models
- Albumentations team for the augmentation library
- OpenCV contributors for computer vision tools

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the existing documentation
- Review the configuration examples

---

**Note**: This project is actively maintained and new features are regularly added. Check the repository for the latest updates and improvements.
