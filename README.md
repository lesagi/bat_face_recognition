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

### GPU Setup Issues (Linux)

If TensorFlow is not detecting your GPU(s) despite having NVIDIA drivers and CUDA installed, follow these steps:

#### Problem: TensorFlow shows "GPU devices: []" despite having GPUs

**Symptoms:**
```bash
TensorFlow version: 2.15.0
Built with CUDA: True
GPU devices: []
```

**Root Cause:** Missing or misconfigured cuDNN library in the system's linker cache.

#### Solution: Install cuDNN in Conda Environment

1. **Verify GPU and CUDA installation:**
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Check CUDA version (should show 12.4 or compatible)
   nvcc --version
   ```

2. **Create diagnostic script** to identify missing libraries:
   ```bash
   # Create check_cuda_setup.sh
   cat > check_cuda_setup.sh << 'EOF'
   #!/bin/bash
   OUTPUT_FILE="cuda_setup_info.txt"
   echo "=== CUDA Setup Information ===" > $OUTPUT_FILE
   echo "Generated on: $(date)" >> $OUTPUT_FILE
   
   echo "[1/5] Checking CUDA compiler version..."
   nvcc --version >> $OUTPUT_FILE 2>&1
   
   echo "[2/5] Checking nvidia-smi..."
   nvidia-smi >> $OUTPUT_FILE 2>&1
   
   echo "[3/5] Checking system CUDA libraries..."
   ldconfig -p | grep -E "libcudart|libcublas|libcudnn" >> $OUTPUT_FILE 2>&1
   
   echo "[4/5] Checking TensorFlow GPU detection..."
   python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))" >> $OUTPUT_FILE 2>&1
   
   echo "[5/5] Checking for missing libraries..."
   for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcudnn.so.8; do
       echo "--- $lib ---" >> $OUTPUT_FILE
       ldconfig -p | grep "$lib" >> $OUTPUT_FILE 2>&1
       if [ $? -ne 0 ]; then
           echo "MISSING: $lib not found in ldconfig cache" >> $OUTPUT_FILE
       fi
   done
   
   echo "✅ Done! Check $OUTPUT_FILE for results"
   EOF
   
   chmod +x check_cuda_setup.sh
   ./check_cuda_setup.sh
   ```

3. **Identify missing library** (typically `libcudnn.so.8`):
   ```bash
   cat cuda_setup_info.txt | grep "MISSING"
   ```

4. **Install cuDNN via conda** (this is the key fix):
   ```bash
   # Activate your conda environment
   conda activate your_env_name
   
   # Install cuDNN compatible with your CUDA version
   # For CUDA 12.4, use cuDNN 8.9
   conda install cudnn=8.9 -c conda-forge

   # Install missing cude-toolkit
   conda install cuda-toolkit=11.7 -c nvidia
   ```

5. **Update LD_LIBRARY_PATH** to include conda libraries:
   ```bash
   # Add to your ~/.bashrc or activate script
   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
   
   # Reload the environment
   source ~/.bashrc
   # Or reactivate conda environment
   conda deactivate
   conda activate your_env_name
   ```

6. **Verify GPU detection:**
   ```bash
   python -c "import tensorflow as tf; print('Built with CUDA:', tf.test.is_built_with_cuda()); print('GPU devices:', tf.config.list_physical_devices('GPU'))"
   ```

   **Expected output** (success):
   ```
   Built with CUDA: True
   2025-10-12 01:21:43.509650: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22476 MB memory:  -> device: 0, name: NVIDIA RTX A5000, pci bus id: 0000:5e:00.0, compute capability: 8.6
   GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
   ```

#### TensorFlow-CUDA-cuDNN Compatibility Matrix

| TensorFlow | CUDA | cuDNN | Python |
|------------|------|-------|--------|
| 2.15.0     | 12.4 | 8.9   | 3.9-3.11 |
| 2.12.0     | 11.8 | 8.6   | 3.9-3.11 |
| 2.10.0     | 11.2 | 8.1   | 3.7-3.10 |

#### Additional Notes

- **Warning messages** like "Unable to register cuDNN factory: Attempting to register factory..." are harmless and can be ignored.
- **TensorRT warnings** ("Could not find TensorRT") are optional and don't affect GPU functionality.
- **System-wide vs Conda cuDNN**: Installing cuDNN in the conda environment is easier and doesn't require sudo access.
- **Multiple CUDA versions**: Having multiple CUDA versions (e.g., 10.1 system-wide, 12.4 for TensorFlow) is fine as long as `LD_LIBRARY_PATH` points to the correct version.

#### Verification Checklist

- [ ] `nvidia-smi` shows your GPU(s)
- [ ] `nvcc --version` shows compatible CUDA version
- [ ] `conda list | grep cudnn` shows cuDNN installed in environment
- [ ] `echo $LD_LIBRARY_PATH` includes `$CONDA_PREFIX/lib`
- [ ] TensorFlow imports without errors
- [ ] `tf.config.list_physical_devices('GPU')` returns your GPU(s)
- [ ] Training actually uses GPU (check with `nvidia-smi` during training)

### Performance Tips

- Use SSD storage for faster I/O
- Process datasets in smaller batches if memory is limited
- Use `--dry-run` to preview operations before processing
- Enable verbose output with `-v` for detailed logging
- **GPU Training**: Ensure GPU is detected before training (see GPU Setup Issues above)

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
