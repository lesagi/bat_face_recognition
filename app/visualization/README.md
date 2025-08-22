# Saliency Maps for Siamese Network Analysis

This module provides functionality to generate saliency maps for trained Siamese networks, helping visualize which parts of input images the model focuses on when making predictions.

## Overview

Saliency maps are visual representations that highlight the regions in an input image that have the most influence on the model's decision. In the context of Siamese networks used for face recognition, saliency maps help understand:

- Which facial features the model considers most important
- How the model's attention varies across different individuals
- Whether the model is focusing on meaningful biological features or artifacts

## Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Matplotlib
- NumPy
- A trained Siamese model (in TensorFlow SavedModel format)

## Input Directory Structure

The script supports two directory structures:

### 1. Nested Directory Structure (Original)
```
input_directory/
├── individual_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── individual_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

Or with nested structure:

```
input_directory/
├── validation/
│   ├── individual_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── individual_2/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── ...
└── test/
    ├── individual_1/
    └── ...
```

### 2. Flat Directory Structure (New)
```
input_directory/
├── individual_1--image1.jpg
├── individual_1--image2.jpg
├── individual_1--image3.jpg
├── individual_2--image1.jpg
├── individual_2--image2.jpg
└── ...
```

Or with nested structure:

```
input_directory/
├── validation/
│  ├── individual_1--image1.jpg
│  ├── individual_1--image2.jpg
│  ├── individual_2--image1.jpg
│  ├── individual_2--image2.jpg
│  └── ...
├── test/
│  ├── individual_1--image1.jpg
│  ├── individual_1--image2.jpg
│  ├── individual_2--image1.jpg
│  ├── individual_2--image2.jpg
│  └── ...
```

**Note:** The script automatically detects which structure you're using. For the flat structure, individual names are extracted from the filename prefix before the double dash (`--`).

## Usage

### Basic Usage

```bash
python run_saliency_maps.py --input_dir /path/to/your/images --model_path /path/to/trained/model
```

### Advanced Usage

```bash
# Specify custom output directory
python run_saliency_maps.py \
    --input_dir /path/to/your/images \
    --model_path /path/to/trained/model \
    --output_dir /path/to/save/saliency/maps

# Use nested directory structure
python run_saliency_maps.py \
    --input_dir /path/to/your/images \
    --model_path /path/to/trained/model \
    --nesting validation \
    --sample_size 25
```

### Command Line Arguments

| Argument        | Required | Description                                                             |
| --------------- | -------- | ----------------------------------------------------------------------- |
| `--input_dir`   | Yes      | Path to input directory containing subdirectories with images           |
| `--model_path`  | Yes      | Path to trained Siamese model directory                                 |
| `--output_dir`  | No       | Path to output directory for saliency maps (default: same as input_dir) |
| `--nesting`     | No       | Nested subdirectory name within input_dir (e.g., 'validation', 'test')  |
| `--sample_size` | No       | Number of images to sample for saliency map generation (default: 50)    |

## Examples

### Example 1: Basic Face Recognition Dataset

```bash
# For a dataset with bat face images organized by individual
python run_saliency_maps.py \
    --input_dir /Users/yourname/bat_faces/unseen_data \
    --model_path /Users/yourname/models/siamesemodelv2_v40
```

### Example 2: Using Available Models

```bash
# Using one of the pre-trained models from the project
python run_saliency_maps.py \
    --input_dir ../face_recognition-with_bg_31_03_25/data/unseen \
    --model_path ../face_rec_rousettus_#1/runs/model/siamesemodelv2_v80
```

### Example 3: Nested Directory Structure

```bash
# For datasets with train/validation/test splits
python run_saliency_maps.py \
    --input_dir /path/to/face_recognition/data \
    --model_path /path/to/trained/model \
    --nesting validation \
    --output_dir /path/to/analysis/results
```

## Output

The script generates:

1. **PDF Report**: A comprehensive PDF file containing:
   - Original images
   - Corresponding saliency maps
   - Image paths as titles
   - Hot colormap visualization for attention areas

2. **Console Output**: Progress information and file locations

The output PDF will be named `{input_directory_name}_saliency_map.pdf` and saved in the specified output directory.

## Available Pre-trained Models

Based on the current project structure, you can use these pre-trained models:

```bash
# Available models in face_rec_rousettus_#1/runs/model/
- siamesemodelv2_v5/
- siamesemodelv2_v10/
- siamesemodelv2_v20/
- siamesemodelv2_v40/
- siamesemodelv2_v60/
- siamesemodelv2_v80/
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```
   Error: Could not load model
   ```
   - Ensure the model path points to a valid TensorFlow SavedModel directory
   - Check that the model was trained with the same network architecture

2. **No Images Found**
   ```
   Error: No images found in any subdirectory
   ```
   - Verify your input directory contains subdirectories with images
   - Ensure image files have common extensions (.jpg, .jpeg, .png, .bmp, .tiff)

3. **Memory Issues**
   ```
   OutOfMemoryError
   ```