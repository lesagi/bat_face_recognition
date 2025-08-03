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

## Directory Structure

Your input directory should be organized as follows:

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
   - Reduce the `--sample_size` parameter
   - Consider using a smaller batch of images

4. **Import Errors**
   ```
   ModuleNotFoundError
   ```
   - Ensure you're running the script from the correct directory
   - Check that all dependencies are installed

### Getting Help

If you encounter issues:

1. Check that your directory structure matches the expected format
2. Verify that your model path is correct
3. Ensure all required dependencies are installed
4. Try with a smaller sample size first

## Technical Details

### How Saliency Maps are Generated

1. **Gradient Calculation**: Uses TensorFlow's GradientTape to compute gradients of the model output with respect to the input image
2. **Attention Visualization**: Highlights regions where changes in pixel values would most affect the model's prediction
3. **Random Pairing**: Each image is paired with a random counterpart to generate the saliency map
4. **Visualization**: Uses matplotlib with a "hot" colormap to display attention intensity

### Performance Considerations

- Processing time depends on:
  - Number of images (sample_size parameter)
  - Model complexity
  - Available computational resources
- Large datasets may require significant processing time
- Consider using GPU acceleration for faster processing

## Configuration

You can modify the default behavior by editing the configuration values in `app/visualization/config.py`:

```python
MODEL_PATH = "path/to/your/default/model"
INPUT_DIR = "path/to/your/default/input"
OUTPUT_DIR = "path/to/your/default/output"
SAMPLE_SIZE = "50"
FIGURE_SIZE = ('15.0', '3.0')
```

## Related Files

- `saliency.py`: Core saliency map generation functionality
- `config.py`: Configuration settings
- `run_saliency_maps.py`: Command-line interface script

## License

This module is part of the bat face recognition project and follows the same licensing terms as the main project. 