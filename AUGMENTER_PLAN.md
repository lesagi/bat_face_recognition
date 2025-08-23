# Image Augmenter Implementation Plan

## Overview

This document outlines the plan for creating a modular, flexible image augmentation system using albumentations that can handle both plain images and YOLO-labeled images. The system will be designed to be easily integrated with the existing `yolo_augmenter` module.

## Architecture Goals

1. **Modularity**: Separate augmentation logic from data processing
2. **Flexibility**: Support multiple input types (plain images, YOLO detection, YOLO segmentation)
3. **Configurability**: Easy to customize augmentation pipelines
4. **Reusability**: Core augmenter can be used independently
5. **Extensibility**: Easy to add new augmentation types and data formats

## System Components

### 1. Core Augmenter (`BaseAugmenter`)
- **Purpose**: Base class providing common augmentation functionality
- **Responsibilities**:
  - Transform pipeline management
  - Image loading/saving
  - Basic augmentation execution
  - Configuration management
  - Error handling and logging

### 2. Specialized Augmenters

#### 2.1 Plain Image Augmenter (`PlainImageAugmenter`)
- **Purpose**: Augment images without any labels
- **Input**: Directory of images
- **Output**: Augmented images with naming convention
- **Features**:
  - Batch processing
  - Configurable naming patterns
  - Support for various image formats

#### 2.2 YOLO Detection Augmenter (`YoloDetectionAugmenter`)
- **Purpose**: Augment images with YOLO detection labels (bounding boxes)
- **Input**: YOLO dataset structure with detection labels
- **Output**: Augmented images + transformed labels
- **Features**:
  - Bounding box transformation
  - Label format preservation
  - Coordinate validation

#### 2.3 YOLO Segmentation Augmenter (`YoloSegmentationAugmenter`)
- **Purpose**: Augment images with YOLO segmentation labels (current implementation)
- **Input**: YOLO dataset structure with segmentation labels
- **Output**: Augmented images + transformed masks + transformed labels
- **Features**:
  - Mask transformation
  - Segmentation coordinate transformation
  - Bounding box recalculation from masks

### 3. Augmentation Pipeline Factory (`AugmentationPipelineFactory`)
- **Purpose**: Create appropriate augmentation pipelines based on data type
- **Responsibilities**:
  - Pipeline selection logic
  - Configuration-based pipeline creation
  - Custom pipeline support

### 4. Configuration System
- **Purpose**: Centralized configuration management
- **Components**:
  - Default augmentation parameters
  - Pipeline presets (light, medium, heavy)
  - Custom pipeline definitions
  - Data type-specific settings

## Implementation Plan

### Phase 1: Core Infrastructure
1. **Create `BaseAugmenter` class**
   - Abstract base class with common methods
   - Transform pipeline management
   - Basic image I/O operations
   - Configuration loading

2. **Create `AugmentationPipelineFactory`**
   - Pipeline creation logic
   - Preset pipeline definitions
   - Custom pipeline support

3. **Create configuration structure**
   - YAML configuration schema
   - Default parameter sets
   - Pipeline presets

### Phase 2: Plain Image Augmenter
1. **Implement `PlainImageAugmenter`**
   - Inherit from `BaseAugmenter`
   - Simple image-only augmentation
   - Batch processing capabilities
   - Naming convention management

2. **Create CLI interface for plain images**
   - Use Click package for command-line interface
   - Progress tracking
   - Output validation

### Phase 3: YOLO Detection Augmenter
1. **Implement `YoloDetectionAugmenter`**
   - Inherit from `BaseAugmenter`
   - Bounding box transformation
   - Label file parsing/writing
   - Coordinate validation

2. **Integration with existing system**
   - Update `yolo_augmenter` to use new system
   - Maintain backward compatibility
   - Add detection support

### Phase 4: Refactor Existing Segmentation Augmenter
1. **Refactor `YoloSegmentationAugmenter`**
   - Inherit from `BaseAugmenter`
   - Use new pipeline factory
   - Maintain existing functionality
   - Improve code organization

2. **Update CLI and interfaces**
   - Unified command-line interface using Click
   - Data type detection
   - Appropriate augmenter selection

### Phase 5: Advanced Features
1. **Add pipeline presets**
   - Light augmentation (minimal changes)
   - Medium augmentation (balanced)
   - Heavy augmentation (aggressive)
   - Custom augmentation (user-defined)

## File Structure

```
app/yolo_augmenter/
├── __init__.py
├── base/
│   ├── __init__.py
│   ├── base_augmenter.py          # Base augmenter class
│   ├── pipeline_factory.py        # Pipeline creation factory
│   └── config.py                  # Configuration management
├── augmenters/
│   ├── __init__.py
│   ├── plain_image_augmenter.py   # Plain image augmentation
│   ├── detection_augmenter.py     # YOLO detection augmentation
│   └── segmentation_augmenter.py  # Refactored segmentation augmenter
├── pipelines/
│   ├── __init__.py
│   ├── presets.py                 # Predefined augmentation pipelines
│   └── custom.py                  # Custom pipeline support
├── cli.py                         # Click-based CLI interface
└── utils/
    ├── __init__.py
    ├── image_utils.py             # Image processing utilities
    ├── label_utils.py             # Label processing utilities
    └── validation.py              # Input validation utilities
```

## Configuration Schema

```yaml
augmentation:
  # Global settings
  output_naming: "_a-{index:03d}"
  
  # Pipeline presets
  presets:
    light:
      geometric_p: 0.3
      color_p: 0.2
      noise_p: 0.1
    medium:
      geometric_p: 0.6
      color_p: 0.5
      noise_p: 0.3
    heavy:
      geometric_p: 0.8
      color_p: 0.7
      noise_p: 0.5
  
  # Custom pipelines
  custom_pipelines:
    my_pipeline:
      transforms:
        - type: "Rotate"
          limit: 15
          p: 0.8
        - type: "ColorJitter"
          brightness: 0.1
          contrast: 0.1
          p: 0.6
  
  # Data type specific settings
  yolo_detection:
    bbox_min_visibility: 0.3
    bbox_format: "yolo"
  
  yolo_segmentation:
    mask_interpolation: "nearest"
    segmentation_simplification: 0.005
```

## Systematic Transformation System

Instead of random transformations, the augmenter will create systematic, gradual transformations that cover the full parameter range in defined steps. This ensures comprehensive coverage of the transformation space.

### How It Works

1. **Parameter Steps**: For each parameter, create steps evenly spaced from 0 to the parameter's maximum value
2. **Custom vs Default Steps**: Each parameter can specify its own `steps` value, or use `default_steps_per_attribute` as fallback
3. **Permutation Generation**: Generate all possible combinations of these parameter values
4. **Systematic Coverage**: Each augmentation represents a specific combination of parameter values

### Examples

#### Example 1: Simple Rotation with Default Steps
```yaml
presets:
  light:
    rotation:
      limit: 15
    default_steps_per_attribute: 3
```

**Generated Steps:**
- Rotation: [0°, 7.5°, 15°] (3 steps from 0 to 15, using default)
- **Total Augmentations**: 3 (one for each rotation value)

#### Example 2: Mixed Custom and Default Steps
```yaml
presets:
  medium:
    rotation:
      limit: 20
      steps: 4
    brightness:
      limit: 0.3
    default_steps_per_attribute: 3
```

**Generated Steps:**
- Rotation: [0°, 6.67°, 13.33°, 20°] (4 steps from 0 to 20, custom)
- Brightness: [0.0, 0.15, 0.3] (3 steps from 0 to 0.3, using default)

**Total Augmentations**: 4 × 3 = 12 (all combinations)
- Aug 0: Rotation=0°, Brightness=0.0
- Aug 1: Rotation=0°, Brightness=0.15
- Aug 2: Rotation=0°, Brightness=0.3
- Aug 3: Rotation=6.67°, Brightness=0.0
- Aug 4: Rotation=6.67°, Brightness=0.15
- Aug 5: Rotation=6.67°, Brightness=0.3
- Aug 6: Rotation=13.33°, Brightness=0.0
- Aug 7: Rotation=13.33°, Brightness=0.15
- Aug 8: Rotation=13.33°, Brightness=0.3
- Aug 9: Rotation=20°, Brightness=0.0
- Aug 10: Rotation=20°, Brightness=0.15
- Aug 11: Rotation=20°, Brightness=0.3

#### Example 3: Complex Pipeline with Mixed Steps
```yaml
presets:
  heavy:
    rotation:
      limit: 30
      steps: 10
    scale:
      range: [0.7, 1.3]
    brightness:
      limit: 0.4
      steps: 5
    contrast:
      limit: 0.4
    default_steps_per_attribute: 4
```

**Generated Steps:**
- Rotation: [0°, 3.33°, 6.67°, 10°, 13.33°, 16.67°, 20°, 23.33°, 26.67°, 30°] (10 steps from 0 to 30, custom)
- Scale: [0.7, 0.9, 1.1, 1.3] (4 steps from 0.7 to 1.3, using default)
- Brightness: [0.0, 0.1, 0.2, 0.3, 0.4] (5 steps from 0 to 0.4, custom)
- Contrast: [0.0, 0.13, 0.27, 0.4] (4 steps from 0 to 0.4, using default)

**Total Augmentations**: 10 × 4 × 5 × 4 = 800 (all combinations)

### Benefits of Systematic Approach

1. **Predictable Output**: Exact number of augmentations based on parameter count and steps
2. **Comprehensive Coverage**: Every possible combination of parameter values is explored
3. **No Randomness**: Reproducible results across different runs
4. **Efficient Training**: Systematic variation ensures model sees all parameter combinations
5. **Easy Debugging**: Each augmentation has a known, predictable transformation
6. **Flexible Control**: Custom steps per parameter or fallback to defaults

### Implementation Details

```python
def generate_parameter_combinations(self, preset_config):
    """Generate all parameter combinations based on preset configuration."""
    default_steps = preset_config.get('default_steps_per_attribute', 4)
    combinations = []
    
    for param_name, param_config in preset_config.items():
        if param_name != 'default_steps_per_attribute':
            # Check if parameter has custom steps or use default
            if isinstance(param_config, dict) and 'steps' in param_config:
                steps = param_config['steps']
                # Extract the actual parameter value
                if 'limit' in param_config:
                    param_value = param_config['limit']
                elif 'range' in param_config:
                    param_value = param_config['range']
                else:
                    continue
            else:
                # Simple parameter, use default steps
                steps = default_steps
                param_value = param_config
            
            # Generate steps from 0 to param_value
            if isinstance(param_value, list):  # Range parameter
                param_steps = np.linspace(param_value[0], param_value[1], steps)
            else:  # Single value parameter
                param_steps = np.linspace(0, param_value, steps)
            
            combinations.append((param_name, param_steps))
    
    # Generate all permutations
    from itertools import product
    param_names = [name for name, _ in combinations]
    param_values = [values for _, values in combinations]
    
    all_combinations = []
    for combo in product(*param_values):
        all_combinations.append(dict(zip(param_names, combo)))
    
    return all_combinations
```

## CLI Interface Design (Using Click)

```bash
# Auto-detect data type (default behavior)
python -m app.yolo_augmenter --input ./dataset --output ./augmented

# Explicitly specify data type
python -m app.yolo_augmenter plain --input ./images --output ./augmented
python -m app.yolo_augmenter detection --input ./dataset --output ./augmented
python -m app.yolo_augmenter segmentation --input ./dataset --output ./augmented

# With preset pipeline
python -m app.yolo_augmenter --input ./images --output ./augmented --preset heavy
python -m app.yolo_augmenter plain --input ./images --output ./augmented --preset heavy

# With custom pipeline
python -m app.yolo_augmenter --input ./images --output ./augmented --pipeline my_pipeline
python -m app.yolo_augmenter plain --input ./images --output ./augmented --pipeline my_pipeline
```

## Benefits of New Architecture

1. **Maintainability**: Clear separation of concerns
2. **Reusability**: Core augmenter can be used independently
3. **Extensibility**: Easy to add new data types and augmentation types
4. **Systematic Coverage**: Comprehensive parameter space exploration
5. **Configuration**: Centralized configuration management
6. **Performance**: Optimized pipeline creation and execution
7. **User Experience**: Consistent interface across different data types
8. **Predictability**: Exact number of augmentations based on configuration

## Migration Strategy

1. **Phase 1**: Implement new system alongside existing one
2. **Phase 2**: Gradually migrate existing functionality
3. **Phase 3**: Deprecate old implementation
4. **Phase 4**: Remove old code and complete migration

## Next Steps

1. Review and approve this updated plan
2. Start with Phase 1 (Core Infrastructure)
3. Implement `BaseAugmenter` class
4. Create `AugmentationPipelineFactory`
5. Set up configuration system
6. Begin implementation of systematic transformation system

## Questions for Discussion

1. Should we maintain 100% backward compatibility with existing CLI?
2. What are the priority augmentation types for your use case?
3. Do you have specific performance requirements?
4. Should we support other label formats beyond YOLO?
5. What level of customization do you need for augmentation pipelines?
6. Are the examples clear enough to show the systematic transformation approach?
