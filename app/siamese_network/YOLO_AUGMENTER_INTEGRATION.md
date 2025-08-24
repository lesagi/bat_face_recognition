# YOLO Augmenter Integration with Siamese Network Preprocessing

## Overview

This document explains how to integrate the YOLO Augmenter system with the existing `preprocess_single_image` function in the Siamese Network preprocessing pipeline. The goal is to add data augmentation capabilities during the preprocessing stage to create multiple augmented versions of each input image.

## Current Architecture

### Existing Pipeline (`preprocess_single_image`)
The current preprocessing pipeline follows these steps:
1. **Face Alignment** - Using YOLO pose landmarks
2. **Segmentation** - YOLO-based bat face detection
3. **Square Cropping** - Around segmented region with margin
4. **Background Replacement** - Using configurable background generators
5. **Resize** - To target size (224x224)
6. **Normalization** - Scale to [0,1] range

### YOLO Augmenter System
The augmenter system provides:
- **Configuration-driven augmentation** via YAML files
- **Multiple preset types** (light, medium, heavy, custom)
- **Transform combinations** with systematic parameter variations
- **Support for noise, color, and geometric transformations**

## Integration Approaches

### Approach 1: Post-Processing Augmentation (Recommended)
Add augmentation as a final step after the complete preprocessing pipeline.

**Pros:**
- Clean separation of concerns
- Augmentation works on fully processed, standardized images
- Easier to debug and maintain
- Consistent input format for augmentation

**Cons:**
- Augmented images may lose some of the preprocessing benefits
- Background replacement happens before augmentation

**Implementation:**
```python
def preprocess_single_image_with_augmentation(
    self, 
    image_path: str, 
    augmentation_preset: str = "noise_color_only",
    num_augmentations: int = 5,
    debug: bool = False, 
    output_dir: str = None
) -> List[np.ndarray]:
    # 1. Run existing preprocessing pipeline
    processed_image = self.preprocess_single_image(image_path, debug, output_dir)
    
    if processed_image is None:
        return []
    
    # 2. Apply augmentation to the processed image
    augmenter = PlainImageAugmenter(
        source_dir=None,  # We'll pass the image directly
        target_dir=None,  # We'll return the augmented images
        preset=augmentation_preset,
        config_path="config/augmentation_noise_color.yml"
    )
    
    # 3. Create augmented versions
    augmented_images = []
    for i in range(num_augmentations):
        augmented = augmenter.augment_single_image(processed_image, i)
        if augmented is not None:
            augmented_images.append(augmented)
    
    return [processed_image] + augmented_images
```

### Approach 2: Mid-Pipeline Augmentation
Insert augmentation between background replacement and resizing.

**Pros:**
- Augmentation works on higher resolution images
- More realistic noise/color variations
- Background replacement preserved

**Cons:**
- More complex pipeline
- Potential for inconsistent results
- Harder to debug

**Implementation:**
```python
# After Step 5 (Background Replacement), before Step 6 (Resize)
if cropped_mask is not None and background_replaced is not None:
    # Apply augmentation here
    augmented_versions = self.apply_augmentation(
        background_replaced, 
        augmentation_preset,
        num_augmentations
    )
    
    # Process each augmented version
    processed_images = []
    for aug_image in augmented_versions:
        # Continue with resize and normalization
        resized = ImageTransforms.resize_square_image(aug_image, self.target_size)
        normalized = resized / self.normalize_scale
        processed_images.append(normalized)
    
    return processed_images
```

### Approach 3: Pre-Augmentation Processing
Apply augmentation before the main preprocessing pipeline.

**Pros:**
- Augmentation works on original, high-quality images
- Most realistic variations

**Cons:**
- Augmentation parameters may not work well with unprocessed images
- Complex integration with existing pipeline
- Potential for pipeline failures on augmented images

## Recommended Implementation: Approach 1

### Why Post-Processing is Best:
1. **Clean Architecture**: Preprocessing and augmentation are separate concerns
2. **Consistent Input**: Augmentation works on standardized 224x224 images
3. **Easier Debugging**: Can isolate augmentation issues from preprocessing issues
4. **Flexible**: Can easily enable/disable augmentation without affecting core pipeline
5. **Performance**: No need to run preprocessing multiple times

### Implementation Steps:

#### Step 1: Modify the Main Class
```python
class SiamesePreprocessingPipeline:
    def __init__(self):
        # ... existing initialization ...
        
        # Add augmentation configuration
        self.augmentation_enabled = siamese_dp.get('augmentation', {}).get('enabled', False)
        self.augmentation_preset = siamese_dp.get('augmentation', {}).get('preset', 'noise_color_only')
        self.augmentation_count = siamese_dp.get('augmentation', {}).get('count', 5)
```

#### Step 2: Add Augmentation Method
```python
def apply_augmentation(self, image: np.ndarray, preset: str, count: int) -> List[np.ndarray]:
    """Apply augmentation to a single processed image."""
    try:
        # Create temporary augmenter instance
        augmenter = PlainImageAugmenter(
            source_dir=None,
            target_dir=None,
            preset=preset,
            config_path="config/augmentation_noise_color.yml"
        )
        
        # Apply augmentation
        augmented_images = []
        for i in range(count):
            augmented = augmenter.augment_single_image(image, i)
            if augmented is not None:
                augmented_images.append(augmented)
        
        return augmented_images
    except Exception as e:
        print(f"⚠️ Augmentation failed: {e}")
        return []
```

#### Step 3: Modify Main Processing Method
```python
def preprocess_single_image_with_augmentation(
    self, 
    image_path: str, 
    debug: bool = False, 
    output_dir: str = None
) -> List[np.ndarray]:
    """Process single image with optional augmentation."""
    
    # Run existing preprocessing pipeline
    processed_image = self.preprocess_single_image(image_path, debug, output_dir)
    
    if processed_image is None:
        return []
    
    result_images = [processed_image]  # Always include original processed image
    
    # Apply augmentation if enabled
    if self.augmentation_enabled:
        augmented_images = self.apply_augmentation(
            processed_image,
            self.augmentation_preset,
            self.augmentation_count
        )
        result_images.extend(augmented_images)
        
        if debug:
            print(f"🔍 Created {len(augmented_images)} augmented versions")
    
    return result_images
```

## Configuration Integration

### Update Siamese Network Config
```yaml
siamese_network:
  training:
    data_preprocessing:
      # ... existing settings ...
      
      # Augmentation settings
      augmentation:
        enabled: true
        preset: "noise_color_only"
        count: 5
        config_file: "config/augmentation_noise_color.yml"
```

### Augmentation Config File
```yaml
augmentation:
  presets:
    noise_color_only:
      default_steps_per_attribute: 2
      # Disable geometric transformations
      rotation: {'enabled': false}
      scale: {'enabled': false}
      perspective: {'enabled': false}
      
      # Enable only noise and color
      brightness: {'limit': 0.15, 'steps': 2}
      contrast: {'limit': 0.15, 'steps': 2}
      saturation: {'limit': 0.2, 'steps': 2}
      hue_shift: {'limit': 10, 'steps': 2}
      gaussian_noise: {'limit': 0.2, 'steps': 2}
      
      # Disable other effects
      blur: {'enabled': false}
      motion_blur: {'enabled': false}
```

## Usage Examples

### Basic Usage
```python
pipeline = SiamesePreprocessingPipeline()

# Process with augmentation
processed_images = pipeline.preprocess_single_image_with_augmentation(
    "input.jpg",
    debug=True,
    output_dir="debug_output"
)

print(f"Created {len(processed_images)} images (1 original + {len(processed_images)-1} augmented)")
```

### Batch Processing with Augmentation
```python
def process_batch_with_augmentation(self, input_dir: str, output_dir: str):
    """Process batch of images with augmentation."""
    image_files = [f for f in os.listdir(input_dir) if is_img_file(f)]
    
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        # Process with augmentation
        processed_images = self.preprocess_single_image_with_augmentation(
            input_path,
            debug=False
        )
        
        # Save all versions
        for i, img in enumerate(processed_images):
            if i == 0:
                output_name = f"{base_name}_processed.jpg"
            else:
                output_name = f"{base_name}_processed_aug{i:03d}.jpg"
            
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, (img * 255).astype(np.uint8))
```

## Benefits of This Integration

1. **Data Augmentation**: Creates multiple training samples from single images
2. **Configurable**: Easy to adjust augmentation parameters via YAML
3. **Non-Destructive**: Original preprocessing pipeline remains unchanged
4. **Flexible**: Can enable/disable augmentation per use case
5. **Consistent**: All images (original + augmented) go through same preprocessing
6. **Debuggable**: Clear separation makes troubleshooting easier

## Considerations

1. **Memory Usage**: Multiple augmented images increase memory requirements
2. **Processing Time**: Augmentation adds processing overhead
3. **Storage**: More output files require more storage space
4. **Quality Control**: Need to ensure augmented images maintain quality
5. **Configuration Management**: Augmentation configs need to be maintained

## Future Enhancements

1. **Adaptive Augmentation**: Adjust augmentation based on image quality
2. **Quality Metrics**: Validate augmented image quality
3. **Batch Augmentation**: Process multiple images simultaneously
4. **Augmentation History**: Track which augmentations were applied
5. **Custom Transforms**: Add project-specific augmentation types
