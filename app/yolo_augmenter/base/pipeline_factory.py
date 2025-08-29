"""
Augmentation pipeline factory for creating systematic transformation pipelines.

Generates all possible parameter combinations and creates corresponding albumentations transforms.
"""

import albumentations as A
import numpy as np
from typing import Dict, Any, List, Tuple
from itertools import product


class AugmentationPipelineFactory:
    """Factory for creating systematic augmentation pipelines."""
    
    def __init__(self, config):
        self.config = config
    
    def create_pipeline_from_preset(self, preset_name: str) -> Tuple[A.Compose, int]:
        """Create augmentation pipeline from preset and return total augmentations count."""
        preset_config = self.config.get_preset(preset_name)
        if not preset_config:
            raise ValueError(f"Preset '{preset_name}' not found")
        
        # Store the current preset name for use in transform creation
        self.current_preset_name = preset_name
        
        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations(preset_config)
        
        # Create transforms for each combination
        transforms = self._create_transforms_from_combinations(param_combinations)
        
        return transforms, len(param_combinations)
    
    def create_pipeline_from_custom(self, pipeline_name: str) -> Tuple[A.Compose, int]:
        """Create augmentation pipeline from custom pipeline and return total augmentations count."""
        pipeline_config = self.config.get_custom_pipeline(pipeline_name)
        if not pipeline_config:
            raise ValueError(f"Custom pipeline '{pipeline_name}' not found")
        
        # For custom pipelines, we might have a different structure
        # This is a placeholder for future implementation
        raise NotImplementedError("Custom pipeline support not yet implemented")
    
    def _generate_parameter_combinations(self, preset_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations based on preset configuration."""
        default_steps = preset_config.get('default_steps_per_attribute', 4)
        combinations = []
        
        for param_name, param_config in preset_config.items():
            if param_name == 'default_steps_per_attribute':
                continue
            
            # Check if this transform is disabled
            if isinstance(param_config, dict) and param_config.get('enabled') == False:
                continue  # Skip disabled transforms entirely
            
            # Extract parameter value and steps
            if isinstance(param_config, dict):
                # Dictionary parameter - extract value and check for custom steps
                if 'limit' in param_config:
                    param_value = param_config['limit']
                elif 'range' in param_config:
                    param_value = param_config['range']
                else:
                    continue
                
                # Check if custom steps specified, otherwise use default
                if 'steps' in param_config:
                    steps = param_config['steps']
                else:
                    steps = default_steps
            else:
                # Simple parameter - use as-is with default steps
                param_value = param_config
                steps = default_steps
            
            # Generate steps from 0 to param_value
            if isinstance(param_value, list):  # Range parameter
                param_steps = np.linspace(param_value[0], param_value[1], steps)
            else:  # Single value parameter
                param_steps = np.linspace(0, param_value, steps)
            
            combinations.append((param_name, param_steps))
        
        # Generate all permutations
        param_names = [name for name, _ in combinations]
        param_values = [values for _, values in combinations]
        
        all_combinations = []
        for combo in product(*param_values):
            all_combinations.append(dict(zip(param_names, combo)))
        
        return all_combinations
    
    def _create_transforms_from_combinations(self, param_combinations: List[Dict[str, Any]]) -> A.Compose:
        """Create albumentations transforms from parameter combinations."""
        # Get the preset configuration to determine which transforms to enable
        # Get the preset configuration using the stored preset name
        if hasattr(self, 'current_preset_name'):
            preset_config = self.config.get_preset(self.current_preset_name)
        else:
            preset_config = None
        
        transforms = []
        
        # Create transforms based on actual configuration values
        if preset_config:
            # Brightness and Contrast
            if self._is_transform_enabled(preset_config, 'brightness') or self._is_transform_enabled(preset_config, 'contrast'):
                brightness_limit = preset_config.get('brightness', {}).get('limit', 0.2)
                contrast_limit = preset_config.get('contrast', {}).get('limit', 0.2)
                transforms.append(A.RandomBrightnessContrast(
                    brightness_limit=(-brightness_limit, brightness_limit),
                    contrast_limit=(-contrast_limit, contrast_limit),
                    p=1.0
                ))
            
            # Saturation and Hue
            if self._is_transform_enabled(preset_config, 'saturation') or self._is_transform_enabled(preset_config, 'hue_shift'):
                saturation_limit = preset_config.get('saturation', {}).get('limit', 0.2)
                hue_limit = preset_config.get('hue_shift', {}).get('limit', 20)
                transforms.append(A.HueSaturationValue(
                    hue_shift_limit=(-hue_limit, hue_limit),
                    sat_shift_limit=(-saturation_limit, saturation_limit),
                    val_shift_limit=(-saturation_limit, saturation_limit),
                    p=1.0
                ))
            
            # Multiplicative Noise
            if self._is_transform_enabled(preset_config, 'noise'):
                noise_limit = preset_config.get('noise', {}).get('limit', 0.2)
                transforms.append(A.MultiplicativeNoise(
                    multiplier=(1.0 - noise_limit, 1.0 + noise_limit),
                    p=1.0
                ))
            
            # Gaussian Noise
            if self._is_transform_enabled(preset_config, 'gaussian_noise'):
                gauss_limit = preset_config.get('gaussian_noise', {}).get('limit', 0.2)
                # For gaussian noise, use var_limit to control noise intensity
                # Higher values = more noise
                transforms.append(A.GaussNoise(
                    var_limit=(0, gauss_limit),
                    p=1.0
                ))
            
            # ISO Noise
            if self._is_transform_enabled(preset_config, 'iso_noise'):
                iso_config = preset_config.get('iso_noise', {})
                intensity = iso_config.get('intensity', [0.1, 0.5])
                color_shift = iso_config.get('color_shift', [0.01, 0.05])
                transforms.append(A.ISONoise(
                    color_shift=color_shift,
                    intensity=intensity,
                    p=1.0
                ))
            
            # RGB Shift
            if self._is_transform_enabled(preset_config, 'rgb_shift'):
                rgb_limit = preset_config.get('rgb_shift', {}).get('limit', 15)
                transforms.append(A.RGBShift(
                    r_shift_limit=(-rgb_limit, rgb_limit),
                    g_shift_limit=(-rgb_limit, rgb_limit),
                    b_shift_limit=(-rgb_limit, rgb_limit),
                    p=1.0
                ))
            
            # Geometric transformations (only if enabled)
            if self._is_transform_enabled(preset_config, 'rotation'):
                rotation_limit = preset_config.get('rotation', {}).get('limit', 30)
                transforms.append(A.Rotate(limit=(-rotation_limit, rotation_limit), p=1.0))
            
            if self._is_transform_enabled(preset_config, 'scale'):
                scale_config = preset_config.get('scale', {})
                if 'range' in scale_config:
                    scale_range = scale_config['range']
                else:
                    scale_limit = scale_config.get('limit', 0.2)
                    scale_range = (1.0 - scale_limit, 1.0 + scale_limit)
                transforms.append(A.Affine(scale=scale_range, p=1.0))
            
            if self._is_transform_enabled(preset_config, 'perspective'):
                perspective_scale = preset_config.get('perspective', {}).get('scale', [0.02, 0.08])
                transforms.append(A.Perspective(scale=perspective_scale, p=1.0))
            
            # Blur transformations
            if self._is_transform_enabled(preset_config, 'blur'):
                blur_limit = preset_config.get('blur', {}).get('limit', 5)
                transforms.append(A.Blur(blur_limit=blur_limit, p=1.0))
            
            if self._is_transform_enabled(preset_config, 'motion_blur'):
                motion_limit = preset_config.get('motion_blur', {}).get('limit', 5)
                transforms.append(A.MotionBlur(blur_limit=motion_limit, p=1.0))
        
        # If no transforms are enabled, return identity transform
        if not transforms:
            return A.Compose([A.NoOp()])
        
        return A.Compose(transforms)
    
    def _is_transform_enabled(self, preset_config: Dict[str, Any], transform_name: str) -> bool:
        """Check if a specific transform is enabled in the preset configuration."""
        if transform_name not in preset_config:
            return False
        
        transform_config = preset_config[transform_name]
        
        # If it's a dictionary with 'enabled' field, check that
        if isinstance(transform_config, dict):
            return transform_config.get('enabled', True)
        
        # If it's a simple value, assume it's enabled
        return True
    
    def get_pipeline_info(self, preset_name: str) -> Dict[str, Any]:
        """Get information about a pipeline including parameter combinations."""
        preset_config = self.config.get_preset(preset_name)
        if not preset_config:
            return {}
        
        param_combinations = self._generate_parameter_combinations(preset_config)
        
        # Analyze the pipeline
        info = {
            'preset_name': preset_name,
            'total_augmentations': len(param_combinations),
            'parameters': {},
            'combinations': param_combinations[:5]  # Show first 5 combinations
        }
        
        # Extract parameter information - only for enabled transforms
        for param_name, param_config in preset_config.items():
            if param_name == 'default_steps_per_attribute':
                continue
                
            # Check if this transform is disabled
            if isinstance(param_config, dict) and param_config.get('enabled') == False:
                continue  # Skip disabled transforms
            
            if isinstance(param_config, dict):
                if 'steps' in param_config:
                    info['parameters'][param_name] = {
                        'steps': param_config['steps'],
                        'custom': True
                    }
                else:
                    info['parameters'][param_name] = {
                        'steps': preset_config.get('default_steps_per_attribute', 4),
                        'custom': False
                    }
            else:
                info['parameters'][param_name] = {
                    'steps': preset_config.get('default_steps_per_attribute', 4),
                    'custom': False
                }
        
        return info
