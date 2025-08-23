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
        # For now, we'll create a single transform that can handle all combinations
        # In the future, we might want to create separate transforms for each combination
        
        # Create a flexible transform that can be parameterized
        transform = A.Compose([
            A.OneOf([
                A.Rotate(limit=(-30, 30), p=0.7),
                A.Affine(
                    scale=(0.7, 1.3),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-30, 30),
                    shear=(-8, 8),
                    p=0.5,
                ),
                A.Perspective(scale=(0.02, 0.08), p=0.3),
            ], p=0.8),
            
            A.OneOf([
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.8
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.4, contrast_limit=0.4, p=0.7
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.6,
                ),
                A.RGBShift(
                    r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5
                ),
            ], p=0.9),
            
            A.OneOf([
                A.GaussNoise(var_limit=(5, 50), p=0.6),
                A.ISONoise(
                    color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4
                ),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.4),
            ], p=0.4),
            
            A.OneOf([
                A.Blur(blur_limit=5, p=0.3),
                A.MotionBlur(blur_limit=5, p=0.3),
                A.Sharpen(alpha=(0.1, 0.5), lightness=(0.7, 1.0), p=0.2),
            ], p=0.2),
        ])
        
        return transform
    
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
        
        # Extract parameter information
        for param_name, param_config in preset_config.items():
            if param_name != 'default_steps_per_attribute':
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
