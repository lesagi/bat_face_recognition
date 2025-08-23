"""
Configuration management for the augmentation system.

Handles loading and managing augmentation presets and custom pipelines.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class AugmentationConfig:
    """Manages augmentation configuration and preset loading."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path:
            self.load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'augmentation': {
                'output_naming': '_a-{index:03d}',
                'presets': {
                    'light': {
                        'default_steps_per_attribute': 3,
                        'rotation': {'limit': 15},
                        'brightness': {'limit': 0.2},
                        'contrast': {'limit': 0.2},
                        'noise': {'limit': 0.1}
                    },
                    'medium': {
                        'default_steps_per_attribute': 5,
                        'rotation': {'limit': 25, 'steps': 6},
                        'brightness': {'limit': 0.3},
                        'contrast': {'limit': 0.3},
                        'scale': {'range': [0.8, 1.2]},
                        'noise': {'limit': 0.2}
                    },
                    'heavy': {
                        'default_steps_per_attribute': 8,
                        'rotation': {'limit': 30, 'steps': 10},
                        'scale': {'range': [0.7, 1.3]},
                        'brightness': {'limit': 0.4, 'steps': 5},
                        'contrast': {'limit': 0.4},
                        'noise': {'limit': 0.3},
                        'blur': {'limit': 0.2}
                    }
                },
                'custom_pipelines': {},
                'yolo_detection': {
                    'bbox_min_visibility': 0.3,
                    'bbox_format': 'yolo'
                },
                'yolo_segmentation': {
                    'mask_interpolation': 'nearest',
                    'segmentation_simplification': 0.005
                }
            }
        }
    
    def load_config(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self._merge_config(user_config)
        except Exception as e:
            print(f"⚠️ Could not load config from {config_path}: {e}")
            print("Using default configuration")
    
    def _merge_config(self, user_config: Dict[str, Any]):
        """Merge user configuration with defaults."""
        if 'augmentation' in user_config:
            # Merge presets
            if 'presets' in user_config['augmentation']:
                self.config['augmentation']['presets'].update(
                    user_config['augmentation']['presets']
                )
            
            # Merge custom pipelines
            if 'custom_pipelines' in user_config['augmentation']:
                self.config['augmentation']['custom_pipelines'].update(
                    user_config['augmentation']['custom_pipelines']
                )
            
            # Merge other settings
            for key, value in user_config['augmentation'].items():
                if key not in ['presets', 'custom_pipelines']:
                    self.config['augmentation'][key] = value
    
    def get_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific preset configuration."""
        presets = self.config['augmentation']['presets']
        return presets.get(preset_name)
    
    def get_custom_pipeline(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """Get a custom pipeline configuration."""
        pipelines = self.config['augmentation']['custom_pipelines']
        return pipelines.get(pipeline_name)
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        return self.config['augmentation'].get(key, default)
    
    def list_presets(self) -> list:
        """List available preset names."""
        return list(self.config['augmentation']['presets'].keys())
    
    def list_custom_pipelines(self) -> list:
        """List available custom pipeline names."""
        return list(self.config['augmentation']['custom_pipelines'].keys())
