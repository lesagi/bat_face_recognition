"""
Base augmenter class providing common functionality for all augmenter types.

This abstract base class handles common operations like image I/O, configuration,
and basic augmentation execution.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .config import AugmentationConfig
from .pipeline_factory import AugmentationPipelineFactory


class BaseAugmenter(ABC):
    """Abstract base class for all augmenter types."""
    
    def __init__(self, source_dir: str, target_dir: str, preset: str = 'medium', 
                 config_path: Optional[str] = None, debug: bool = False):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.preset = preset
        self.debug = debug
        
        # Initialize configuration and pipeline factory
        self.config = AugmentationConfig(config_path)
        self.pipeline_factory = AugmentationPipelineFactory(self.config)
        
        # Track current preset for pipeline factory
        self.config.current_preset = preset
        
        # Create target directory structure
        self.setup_directories()
        
        # Get pipeline and augmentation count
        self.transform, self.total_augmentations = self.pipeline_factory.create_pipeline_from_preset(preset)
        
        if self.debug:
            self._print_pipeline_info()
    
    def _print_pipeline_info(self):
        """Print information about the current pipeline."""
        info = self.pipeline_factory.get_pipeline_info(self.preset)
        print(f"🔧 Pipeline: {info['preset_name']}")
        print(f"📊 Total augmentations: {info['total_augmentations']}")
        print(f"⚙️ Parameters:")
        for param_name, param_info in info['parameters'].items():
            steps = param_info['steps']
            custom = "custom" if param_info['custom'] else "default"
            print(f"   {param_name}: {steps} steps ({custom})")
        print()
    
    def setup_directories(self):
        """Create the target directory structure. Override in subclasses."""
        self.target_dir.mkdir(parents=True, exist_ok=True)
        if self.debug:
            print(f"✅ Created target directory: {self.target_dir}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported image file extensions."""
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load image from path and convert to RGB."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None
    
    def save_image(self, image: np.ndarray, output_path: Path):
        """Save image to path, converting RGB to BGR for OpenCV."""
        try:
            # Convert RGB to BGR for saving
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), image_bgr)
            
            if self.debug:
                print(f"💾 Saved: {output_path.name}")
        except Exception as e:
            print(f"❌ Error saving image {output_path}: {e}")
    
    def get_output_filename(self, base_name: str, index: int, extension: str) -> str:
        """Generate output filename using configured naming pattern."""
        naming_pattern = self.config.get_setting('output_naming', '_a-{index:03d}')
        return f"{base_name}{naming_pattern.format(index=index)}{extension}"
    
    def get_image_files(self, directory: Path) -> List[Path]:
        """Get list of image files from directory."""
        image_files = []
        for ext in self.get_supported_extensions():
            image_files.extend(list(directory.glob(f"*{ext}")))
            image_files.extend(list(directory.glob(f"*{ext.upper()}")))
        return sorted(image_files)
    
    def validate_input(self) -> bool:
        """Validate input directory structure. Override in subclasses."""
        if not self.source_dir.exists():
            print(f"❌ Source directory not found: {self.source_dir}")
            return False
        return True
    
    @abstractmethod
    def augment_single_item(self, item_path: Path, index: int) -> bool:
        """Augment a single item (image, dataset, etc.). Override in subclasses."""
        pass
    
    def augment_dataset(self) -> Dict[str, int]:
        """Main method to augment the entire dataset."""
        print(f"🔄 Starting augmentation with preset: {self.preset}")
        print(f"📊 Target: {self.total_augmentations} augmentations per item")
        print()
        
        if not self.validate_input():
            return {}
        
        # Get items to process (implemented by subclasses)
        items = self._get_items_to_process()
        if not items:
            print("❌ No items found to process")
            return {}
        
        print(f"📁 Found {len(items)} items to process")
        print()
        
        successful_items = 0
        total_attempted = len(items) * self.total_augmentations
        
        for item_path in items:
            try:
                if self.augment_single_item(item_path, successful_items):
                    successful_items += 1
            except Exception as e:
                print(f"❌ Error processing {item_path.name}: {e}")
                continue
        
        print(f"🎯 Summary:")
        print(f"   Items processed: {successful_items}/{len(items)}")
        print(f"   Total augmentations: {successful_items * self.total_augmentations}")
        print(f"   Success rate: {successful_items/len(items)*100:.1f}%")
        
        return {'processed': successful_items, 'total': len(items)}
    
    @abstractmethod
    def _get_items_to_process(self) -> List[Path]:
        """Get list of items to process. Override in subclasses."""
        pass
    
    def list_available_presets(self):
        """List available preset configurations."""
        presets = self.config.list_presets()
        print("📋 Available presets:")
        for preset in presets:
            info = self.pipeline_factory.get_pipeline_info(preset)
            print(f"   {preset}: {info['total_augmentations']} augmentations")
    
    def get_preset_info(self, preset_name: str):
        """Get detailed information about a specific preset."""
        info = self.pipeline_factory.get_pipeline_info(preset_name)
        if not info:
            print(f"❌ Preset '{preset_name}' not found")
            return
        
        print(f"🔧 Preset: {info['preset_name']}")
        print(f"📊 Total augmentations: {info['total_augmentations']}")
        print(f"⚙️ Parameters:")
        for param_name, param_info in info['parameters'].items():
            steps = param_info['steps']
            custom = "custom" if param_info['custom'] else "default"
            print(f"   {param_name}: {steps} steps ({custom})")
        
        if info['combinations']:
            print(f"\n📝 Sample combinations (first 5):")
            for i, combo in enumerate(info['combinations']):
                print(f"   {i}: {combo}")
