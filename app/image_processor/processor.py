"""
Enhanced image processor for plain image processing.

This module provides a comprehensive image processing pipeline that can handle:
- Plain image processing (no model required)
- Batch processing and complex pipelines
"""

import os
from typing import Union, Tuple, List, Callable

import cv2
import numpy as np

from utils.image_utils import strip_filename_from_path


class ImageProcessor:

    def __init__(self):
        pass
    
    def _load_image_array(self, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            return cv2.imread(image)
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise ValueError("Image must be a file path or numpy array")
    
    def process(
        self,
        image_path: str,
        output_path: str,
        processing_function: Callable[[np.ndarray], np.ndarray],
        suffix: str = "",
        **processing_kwargs,
    ) -> bool:
        try:
            image_array = self._load_image_array(image_path)
            processed_image = processing_function(image_array, **processing_kwargs)
            return self._save_processed_image(processed_image, image_path, output_path, suffix)
        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")
            return False
    
    def process_pipeline(
        self,
        image_path: str,
        output_path: str,
        processing_steps: List[Tuple[Callable, dict]],
        suffix: str = "",
    ) -> bool:
        try:
            image_array = self._load_image_array(image_path)
            
            for i, (processing_function, kwargs) in enumerate(processing_steps):
                try:
                    image_array = processing_function(image_array, **kwargs)
                    print(f"✅ Applied step {i+1}/{len(processing_steps)}")
                except Exception as e:
                    print(f"❌ Error in processing step {i+1}: {e}")
                    return False
            
            return self._save_processed_image(image_array, image_path, output_path, suffix)
        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")
            return False
    
    def process_batch(
        self,
        input_directory: str,
        output_directory: str,
        processing_function: Callable,
        file_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
        suffix: str = "",
        **processing_kwargs,
    ) -> dict:
        if not os.path.exists(input_directory):
            raise FileNotFoundError(f"Input directory not found: {input_directory}")
        
        os.makedirs(output_directory, exist_ok=True)
        
        image_files = []
        for ext in file_extensions:
            image_files.extend(
                [f for f in os.listdir(input_directory) if f.lower().endswith(ext)]
            )
        
        if not image_files:
            print(f"⚠️  No image files found in {input_directory}")
            return {"success": 0, "failed": 0, "total": 0}
        
        print(f"🎯 Processing {len(image_files)} images...")
        
        success_count = 0
        failed_count = 0
        
        for i, filename in enumerate(image_files, 1):
            input_path = os.path.join(input_directory, filename)
            output_filename = strip_filename_from_path(filename) + suffix + os.path.splitext(filename)[1]
            output_path = os.path.join(output_directory, output_filename)
            
            print(f"📸 Processing {i}/{len(image_files)}: {filename}")
            
            success = self.process(input_path, output_path, processing_function, suffix, **processing_kwargs)
            
            if success:
                success_count += 1
                print(f"✅ Successfully processed: {filename}")
            else:
                failed_count += 1
                print(f"❌ Failed to process: {filename}")
        
        print(f"\n📊 Batch processing complete!")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   📁 Total: {len(image_files)}")
        
        return {"success": success_count, "failed": failed_count, "total": len(image_files)}
    
    def _save_processed_image(
        self,
        processed_image: np.ndarray,
        original_path: str,
        output_path: str,
        suffix: str = "",
    ) -> bool:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            original_filename = os.path.basename(original_path)
            name, ext = os.path.splitext(original_filename)
            output_filename = name + suffix + ext
            full_output_path = os.path.join(output_path, output_filename)
            
            cv2.imwrite(full_output_path, processed_image)
            print(f"💾 Saved processed image: {full_output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving processed image: {e}")
            return False
    
    def process_image(self, *args, **kwargs):
        return self.process(*args, **kwargs)
