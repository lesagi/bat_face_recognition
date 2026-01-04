#!/usr/bin/env python3
"""
Siamese Network Advanced Image Processor

This module implements the complete advanced image processing pipeline for preparing images
for the siamese network model. This processor includes segmentation-based background 
replacement and cropping for optimal siamese network training.

Pipeline steps:
1. Face Alignment → 2. YOLO Segmentation → 3. Square Cropping → 4. Background Replacement → 5. Resize to Target → 6. Normalize to [0,1]

Features:
- YOLO-based bat face segmentation
- Configurable background replacement using BackgroundGenerator
- Smart square cropping around detected bat faces with buffer
- Multiple face alignment options: OpenCV, MediaPipe, or YOLO Pose
- YOLO Pose integration for accurate bat face landmark detection
- Standardized 224x224 output for siamese network
- Batch and single image processing support

Note: Both YOLO segmentation and YOLO pose models are REQUIRED for this pipeline.
The segmentation model is used for bat face detection and background replacement.
The pose model is used for accurate face alignment and landmark detection.

Usage:
    # Process single image
    python input_processor.py process-single --input image.jpg --output processed.jpg --segmentation-model face_segmentation.pt --pose-model face_pose.pt
    
    # Process batch of images
    python input_processor.py process-batch --input-dir /path/to/images/ --output-dir /path/to/processed/ --segmentation-model face_segmentation.pt --pose-model face_pose.pt
    
    # Get help
    python input_processor.py --help
    python input_processor.py process-single --help
    python input_processor.py process-batch --help
"""

import os
import sys

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import cv2
import click

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try absolute imports first (when run as module), fall back to relative imports (when run directly)
try:
    from app.image_processor import ImageTransforms
    from app.background_generation.background_generator import BackgroundGenerator
    from app.utils.image_utils import is_img_file
    from app.config.loader import load_config
except ImportError:
    # Fallback to relative imports when running script directly
    # Make sure the parent directory is in the path for relative imports
    sys.path.insert(0, parent_dir)
    from image_processor import ImageTransforms
    from background_generation.background_generator import BackgroundGenerator
    from app.utils.image_utils import is_img_file
    from app.config.loader import load_config

# Import prediction structures and caching
try:
    print("🔍 Attempting to import prediction modules...")
    from app.siamese_preprocessing.prediction_structures import (
        SegmentationPrediction, 
        PosePrediction, 
        PredictionBundle
    )
    print("✅ Successfully imported prediction_structures")
    from app.siamese_preprocessing.prediction_cache import CacheManager, CacheConfig
    print("✅ Successfully imported prediction_cache")
    from app.siamese_preprocessing.prediction_transforms import PredictionTransformer, CoordinateMapper
    print("✅ Successfully imported prediction_transforms")
    PREDICTION_IMPORTS_AVAILABLE = True
    print("✅ All prediction imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Fallback for when prediction modules are not available
    SegmentationPrediction = None
    PosePrediction = None
    PredictionBundle = None
    CacheManager = None
    CacheConfig = None
    PredictionTransformer = None
    CoordinateMapper = None
    PREDICTION_IMPORTS_AVAILABLE = False
    print("⚠️ Prediction imports not available, skipping cache")


class SiamesePreprocessingPipeline:
    """Advanced preprocessing pipeline for siamese network with segmentation and pose-based processing.

    This pipeline uses YOLO segmentation to detect bat faces and YOLO pose for face alignment.
    It replaces backgrounds with generated content and performs smart cropping for optimal 
    siamese network performance.
    """

    def __init__(self):
        # Load configuration
        config = load_config()
        siamese_dp = config.siamese_network.training.get('data_preprocessing', {})
        
        # Margin ratio for cropping
        face_outer_margin_ratio = siamese_dp.get('face_outer_margin_ratio', 0.5)

        # Background generator type from config
        background_generators = {
            "blur": BackgroundGenerator.blur,
            "noise": BackgroundGenerator.noise,
            "gradient": BackgroundGenerator.gradient,
            "picsum": BackgroundGenerator.picsum,
            "solid_color": BackgroundGenerator.solid_color,
        }
        bg_config = siamese_dp.get('background_replacement', {})
        
        self.background_enabled = bg_config.get('enabled', False)  # Default to False
        self.background_generator = None
        if self.background_enabled:
            self.background_generator = background_generators.get(bg_config.get('type', 'blur'), BackgroundGenerator.blur)

        self.face_outer_margin_ratio = face_outer_margin_ratio
        self.target_size = siamese_dp.get('target_size', 224)
        self.normalize_scale = siamese_dp.get('scale_factor', 255.0)
        
        # Prediction caching configuration
        if PREDICTION_IMPORTS_AVAILABLE:
            cache_config = siamese_dp.get('prediction_caching', {})
            self.cache_manager = CacheManager(CacheConfig(
                max_cache_size=cache_config.get('cache_size', 1000),
                max_memory_mb=cache_config.get('max_memory_mb', 512),
                cache_ttl_hours=cache_config.get('cache_ttl_hours', 24),
                enable_persistence=cache_config.get('persistence', 'memory') == 'disk',
                cache_dir=cache_config.get('cache_dir', 'cache/predictions')
            ))
        else:
            self.cache_manager = None
        
        # Augmentation configuration
        augmentation_config = siamese_dp.get('augmentation', {})
        self.augmentation_enabled = augmentation_config.get('enabled', False)
        self.augmentation_preset = augmentation_config.get('preset', 'noise_color_only')
        self.augmentation_count = augmentation_config.get('count', 5)
        self.augmentation_config_file = augmentation_config.get('config_file', os.path.join(parent_dir, 'config', 'augmentation_noise_color.yml'))
        
        # Prediction transformation settings
        self.transform_predictions = augmentation_config.get('transform_predictions', True)
        self.validate_transformations = augmentation_config.get('validate_transformations', True)
    
    def _transform_segmentation_for_augmentation(self, segmentation_prediction: SegmentationPrediction, 
                                                transformation_matrix: Any) -> SegmentationPrediction:
        """Transform segmentation prediction to match augmented image dimensions."""
        try:
            if not PREDICTION_IMPORTS_AVAILABLE:
                return segmentation_prediction
            
            # Extract the mask from the prediction
            original_mask = segmentation_prediction.mask
            
            # For now, we'll handle basic transformations
            # In a full implementation, this would parse the transformation matrix
            # and apply the appropriate transformations
            
            # Create a new segmentation prediction with transformed mask
            transformed_prediction = SegmentationPrediction(
                mask=original_mask.copy(),  # For now, just copy the mask
                confidence=segmentation_prediction.confidence,
                bounding_box=segmentation_prediction.bounding_box,
                class_id=segmentation_prediction.class_id,
                class_name=segmentation_prediction.class_name,
                original_image_shape=segmentation_prediction.original_image_shape,
                model_resolution=segmentation_prediction.model_resolution,
                timestamp=segmentation_prediction.timestamp
            )
            
            # TODO: Apply actual transformation matrix to mask and bounding box
            # This would involve:
            # 1. Parsing the transformation matrix from the augmentation system
            # 2. Using MaskTransformer to transform the mask
            # 3. Using BoundingBoxTransformer to transform the bounding box
            
            return transformed_prediction
            
        except Exception as e:
            if hasattr(self, 'debug') and self.debug:
                print(f"⚠️ Failed to transform segmentation prediction: {e}")
            return segmentation_prediction
    
    def _transform_pose_for_augmentation(self, pose_prediction: PosePrediction, 
                                        transformation_matrix: Any) -> PosePrediction:
        """Transform pose prediction to match augmented image dimensions."""
        try:
            if not PREDICTION_IMPORTS_AVAILABLE:
                return pose_prediction
            
            # Extract the keypoints from the prediction
            original_keypoints = pose_prediction.keypoints
            
            # Try to parse and apply the transformation matrix
            if isinstance(transformation_matrix, str) and transformation_matrix.startswith("augmentation_"):
                # This is a placeholder - in real implementation, we'd have actual transformation data
                # For now, we'll create a basic transformation based on the augmentation type
                
                # Create a new pose prediction with transformed keypoints
                transformed_prediction = PosePrediction(
                    keypoints=original_keypoints.copy(),
                    bounding_box=pose_prediction.bounding_box,
                    confidence=pose_prediction.confidence,
                    class_id=pose_prediction.class_id,
                    class_name=pose_prediction.class_name,
                    original_image_shape=pose_prediction.original_image_shape,
                    model_resolution=pose_prediction.model_resolution,
                    timestamp=pose_prediction.timestamp
                )
                
                # TODO: Apply actual transformation matrix to keypoints and bounding box
                # This would involve:
                # 1. Parsing the transformation matrix from the augmentation system
                # 2. Using KeypointTransformer to transform the keypoints
                # 3. Using BoundingBoxTransformer to transform the bounding box
                
                return transformed_prediction
            else:
                # If we have a real transformation matrix, apply it
                # This would use our PredictionTransformer utilities
                return pose_prediction
                
        except Exception as e:
            if hasattr(self, 'debug') and self.debug:
                print(f"⚠️ Failed to transform pose prediction: {e}")
            return pose_prediction
    
    def _create_transformation_matrix_from_augmentation(self, augmentation_type: str, 
                                                      augmentation_params: dict) -> Any:
        """Create a transformation matrix from augmentation parameters.
        
        This method creates a transformation matrix that can be used to transform
        predictions to match augmented image dimensions and transformations.
        
        Args:
            augmentation_type: Type of augmentation (e.g., 'rotation', 'scale', 'translation')
            augmentation_params: Parameters for the augmentation
            
        Returns:
            Transformation matrix or transformation description
        """
        try:
            if not PREDICTION_IMPORTS_AVAILABLE:
                return None
            
            # This is a placeholder implementation
            # In a full implementation, this would:
            # 1. Parse augmentation parameters
            # 2. Create appropriate transformation matrices
            # 3. Return matrices that can be used by our transformation utilities
            
            if augmentation_type == "rotation":
                # Create rotation transformation matrix
                angle = augmentation_params.get('angle', 0)
                # TODO: Use CoordinateMapper.create_rotation_mapping()
                return f"rotation_{angle}"
                
            elif augmentation_type == "scale":
                # Create scale transformation matrix
                scale_x = augmentation_params.get('scale_x', 1.0)
                scale_y = augmentation_params.get('scale_y', 1.0)
                # TODO: Use CoordinateMapper.create_resize_mapping()
                return f"scale_{scale_x}_{scale_y}"
                
            elif augmentation_type == "translation":
                # Create translation transformation matrix
                dx = augmentation_params.get('dx', 0)
                dy = augmentation_params.get('dy', 0)
                # TODO: Use CoordinateMapper.create_crop_mapping() or custom translation
                return f"translation_{dx}_{dy}"
                
            else:
                # Unknown augmentation type
                return f"unknown_{augmentation_type}"
                
        except Exception as e:
            if hasattr(self, 'debug') and self.debug:
                print(f"⚠️ Failed to create transformation matrix: {e}")
            return None

    def _get_cached_predictions(self, image_path: str) -> Optional[PredictionBundle]:
        """Get cached predictions for an image if available."""
        if not PREDICTION_IMPORTS_AVAILABLE or self.cache_manager is None:
            return None
        
        try:
            return self.cache_manager.get_predictions(image_path, "both")
        except Exception as e:
            if hasattr(self, 'debug') and self.debug:
                print(f"⚠️ Failed to get cached predictions: {e}")
            return None
    
    def _cache_predictions(self, image_path: str, predictions: PredictionBundle, force: bool = False) -> bool:
        """Cache predictions for an image."""
        if not PREDICTION_IMPORTS_AVAILABLE or self.cache_manager is None:
            return False
        
        try:
            if force and not self.cache_manager.is_cached(image_path, "both"):
                return False
            
            return self.cache_manager.cache_predictions(image_path, "both", predictions)
        except Exception as e:
            if hasattr(self, 'debug') and self.debug:
                print(f"⚠️ Failed to cache predictions: {e}")
            return False
    
    def _create_prediction_bundle(self, image_path: str, segmentation_prediction: Optional[SegmentationPrediction] = None, 
                                 pose_prediction: Optional[PosePrediction] = None) -> Optional[PredictionBundle]:
        """Create a prediction bundle from individual predictions."""
        if not PREDICTION_IMPORTS_AVAILABLE:
            print(f"⚠️ Prediction imports not available, skipping cache")
            return None
        
        try:
            print(f"🔍 Creating prediction bundle: {image_path}")
            print(f"🔍 Segmentation prediction: {segmentation_prediction}")
            print(f"🔍 Pose prediction: {pose_prediction}")
            bundle = PredictionBundle(
                segmentation=segmentation_prediction,
                pose=pose_prediction,
                image_path=image_path
            )
            return bundle
        except Exception as e:
            if hasattr(self, 'debug') and self.debug:
                print(f"⚠️ Failed to create prediction bundle: {e}")
            return None
    
    def _transform_predictions_for_augmentation(self, predictions: PredictionBundle, 
                                              transformation_matrix: Any) -> Optional[PredictionBundle]:
        """Transform predictions to match augmented image dimensions.
        
        This method applies the transformation matrix from the augmentation system
        to transform cached predictions (masks, keypoints, bounding boxes) so they
        match the augmented image dimensions and transformations.
        """
        if not PREDICTION_IMPORTS_AVAILABLE or PredictionTransformer is None:
            return predictions  # Return original if transformation not available
        
        if predictions is None:
            return None
        
        try:
            # Create a copy of the predictions to avoid modifying the original
            transformed_bundle = PredictionBundle(
                segmentation=predictions.segmentation,
                pose=predictions.pose,
                image_path=predictions.image_path,
                image_hash=predictions.image_hash,
                timestamp=predictions.timestamp,
                processing_steps=predictions.processing_steps.copy(),
                model_versions=predictions.model_versions.copy()
            )
            
            # Transform segmentation predictions if available
            if predictions.segmentation:
                transformed_bundle.segmentation = self._transform_segmentation_for_augmentation(
                    predictions.segmentation, transformation_matrix
                )
            
            # Transform pose predictions if available
            if predictions.pose:
                transformed_bundle.pose = self._transform_pose_for_augmentation(
                    predictions.pose, transformation_matrix
                )
            
            # Add transformation step to processing history
            transformed_bundle.add_processing_step("Augmentation prediction transformation")
            
            if hasattr(self, 'debug') and self.debug:
                print(f"🔍 Transformed predictions for augmentation")
            
            return transformed_bundle
            
        except Exception as e:
            if hasattr(self, 'debug') and self.debug:
                print(f"⚠️ Failed to transform predictions: {e}")
            return predictions  # Return original on error
    
    def apply_augmentation(self, image: np.ndarray, preset: str, count: int) -> List[np.ndarray]:
        """Apply augmentation to a single processed image."""
        try:
            # Import augmentation system
            try:
                from yolo_augmenter.base.config import AugmentationConfig
                from yolo_augmenter.base.pipeline_factory import AugmentationPipelineFactory
                AUGMENTATION_AVAILABLE = True
            except ImportError:
                AUGMENTATION_AVAILABLE = False
                print("⚠️ YOLO Augmenter not available, skipping augmentation")
                return []
            
            if not AUGMENTATION_AVAILABLE:
                return []
            
            # Create configuration and pipeline factory
            config = AugmentationConfig(self.augmentation_config_file)
            pipeline_factory = AugmentationPipelineFactory(config)
            
            # Create augmentation pipeline for the specified preset
            transform, total_augmentations = pipeline_factory.create_pipeline_from_preset(preset)
            
            if transform is None:
                print(f"⚠️ Failed to create augmentation pipeline for preset: {preset}")
                return []
            
            # Apply augmentation
            augmented_images = []
            print(f"🔍 Applying augmentation preset: {preset}")
            print(f"🔍 Transform object: {transform}")
            print(f"🔍 Image shape: {image.shape}")
            
            for i in range(count):
                try:
                    # Apply the transform to the image
                    print(f"🔍 Applying transform {i+1}/{count}...")
                    augmented = transform(image=image)['image']
                    if augmented is not None:
                        augmented_images.append(augmented)
                        print(f"✅ Created augmented image {i+1}/{count} with shape: {augmented.shape}")
                        
                        if hasattr(self, 'debug') and self.debug:
                            print(f"🔍 Created augmented image {i+1}/{count}")
                            
                except Exception as e:
                    print(f"⚠️ Augmentation {i+1} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            return augmented_images
            
        except Exception as e:
            print(f"⚠️ Augmentation failed: {e}")
            return []

    def preprocess_single_image_from_path(self, image_path: str, debug: bool = False, output_dir: str = None) -> Optional[np.ndarray]:
        image_array = cv2.imread(image_path)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        if image_array is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        
        return self.preprocess_single_image(image_array, base_filename, image_path, debug, output_dir)
            

    def preprocess_single_image(self, img: np.ndarray, base_filename: str, cached_predictions_key: str = None, debug: bool = False, output_dir: str = None) -> Optional[np.ndarray]:
        try:
            current_image = img
            
            # Setup debug output and base filename
            if debug:
                if not output_dir:
                    raise Exception("Debug flag is True but no debug directory provided. Please specify --debug-dir.")
                
                os.makedirs(output_dir, exist_ok=True)
            
            # Load cached predictions if available
            cached_predictions = self._get_cached_predictions(cached_predictions_key)
            
            # Step 1: Face alignment using YOLO pose landmarks
            # Use cached pose prediction if available, otherwise run model inference
            pose_prediction = cached_predictions.pose if cached_predictions else None
            alignment_result = ImageTransforms.align_face_landmarks(
                current_image,
                pose_prediction=pose_prediction,
                face_detector_type="yolo_pose",
                debug=debug,
                debug_dir=output_dir,
                base_filename=base_filename
            )
            
            if alignment_result is not None and alignment_result[0] is not None:
                current_image = alignment_result[0]  # Extract the aligned image from the tuple
                # Update pose prediction if we got a new one
                if alignment_result[1] is not None:
                    pose_prediction = alignment_result[1]
            
            # Step 2: Get segmentation mask from aligned/original image
            # Use cached segmentation prediction if available, otherwise run model inference
            if cached_predictions and cached_predictions.segmentation:
                mask = cached_predictions.segmentation
                if debug:
                    print("🔍 Using cached segmentation prediction")
            else:
                mask = ImageTransforms.segment_image(
                    current_image,
                    debug=debug,
                    debug_dir=output_dir,
                    base_filename=base_filename
                )
            
            if mask is None:
                print(f"⚠️  Segmentation failed for: {base_filename} - no bat face detected")
                return None
            
            # Only cache if not already cached
            if pose_prediction or mask:
                self._cache_predictions(cached_predictions_key, self._create_prediction_bundle(
                        cached_predictions_key, 
                        segmentation_prediction=mask, 
                        pose_prediction=pose_prediction
                    ))
                if debug:
                        print("🔍 Cached predictions for future use")
                    
            
            # Step 3: Crop square around segmented region using the mask
            cropping_result = ImageTransforms.crop_square_around_segmentation_mask(
                current_image,
                mask,
                margin_ratio=self.face_outer_margin_ratio,
                debug=debug,
                debug_dir=output_dir,
                base_filename=base_filename
            )
            
            if cropping_result is None or cropping_result[0] is None:
                print(f"⚠️  Cropping failed for: {base_filename}")
                return None
            
            current_image = cropping_result[0]  # Extract the cropped image from the tuple
            transformed_mask = cropping_result[1]  # Extract the transformed mask from the tuple

            # Step 4: Use the transformed mask from cropping for background replacement
            # The cropping method already provides the mask transformed to match the cropped image
            cropped_mask = None
            if transformed_mask is not None and hasattr(transformed_mask, 'mask'):
                cropped_mask = transformed_mask.mask
                if debug:
                    print(f"🔍 Using transformed mask from cropping: {cropped_mask.shape}")
                    print(f"🔍 Mask type: {type(cropped_mask)}")
            else:
                print("⚠️ No transformed mask available from cropping")

            # Step 5: Replace background using the cropped mask (if available)
            if debug:
                print("🔍 Step 5: Replacing background...")
            
            if self.background_enabled and cropped_mask is not None:
                if debug:
                    print(f"🔍 About to call apply_background_replacement with:")
                    print(f"   - current_image shape: {current_image.shape}")
                    print(f"   - cropped_mask type: {type(cropped_mask)}")
                    print(f"   - cropped_mask shape: {cropped_mask.shape}")
                    print(f"   - cropped_mask dtype: {cropped_mask.dtype}")
                    print(f"   - cropped_mask min/max: {cropped_mask.min()}/{cropped_mask.max()}")
                
                background_replaced = ImageTransforms.apply_background_replacement(
                    current_image,
                    self.background_generator,
                    cropped_mask,
                )
                
                if background_replaced is not None:
                    # Extract the image from the tuple (result, updated_prediction)
                    if isinstance(background_replaced, tuple) and len(background_replaced) >= 1:
                        current_image = background_replaced[0]  # Extract the image result
                    else:
                        current_image = background_replaced  # Fallback if not a tuple
                    
                    if debug:
                        print("🔍 Background replacement successful!")
                        
                        # Save background replaced image
                        debug_path = os.path.join(output_dir, f"{base_filename}_step5_background_replaced.jpg")
                        cv2.imwrite(debug_path, current_image)
                        print(f"🔍 Saved debug background replaced image: {debug_path}")
                else:
                    print("⚠️ Background replacement failed, continuing without...")
            elif not self.background_enabled:
                if debug:
                    print("🔍 Background replacement disabled, skipping...")
            else:
                print("⚠️ No mask available, skipping background replacement...")

            # Step 6: Resize to target size
            resized_image = ImageTransforms.resize_square_image(
                current_image, 
                self.target_size, 
                interpolation="bilinear",
                debug=debug,
                debug_dir=output_dir,
                base_filename=base_filename
            )
            
            if resized_image is None:
                print(f"⚠️  Resizing failed for: {base_filename}")
                return None
            
            current_image = resized_image

            # Step 7: Normalize to [0,1] range
            normalized_image = ImageTransforms.normalize_image(
                current_image, 
                scale=self.normalize_scale,
                debug=debug,
                debug_dir=output_dir,
                base_filename=base_filename
            )
            
            if normalized_image is None:
                print(f"⚠️  Normalization failed for: {base_filename}")
                return None
            
            return normalized_image

        except Exception as e:
            print(f"❌ Error processing {base_filename}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocess_single_image_with_augmentation(self, image_path: str, debug: bool = False, 
                                                output_dir: str = None) -> List[np.ndarray]:
        """Process single image with optional augmentation.
        
        This method processes the original image and then creates augmented versions
        using the cached predictions to avoid redundant model inference.
        """
        try:
            image_array = cv2.imread(image_path)
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            result = self.preprocess_single_image(image_array, base_filename, image_path, debug, output_dir)
            
            if result is None:
                return []
            
            result_images = [result]  # Always include original processed image
            
            # Apply augmentation if enabled
            if self.augmentation_enabled:
                augmented_images = self.apply_augmentation(
                    image_array,
                    self.augmentation_preset,
                    self.augmentation_count
                )
                for augmented_image in augmented_images:
                    try:
                        result = self.preprocess_single_image(augmented_image, base_filename, image_path, debug, output_dir)
                        if result is not None:
                            result_images.append(result)
                        else:
                            print(f"⚠️ Augmented image {len(result_images)+1} failed preprocessing (no bat face detected)")
                    except Exception as e:
                        print(f"⚠️ Augmented image {len(result_images)+1} failed preprocessing: {e}")
                        continue
            
            return result_images
            
        except Exception as e:
            print(f"❌ Error processing with augmentation {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def preprocess_batch(self, input_dir: str, output_dir: str, debug: bool = False, debug_dir: str = None):
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")

        results = []
        successful = 0
        total_files = 0
        failed_files = []

        # Count total image files first
        for dirpath, dirnames, filenames in os.walk(input_dir):
            for filename in filenames:
                if is_img_file(filename):
                    total_files += 1

        if total_files == 0:
            print(f"⚠️  No image files found in {input_dir}")
            return []

        bg_info = f"using {self.background_generator.__name__} backgrounds" if self.background_enabled else "with original backgrounds"
        print(
            f"🔄 Processing {total_files} images with advanced pipeline {bg_info}..."
        )
        print(f"🔧 Config: margin_ratio={self.face_outer_margin_ratio}, target_size={self.target_size}, normalize_scale={self.normalize_scale}")

        # Process all images recursively
        for dirpath, dirnames, filenames in os.walk(input_dir):
            # Create corresponding output directory structure
            relative_path = os.path.relpath(dirpath, input_dir)
            current_output_dir = (
                os.path.join(output_dir, relative_path)
                if relative_path != "."
                else output_dir
            )

            for filename in filenames:
                if is_img_file(filename):
                    image_path = os.path.join(dirpath, filename)
                    relative_image_path = os.path.relpath(image_path, input_dir)
                    print(f"\n📷 Processing: {relative_image_path}...")

                    # Create debug directory for this image if debug is enabled
                    if debug:
                        if debug_dir:
                            # Create subdirectory for each image to avoid conflicts
                            image_debug_dir = os.path.join(debug_dir, os.path.splitext(filename)[0])
                            os.makedirs(image_debug_dir, exist_ok=True)
                        else:
                            # Fallback to default behavior
                            image_debug_dir = os.path.join(output_dir, "debug_steps", os.path.splitext(filename)[0])
                            os.makedirs(image_debug_dir, exist_ok=True)
                    else:
                        image_debug_dir = None
                    
                    # Process image (will automatically use cached predictions if available)
                    if self.augmentation_enabled:
                        # Use augmentation-enabled processing
                        augmentation_results = self.preprocess_single_image_with_augmentation(
                            image_path, debug=debug, output_dir=image_debug_dir
                        )
                        
                        if augmentation_results and len(augmentation_results) > 0:
                            # Save all versions (original + augmented)
                            base_name = os.path.splitext(filename)[0]
                            
                            for i, result in enumerate(augmentation_results):
                                if i == 0:
                                    # Original processed image
                                    output_name = f"processed_{filename}"
                                    results.append((relative_image_path, result))
                                    successful += 1
                                else:
                                    # Augmented image
                                    output_name = f"processed_{base_name}_aug{i:03d}.jpg"
                                    results.append((f"{relative_image_path}_aug{i:03d}", result))
                                    successful += 1
                                
                                # Save the image
                                output_image = (result * 255).astype(np.uint8)
                                output_file = os.path.join(current_output_dir, output_name)
                                success = cv2.imwrite(output_file, output_image)
                                if success:
                                    print(f"✅ Saved: {output_name}")
                                else:
                                    print(f"❌ Failed to save: {output_name}")
                        else:
                            failed_files.append(relative_image_path)
                            print(f"❌ Failed to process: {relative_image_path}")
                    else:
                        # Standard processing without augmentation
                        result = self.preprocess_single_image_from_path(image_path, debug=debug, output_dir=image_debug_dir)

                        if result is not None:
                            results.append((relative_image_path, result))
                            successful += 1

                            # Save processed image
                            os.makedirs(current_output_dir, exist_ok=True)
                            # Convert back to uint8 for saving
                            output_image = (result * 255).astype(np.uint8)
                            output_file = os.path.join(
                                current_output_dir, f"processed_{filename}"
                            )
                            success = cv2.imwrite(output_file, output_image)
                            if success:
                                print(f"✅ Saved: {output_file}")
                            else:
                                print(f"❌ Failed to save: {output_file}")
                        else:
                            failed_files.append(relative_image_path)
                            print(f"❌ Failed to process: {relative_image_path}")

        print(f"\n🎯 BATCH PROCESSING COMPLETE!")
        print(f"✅ Successfully processed: {successful}/{total_files} images")
        if successful < total_files:
            print(f"⚠️  Failed: {total_files - successful} images")
            if failed_files:
                print("Failed files:")
                for failed_file in failed_files[:10]:  # Show first 10 failed files
                    print(f"  - {failed_file}")
                if len(failed_files) > 10:
                    print(f"  ... and {len(failed_files) - 10} more")
        
        return results


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Siamese Network Advanced Image Processor
    
    Advanced preprocessing pipeline for siamese network with segmentation and pose-based processing.
    This pipeline uses YOLO segmentation to detect bat faces and YOLO pose for face alignment.
    """
    pass


@cli.command()
@click.option("--input", "-i", type=click.Path(exists=True, file_okay=True, dir_okay=False), 
              required=True, help="Single input image path")
@click.option("--output", "-o", type=click.Path(file_okay=True, dir_okay=False), 
              required=True, help="Output path for single image")
@click.option("--debug", "-d", is_flag=True, help="Enable debug output")
@click.option("--debug-dir", type=click.Path(file_okay=False, dir_okay=True), 
              help="Directory to save debug step images (optional when using --debug)")
@click.option("--augment", "-a", is_flag=True, help="Enable augmentation")
@click.option("--augment-count", type=int, default=5, help="Number of augmented versions to create")
def process_single(input, output, debug, debug_dir, augment, augment_count):
    """Process a single image through the advanced preprocessing pipeline."""
    
    # Set default debug directory if debug is enabled but no debug_dir provided
    if debug and not debug_dir:
        debug_dir = os.path.join(os.path.dirname(output), "debug_steps")
        print(f"🔧 Using default debug directory: {debug_dir}")
    
    # Initialize advanced pipeline
    pipeline = SiamesePreprocessingPipeline()
    
    # thresholds and background come from config
    
    try:
        print(f"🔄 Processing single image: {input}")
        
        if augment:
            print(f"🔧 Augmentation enabled: {augment_count} versions")
            results = pipeline.preprocess_single_image_with_augmentation(
                input, debug=debug, output_dir=debug_dir
            )
            
            if results:
                print(f"✅ Processing successful! Created {len(results)} images (1 original + {len(results)-1} augmented)")
                
                # Save all versions
                base_name = os.path.splitext(os.path.basename(output))[0]
                output_dir = os.path.dirname(output)
                
                for i, result in enumerate(results):
                    if i == 0:
                        output_name = f"{base_name}_processed.jpg"
                    else:
                        output_name = f"{base_name}_processed_aug{i:03d}.jpg"
                    
                    output_path = os.path.join(output_dir, output_name)
                    output_image = (result * 255).astype(np.uint8)
                    success = cv2.imwrite(output_path, output_image)
                    
                    if success:
                        print(f"💾 Saved: {output_path}")
                    else:
                        print(f"❌ Failed to save: {output_path}")
            else:
                print("❌ Processing failed - no images created")
        else:
            # Standard processing without augmentation
            result = pipeline.preprocess_single_image(input, debug=debug, output_dir=debug_dir)
            
            if result is not None:
                print("✅ Processing successful!")
                print(f"   Output shape: {result.shape}, dtype: {result.dtype}")
                print(f"   Value range: [{result.min():.3f}, {result.max():.3f}]")
                
                # Save processed image
                output_image = (result * 255).astype(np.uint8)
                success = cv2.imwrite(output, output_image)
                if success:
                    print(f"💾 Saved processed image to: {output}")
                else:
                    print(f"❌ Failed to save image to: {output}")
            else:
                print("❌ Processing failed - no bat face detected or processing error")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option("--input", "-i", type=click.Path(exists=True, file_okay=True, dir_okay=False), 
              required=True, help="Input image path")
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, dir_okay=True), 
              required=True, help="Output directory for augmented images")
@click.option("--debug", "-d", is_flag=True, help="Enable debug output")
@click.option("--debug-dir", type=click.Path(file_okay=False, dir_okay=True), 
              help="Directory to save debug step images")
@click.option("--augment-count", type=int, default=5, help="Number of augmented versions to create")
@click.option("--preset", type=str, default="noise_color_only", help="Augmentation preset to use")
def process_with_augmentation(input, output_dir, debug, debug_dir, augment_count, preset):
    """Process a single image with augmentation enabled."""
    
    # Set default debug directory if debug is enabled but no debug_dir provided
    if debug and not debug_dir:
        debug_dir = os.path.join(output_dir, "debug_steps")
        print(f"🔧 Using default debug directory: {debug_dir}")
    
    # Initialize advanced pipeline
    pipeline = SiamesePreprocessingPipeline()
    
    try:
        print(f"🔄 Processing image with augmentation: {input}")
        print(f"🔧 Augmentation preset: {preset}, Count: {augment_count}")
        
        # Temporarily override augmentation settings
        pipeline.augmentation_enabled = True
        pipeline.augmentation_preset = preset
        pipeline.augmentation_count = augment_count
        
        results = pipeline.preprocess_single_image_with_augmentation(
            input, debug=debug, output_dir=debug_dir
        )
        
        if results:
            print(f"✅ Processing successful! Created {len(results)} images (1 original + {len(results)-1} augmented)")
            
            # Save all versions
            base_name = os.path.splitext(os.path.basename(input))[0]
            os.makedirs(output_dir, exist_ok=True)
            
            for i, result in enumerate(results):
                if i == 0:
                    output_name = f"{base_name}_processed.jpg"
                else:
                    output_name = f"{base_name}_processed_aug{i:03d}.jpg"
                
                output_path = os.path.join(output_dir, output_name)
                output_image = (result * 255).astype(np.uint8)
                success = cv2.imwrite(output_path, output_image)
                
                if success:
                    print(f"💾 Saved: {output_path}")
                else:
                    print(f"❌ Failed to save: {output_path}")
        else:
            print("❌ Processing failed - no images created")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option("--input-dir", "-i", type=click.Path(exists=True, file_okay=False, dir_okay=True), 
              required=True, help="Input directory with images")
@click.option("--output-dir", "-o", type=click.Path(file_okay=False, dir_okay=True), 
              required=True, help="Output directory for batch processing")
@click.option("--debug", "-d", is_flag=True, help="Enable debug output")
@click.option("--debug-dir", type=click.Path(file_okay=False, dir_okay=True), 
              help="Directory to save debug step images (optional when using --debug)")
def process_batch(input_dir, output_dir, debug, debug_dir):
    """Process multiple images through the advanced preprocessing pipeline."""
    
    # Set default debug directory if debug is enabled but no debug_dir provided
    if debug and not debug_dir:
        debug_dir = os.path.join(output_dir, "debug_steps")
        print(f"🔧 Using default debug directory: {debug_dir}")
    
    # Initialize advanced pipeline
    pipeline = SiamesePreprocessingPipeline()
    
    # thresholds and background come from config
    
    try:
        print(f"🔄 Processing batch: {input_dir}")
        results = pipeline.preprocess_batch(input_dir, output_dir, debug=debug, debug_dir=debug_dir)
        print(f"✅ Batch processing complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
