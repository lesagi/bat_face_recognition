"""
Prediction transformation utilities for Siamese Network preprocessing pipeline.

This module provides utilities for transforming model predictions (masks, keypoints,
bounding boxes) to match image transformations, enabling prediction reuse across
augmented image processing.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Union, Any
from dataclasses import dataclass

from prediction_structures import (
    SegmentationPrediction, 
    PosePrediction, 
    PredictionBundle
)


@dataclass
class TransformationMatrix:
    """Represents a 2D transformation matrix for coordinate mapping."""
    
    matrix: np.ndarray  # 2x3 or 3x3 transformation matrix
    matrix_type: str    # 'affine', 'perspective', 'custom'
    
    def __post_init__(self):
        """Validate transformation matrix."""
        if self.matrix.shape not in [(2, 3), (3, 3)]:
            raise ValueError(f"Invalid matrix shape: {self.matrix.shape}. Expected (2,3) or (3,3)")
    
    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform a single point using the matrix."""
        if self.matrix.shape == (2, 3):
            # Affine transformation
            x, y = point
            new_x = self.matrix[0, 0] * x + self.matrix[0, 1] * y + self.matrix[0, 2]
            new_y = self.matrix[1, 0] * x + self.matrix[1, 1] * y + self.matrix[1, 2]
            return (float(new_x), float(new_y))
        else:
            # Perspective transformation
            x, y = point
            # Convert to homogeneous coordinates
            point_homog = np.array([x, y, 1.0])
            transformed = self.matrix @ point_homog
            # Convert back from homogeneous coordinates
            if transformed[2] != 0:
                return (float(transformed[0] / transformed[2]), float(transformed[1] / transformed[2]))
            else:
                return (float(transformed[0]), float(transformed[1]))
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform multiple points using the matrix."""
        if len(points.shape) == 1:
            # Single point
            return np.array(self.transform_point(tuple(points)))
        elif len(points.shape) == 2:
            # Multiple points
            if points.shape[1] == 2:
                # Points are (x, y)
                transformed = []
                for point in points:
                    transformed.append(self.transform_point(tuple(point)))
                return np.array(transformed)
            elif points.shape[1] == 3:
                # Points are (x, y, confidence) - preserve confidence
                transformed = []
                for point in points:
                    x, y = self.transform_point((point[0], point[1]))
                    transformed.append([x, y, point[2]])
                return np.array(transformed)
        raise ValueError(f"Invalid points shape: {points.shape}")


class MaskTransformer:
    """Transform segmentation masks to match image transformations."""
    
    @staticmethod
    def resize_mask(
        mask: np.ndarray, 
        target_size: Tuple[int, int], 
        interpolation: str = "nearest"
    ) -> np.ndarray:
        """Resize mask to target dimensions.
        
        Args:
            mask: Input binary mask
            target_size: Target (width, height) dimensions
            interpolation: Interpolation method ('nearest', 'linear', 'cubic')
        
        Returns:
            Resized mask
        """
        # Map interpolation methods
        methods = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA
        }
        method = methods.get(interpolation, cv2.INTER_NEAREST)
        
        # Resize mask
        resized = cv2.resize(mask, target_size, interpolation=method)
        
        # Ensure binary output for nearest interpolation
        if interpolation == "nearest":
            resized = (resized > 0.5).astype(np.uint8)
        
        return resized
    
    @staticmethod
    def crop_mask(
        mask: np.ndarray, 
        crop_region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Crop mask to specified region.
        
        Args:
            mask: Input binary mask
            crop_region: (x1, y1, x2, y2) crop coordinates
        
        Returns:
            Cropped mask
        """
        x1, y1, x2, y2 = crop_region
        return mask[y1:y2, x1:x2]
    
    @staticmethod
    def rotate_mask(
        mask: np.ndarray, 
        angle: float, 
        center: Optional[Tuple[float, float]] = None,
        scale: float = 1.0
    ) -> np.ndarray:
        """Rotate mask by specified angle.
        
        Args:
            mask: Input binary mask
            angle: Rotation angle in degrees (positive = counterclockwise)
            center: Rotation center (x, y). If None, uses mask center
            scale: Scale factor
        
        Returns:
            Rotated mask
        """
        if center is None:
            h, w = mask.shape[:2]
            center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Apply rotation
        rotated = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))
        
        # Ensure binary output
        rotated = (rotated > 0.5).astype(np.uint8)
        
        return rotated
    
    @staticmethod
    def apply_affine_transform(
        mask: np.ndarray, 
        transform_matrix: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Apply affine transformation to mask.
        
        Args:
            mask: Input binary mask
            transform_matrix: 2x3 affine transformation matrix
            output_size: Output size (width, height). If None, uses input size
        
        Returns:
            Transformed mask
        """
        if output_size is None:
            output_size = (mask.shape[1], mask.shape[0])
        
        # Apply transformation
        transformed = cv2.warpAffine(mask, transform_matrix, output_size)
        
        # Ensure binary output
        transformed = (transformed > 0.5).astype(np.uint8)
        
        return transformed
    
    @staticmethod
    def transform_mask_for_crop(
        mask: np.ndarray,
        crop_region: Tuple[int, int, int, int],
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Transform mask for cropping operation.
        
        Args:
            mask: Input binary mask
            crop_region: (x1, y1, x2, y2) crop coordinates
            target_size: Optional target size for output mask
        
        Returns:
            Transformed mask
        """
        # Crop mask
        cropped = MaskTransformer.crop_mask(mask, crop_region)
        
        # Resize if target size specified
        if target_size is not None:
            cropped = MaskTransformer.resize_mask(cropped, target_size)
        
        return cropped


class KeypointTransformer:
    """Transform pose keypoints to match image transformations."""
    
    @staticmethod
    def transform_keypoints_for_resize(
        keypoints: np.ndarray,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Transform keypoints for image resizing.
        
        Args:
            keypoints: Input keypoints array (N, 3) where 3 = (x, y, confidence)
            original_size: Original image size (width, height)
            target_size: Target image size (width, height)
        
        Returns:
            Transformed keypoints
        """
        if keypoints.size == 0:
            return keypoints
        
        # Calculate scale factors
        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]
        
        # Create copy to avoid modifying original
        transformed = keypoints.copy()
        
        # Scale x and y coordinates, preserve confidence
        transformed[:, 0] *= scale_x
        transformed[:, 1] *= scale_y
        
        return transformed
    
    @staticmethod
    def transform_keypoints_for_crop(
        keypoints: np.ndarray,
        crop_region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Transform keypoints for image cropping.
        
        Args:
            keypoints: Input keypoints array (N, 3) where 3 = (x, y, confidence)
            crop_region: (x1, y1, x2, y2) crop coordinates
        
        Returns:
            Transformed keypoints
        """
        if keypoints.size == 0:
            return keypoints
        
        x1, y1, x2, y2 = crop_region
        
        # Create copy to avoid modifying original
        transformed = keypoints.copy()
        
        # Adjust coordinates relative to crop region
        transformed[:, 0] -= x1
        transformed[:, 1] -= y1
        
        # Filter out keypoints outside crop region
        valid_mask = (
            (transformed[:, 0] >= 0) & 
            (transformed[:, 0] < (x2 - x1)) &
            (transformed[:, 1] >= 0) & 
            (transformed[:, 1] < (y2 - y1))
        )
        
        return transformed[valid_mask]
    
    @staticmethod
    def transform_keypoints_for_rotation(
        keypoints: np.ndarray,
        angle: float,
        center: Optional[Tuple[float, float]] = None,
        image_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Transform keypoints for image rotation.
        
        Args:
            keypoints: Input keypoints array (N, 3) where 3 = (x, y, confidence)
            angle: Rotation angle in degrees (positive = counterclockwise)
            center: Rotation center (x, y). If None, uses image center
            image_size: Image size (width, height) for center calculation
        
        Returns:
            Transformed keypoints
        """
        if keypoints.size == 0:
            return keypoints
        
        if center is None and image_size is not None:
            center = (image_size[0] // 2, image_size[1] // 2)
        elif center is None:
            raise ValueError("Either center or image_size must be provided")
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Transform keypoints
        transformed = KeypointTransformer.apply_affine_transform(
            keypoints, rotation_matrix
        )
        
        return transformed
    
    @staticmethod
    def apply_affine_transform(
        keypoints: np.ndarray,
        transform_matrix: np.ndarray
    ) -> np.ndarray:
        """Apply affine transformation to keypoints.
        
        Args:
            keypoints: Input keypoints array (N, 3) where 3 = (x, y, confidence)
            transform_matrix: 2x3 affine transformation matrix
        
        Returns:
            Transformed keypoints
        """
        if keypoints.size == 0:
            return keypoints
        
        # Extract x, y coordinates
        xy_coords = keypoints[:, :2]
        
        # Apply transformation to x, y coordinates
        transformed_xy = cv2.transform(
            xy_coords.reshape(-1, 1, 2), 
            transform_matrix
        ).reshape(-1, 2)
        
        # Create output array with transformed coordinates and original confidence
        transformed = np.column_stack([transformed_xy, keypoints[:, 2]])
        
        return transformed
    
    @staticmethod
    def validate_keypoints(
        keypoints: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """Validate and filter keypoints to ensure they're within image bounds.
        
        Args:
            keypoints: Input keypoints array (N, 3) where 3 = (x, y, confidence)
            image_size: Image size (width, height)
        
        Returns:
            Valid keypoints within image bounds
        """
        if keypoints.size == 0:
            return keypoints
        
        width, height = image_size
        
        # Check bounds
        valid_mask = (
            (keypoints[:, 0] >= 0) & 
            (keypoints[:, 0] < width) &
            (keypoints[:, 1] >= 0) & 
            (keypoints[:, 1] < height)
        )
        
        return keypoints[valid_mask]


class BoundingBoxTransformer:
    """Transform bounding boxes to match image transformations."""
    
    @staticmethod
    def transform_bbox_for_resize(
        bbox: Tuple[float, float, float, float],
        original_size: Tuple[int, int],
        target_size: Tuple[int, int]
    ) -> Tuple[float, float, float, float]:
        """Transform bounding box for image resizing.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            original_size: Original image size (width, height)
            target_size: Target image size (width, height)
        
        Returns:
            Transformed bounding box
        """
        x1, y1, x2, y2 = bbox
        orig_w, orig_h = original_size
        target_w, target_h = target_size
        
        # Calculate scale factors
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # Scale coordinates
        new_x1 = x1 * scale_x
        new_y1 = y1 * scale_y
        new_x2 = x2 * scale_x
        new_y2 = y2 * scale_y
        
        return (new_x1, new_y1, new_x2, new_y2)
    
    @staticmethod
    def transform_bbox_for_crop(
        bbox: Tuple[float, float, float, float],
        crop_region: Tuple[int, int, int, int]
    ) -> Tuple[float, float, float, float]:
        """Transform bounding box for image cropping.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            crop_region: (x1, y1, x2, y2) crop coordinates
        
        Returns:
            Transformed bounding box
        """
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
        
        # Adjust coordinates relative to crop region
        new_x1 = bbox_x1 - crop_x1
        new_y1 = bbox_y1 - crop_y1
        new_x2 = bbox_x2 - crop_x1
        new_y2 = bbox_y2 - crop_y1
        
        # Clamp to crop region bounds
        new_x1 = max(0, min(new_x1, crop_x2 - crop_x1))
        new_y1 = max(0, min(new_y1, crop_y2 - crop_y1))
        new_x2 = max(new_x1, min(new_x2, crop_x2 - crop_x1))
        new_y2 = max(new_y1, min(new_y2, crop_y2 - crop_y1))
        
        return (new_x1, new_y1, new_x2, new_y2)
    
    @staticmethod
    def transform_bbox_for_rotation(
        bbox: Tuple[float, float, float, float],
        angle: float,
        center: Optional[Tuple[float, float]] = None,
        image_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[float, float, float, float]:
        """Transform bounding box for image rotation.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            angle: Rotation angle in degrees (positive = counterclockwise)
            center: Rotation center (x, y). If None, uses image center
            image_size: Image size (width, height) for center calculation
        
        Returns:
            Transformed bounding box
        """
        if center is None and image_size is not None:
            center = (image_size[0] // 2, image_size[1] // 2)
        elif center is None:
            raise ValueError("Either center or image_size must be provided")
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Transform bbox corners
        x1, y1, x2, y2 = bbox
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.float32)
        
        # Apply rotation
        transformed_corners = cv2.transform(
            corners.reshape(-1, 1, 2), 
            rotation_matrix
        ).reshape(-1, 2)
        
        # Calculate new bounding box from transformed corners
        new_x1 = float(np.min(transformed_corners[:, 0]))
        new_y1 = float(np.min(transformed_corners[:, 1]))
        new_x2 = float(np.max(transformed_corners[:, 0]))
        new_y2 = float(np.max(transformed_corners[:, 1]))
        
        return (new_x1, new_y1, new_x2, new_y2)
    
    @staticmethod
    def validate_bbox(
        bbox: Tuple[float, float, float, float],
        image_size: Tuple[int, int]
    ) -> Tuple[float, float, float, float]:
        """Validate and clamp bounding box to image bounds.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            image_size: Image size (width, height)
        
        Returns:
            Validated bounding box within image bounds
        """
        x1, y1, x2, y2 = bbox
        width, height = image_size
        
        # Clamp to image bounds
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(x1, min(x2, width))
        y2 = max(y1, min(y2, height))
        
        return (x1, y1, x2, y2)


class CoordinateMapper:
    """Map coordinates between different coordinate systems and transformations."""
    
    @staticmethod
    def create_resize_mapping(
        original_size: Tuple[int, int],
        target_size: Tuple[int, int]
    ) -> TransformationMatrix:
        """Create transformation matrix for resizing.
        
        Args:
            original_size: Original image size (width, height)
            target_size: Target image size (width, height)
        
        Returns:
            Transformation matrix for resizing
        """
        # Simple scaling transformation
        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]
        
        matrix = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0]
        ], dtype=np.float32)
        
        return TransformationMatrix(matrix, 'affine')
    
    @staticmethod
    def create_crop_mapping(
        crop_region: Tuple[int, int, int, int]
    ) -> TransformationMatrix:
        """Create transformation matrix for cropping.
        
        Args:
            crop_region: (x1, y1, x2, y2) crop coordinates
        
        Returns:
            Transformation matrix for cropping
        """
        x1, y1, x2, y2 = crop_region
        
        # Translation transformation
        matrix = np.array([
            [1, 0, -x1],
            [0, 1, -y1]
        ], dtype=np.float32)
        
        return TransformationMatrix(matrix, 'affine')
    
    @staticmethod
    def create_rotation_mapping(
        angle: float,
        center: Tuple[float, float]
    ) -> TransformationMatrix:
        """Create transformation matrix for rotation.
        
        Args:
            angle: Rotation angle in degrees (positive = counterclockwise)
            center: Rotation center (x, y)
        
        Returns:
            Transformation matrix for rotation
        """
        # Get OpenCV rotation matrix
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        return TransformationMatrix(matrix, 'affine')
    
    @staticmethod
    def combine_transformations(
        transformations: List[TransformationMatrix]
    ) -> TransformationMatrix:
        """Combine multiple transformations into a single matrix.
        
        Args:
            transformations: List of transformation matrices
        
        Returns:
            Combined transformation matrix
        """
        if not transformations:
            raise ValueError("No transformations provided")
        
        if len(transformations) == 1:
            return transformations[0]
        
        # Start with first transformation
        combined = transformations[0].matrix
        
        # Apply subsequent transformations
        for transform in transformations[1:]:
            if combined.shape == (2, 3) and transform.matrix.shape == (2, 3):
                # Both are affine, combine them
                combined = transform.matrix @ combined
            else:
                # Convert to 3x3 for perspective transformations
                if combined.shape == (2, 3):
                    combined = np.vstack([combined, [0, 0, 1]])
                if transform.matrix.shape == (2, 3):
                    transform_3x3 = np.vstack([transform.matrix, [0, 0, 1]])
                else:
                    transform_3x3 = transform.matrix
                
                combined = transform_3x3 @ combined
        
        # Convert back to 2x3 if it's still affine
        if combined.shape == (3, 3) and np.allclose(combined[2, :], [0, 0, 1]):
            combined = combined[:2, :]
        
        return TransformationMatrix(combined, 'custom')
    
    @staticmethod
    def map_coordinates(
        coordinates: Union[Tuple[float, float], np.ndarray],
        transformation: TransformationMatrix
    ) -> Union[Tuple[float, float], np.ndarray]:
        """Map coordinates using transformation matrix.
        
        Args:
            coordinates: Single coordinate (x, y) or array of coordinates
            transformation: Transformation matrix to apply
        
        Returns:
            Transformed coordinates
        """
        if isinstance(coordinates, tuple):
            return transformation.transform_point(coordinates)
        elif isinstance(coordinates, np.ndarray):
            return transformation.transform_points(coordinates)
        else:
            raise TypeError(f"Unsupported coordinates type: {type(coordinates)}")


class PredictionTransformer:
    """High-level interface for transforming prediction objects."""
    
    @staticmethod
    def transform_segmentation_prediction(
        prediction: SegmentationPrediction,
        transformation: TransformationMatrix,
        target_size: Optional[Tuple[int, int]] = None
    ) -> SegmentationPrediction:
        """Transform segmentation prediction using transformation matrix.
        
        Args:
            prediction: Input segmentation prediction
            transformation: Transformation matrix to apply
            target_size: Optional target size for output
        
        Returns:
            Transformed segmentation prediction
        """
        # Transform mask
        if target_size is not None:
            transformed_mask = MaskTransformer.resize_mask(
                prediction.mask, target_size
            )
        else:
            transformed_mask = prediction.mask
        
        # Transform bounding box
        transformed_bbox = BoundingBoxTransformer.transform_bbox_for_resize(
            prediction.bounding_box,
            prediction.original_image_shape[::-1] if prediction.original_image_shape else (1, 1),
            target_size[::-1] if target_size else (1, 1)
        )
        
        # Create new prediction object
        return SegmentationPrediction(
            mask=transformed_mask,
            confidence=prediction.confidence,
            bounding_box=transformed_bbox,
            class_id=prediction.class_id,
            class_name=prediction.class_name,
            original_image_shape=target_size[::-1] if target_size else prediction.original_image_shape,
            model_resolution=prediction.model_resolution,
            timestamp=prediction.timestamp
        )
    
    @staticmethod
    def transform_pose_prediction(
        prediction: PosePrediction,
        transformation: TransformationMatrix,
        target_size: Optional[Tuple[int, int]] = None
    ) -> PosePrediction:
        """Transform pose prediction using transformation matrix.
        
        Args:
            prediction: Input pose prediction
            transformation: Transformation matrix to apply
            target_size: Optional target size for output
        
        Returns:
            Transformed pose prediction
        """
        # Transform keypoints
        transformed_keypoints = transformation.transform_points(prediction.keypoints)
        
        # Transform bounding box
        transformed_bbox = BoundingBoxTransformer.transform_bbox_for_resize(
            prediction.bounding_box,
            prediction.original_image_shape[::-1] if prediction.original_image_shape else (1, 1),
            target_size[::-1] if target_size else (1, 1)
        )
        
        # Create new prediction object
        return PosePrediction(
            keypoints=transformed_keypoints,
            bounding_box=transformed_bbox,
            confidence=prediction.confidence,
            class_id=prediction.class_id,
            class_name=prediction.class_name,
            original_image_shape=target_size[::-1] if target_size else prediction.original_image_shape,
            model_resolution=prediction.model_resolution,
            timestamp=prediction.timestamp
        )
    
    @staticmethod
    def transform_prediction_bundle(
        bundle: PredictionBundle,
        transformation: TransformationMatrix,
        target_size: Optional[Tuple[int, int]] = None
    ) -> PredictionBundle:
        """Transform entire prediction bundle using transformation matrix.
        
        Args:
            bundle: Input prediction bundle
            transformation: Transformation matrix to apply
            target_size: Optional target size for output
        
        Returns:
            Transformed prediction bundle
        """
        # Transform segmentation prediction if available
        transformed_segmentation = None
        if bundle.segmentation:
            transformed_segmentation = PredictionTransformer.transform_segmentation_prediction(
                bundle.segmentation, transformation, target_size
            )
        
        # Transform pose prediction if available
        transformed_pose = None
        if bundle.pose:
            transformed_pose = PredictionTransformer.transform_pose_prediction(
                bundle.pose, transformation, target_size
            )
        
        # Create new bundle
        transformed_bundle = PredictionBundle(
            segmentation=transformed_segmentation,
            pose=transformed_pose,
            image_path=bundle.image_path,
            image_hash=bundle.image_hash,
            timestamp=bundle.timestamp,
            processing_steps=bundle.processing_steps.copy(),
            model_versions=bundle.model_versions.copy()
        )
        
        # Add transformation step
        transformed_bundle.add_processing_step(f"Transformed with {transformation.matrix_type}")
        
        return transformed_bundle
