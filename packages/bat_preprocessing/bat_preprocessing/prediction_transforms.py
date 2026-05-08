"""Prediction transformation utilities.

Geometric helpers for transforming masks, keypoints, and bounding boxes so
they remain consistent with image transformations applied during the
preprocessing pipeline. Pure numpy + OpenCV — ported from
``app/siamese_preprocessing/prediction_transforms.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .prediction_structures import (
    PosePrediction,
    PredictionBundle,
    SegmentationPrediction,
)


@dataclass
class TransformationMatrix:
    """A 2-D transformation matrix (2x3 affine or 3x3 perspective)."""

    matrix: np.ndarray
    matrix_type: str  # "affine", "perspective", "custom"

    def __post_init__(self) -> None:
        if self.matrix.shape not in [(2, 3), (3, 3)]:
            raise ValueError(
                f"Invalid matrix shape: {self.matrix.shape}. Expected (2,3) or (3,3)"
            )

    def transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        if self.matrix.shape == (2, 3):
            x, y = point
            new_x = self.matrix[0, 0] * x + self.matrix[0, 1] * y + self.matrix[0, 2]
            new_y = self.matrix[1, 0] * x + self.matrix[1, 1] * y + self.matrix[1, 2]
            return (float(new_x), float(new_y))
        x, y = point
        homog = np.array([x, y, 1.0])
        transformed = self.matrix @ homog
        if transformed[2] != 0:
            return (
                float(transformed[0] / transformed[2]),
                float(transformed[1] / transformed[2]),
            )
        return (float(transformed[0]), float(transformed[1]))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.ndim == 1:
            return np.array(self.transform_point(tuple(points)))
        if points.ndim == 2:
            if points.shape[1] == 2:
                return np.array(
                    [self.transform_point(tuple(p)) for p in points], dtype=np.float32
                )
            if points.shape[1] == 3:
                out = []
                for p in points:
                    x, y = self.transform_point((float(p[0]), float(p[1])))
                    out.append([x, y, float(p[2])])
                return np.array(out, dtype=np.float32)
        raise ValueError(f"Invalid points shape: {points.shape}")


class MaskTransformer:
    """Geometric transforms for binary segmentation masks."""

    @staticmethod
    def resize_mask(
        mask: np.ndarray,
        target_size: Tuple[int, int],
        interpolation: str = "nearest",
    ) -> np.ndarray:
        methods = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }
        method = methods.get(interpolation, cv2.INTER_NEAREST)
        resized = cv2.resize(mask, target_size, interpolation=method)
        if interpolation == "nearest":
            resized = (resized > 0.5).astype(np.uint8)
        return resized

    @staticmethod
    def crop_mask(
        mask: np.ndarray, crop_region: Tuple[int, int, int, int]
    ) -> np.ndarray:
        x1, y1, x2, y2 = crop_region
        return mask[y1:y2, x1:x2]

    @staticmethod
    def rotate_mask(
        mask: np.ndarray,
        angle: float,
        center: Optional[Tuple[float, float]] = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        if center is None:
            h, w = mask.shape[:2]
            center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(
            mask, rotation_matrix, (mask.shape[1], mask.shape[0])
        )
        rotated = (rotated > 0.5).astype(np.uint8)
        return rotated

    @staticmethod
    def apply_affine_transform(
        mask: np.ndarray,
        transform_matrix: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        if output_size is None:
            output_size = (mask.shape[1], mask.shape[0])
        transformed = cv2.warpAffine(mask, transform_matrix, output_size)
        return (transformed > 0.5).astype(np.uint8)

    @staticmethod
    def transform_mask_for_crop(
        mask: np.ndarray,
        crop_region: Tuple[int, int, int, int],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        cropped = MaskTransformer.crop_mask(mask, crop_region)
        if target_size is not None:
            cropped = MaskTransformer.resize_mask(cropped, target_size)
        return cropped


class KeypointTransformer:
    """Geometric transforms for pose keypoints."""

    @staticmethod
    def transform_keypoints_for_resize(
        keypoints: np.ndarray,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        if keypoints.size == 0:
            return keypoints
        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]
        transformed = keypoints.copy()
        transformed[:, 0] *= scale_x
        transformed[:, 1] *= scale_y
        return transformed

    @staticmethod
    def transform_keypoints_for_crop(
        keypoints: np.ndarray,
        crop_region: Tuple[int, int, int, int],
    ) -> np.ndarray:
        if keypoints.size == 0:
            return keypoints
        x1, y1, x2, y2 = crop_region
        transformed = keypoints.copy()
        transformed[:, 0] -= x1
        transformed[:, 1] -= y1
        valid_mask = (
            (transformed[:, 0] >= 0)
            & (transformed[:, 0] < (x2 - x1))
            & (transformed[:, 1] >= 0)
            & (transformed[:, 1] < (y2 - y1))
        )
        return transformed[valid_mask]

    @staticmethod
    def apply_affine_transform(
        keypoints: np.ndarray,
        transform_matrix: np.ndarray,
    ) -> np.ndarray:
        if keypoints.size == 0:
            return keypoints
        xy = keypoints[:, :2].astype(np.float32)
        transformed_xy = cv2.transform(
            xy.reshape(-1, 1, 2), transform_matrix
        ).reshape(-1, 2)
        return np.column_stack([transformed_xy, keypoints[:, 2]])

    @staticmethod
    def transform_keypoints_for_rotation(
        keypoints: np.ndarray,
        angle: float,
        center: Optional[Tuple[float, float]] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        if keypoints.size == 0:
            return keypoints
        if center is None and image_size is not None:
            center = (image_size[0] // 2, image_size[1] // 2)
        elif center is None:
            raise ValueError("Either center or image_size must be provided")
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return KeypointTransformer.apply_affine_transform(keypoints, rotation_matrix)

    @staticmethod
    def validate_keypoints(
        keypoints: np.ndarray, image_size: Tuple[int, int]
    ) -> np.ndarray:
        if keypoints.size == 0:
            return keypoints
        width, height = image_size
        valid = (
            (keypoints[:, 0] >= 0)
            & (keypoints[:, 0] < width)
            & (keypoints[:, 1] >= 0)
            & (keypoints[:, 1] < height)
        )
        return keypoints[valid]


class BoundingBoxTransformer:
    """Geometric transforms for bounding boxes."""

    @staticmethod
    def transform_bbox_for_resize(
        bbox: Tuple[float, float, float, float],
        original_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]
        return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)

    @staticmethod
    def transform_bbox_for_crop(
        bbox: Tuple[float, float, float, float],
        crop_region: Tuple[int, int, int, int],
    ) -> Tuple[float, float, float, float]:
        bx1, by1, bx2, by2 = bbox
        cx1, cy1, cx2, cy2 = crop_region
        nx1 = bx1 - cx1
        ny1 = by1 - cy1
        nx2 = bx2 - cx1
        ny2 = by2 - cy1
        nx1 = max(0, min(nx1, cx2 - cx1))
        ny1 = max(0, min(ny1, cy2 - cy1))
        nx2 = max(nx1, min(nx2, cx2 - cx1))
        ny2 = max(ny1, min(ny2, cy2 - cy1))
        return (nx1, ny1, nx2, ny2)

    @staticmethod
    def transform_bbox_for_rotation(
        bbox: Tuple[float, float, float, float],
        angle: float,
        center: Optional[Tuple[float, float]] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, float, float, float]:
        if center is None and image_size is not None:
            center = (image_size[0] // 2, image_size[1] // 2)
        elif center is None:
            raise ValueError("Either center or image_size must be provided")
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        x1, y1, x2, y2 = bbox
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        )
        transformed = cv2.transform(corners.reshape(-1, 1, 2), rotation_matrix).reshape(
            -1, 2
        )
        return (
            float(np.min(transformed[:, 0])),
            float(np.min(transformed[:, 1])),
            float(np.max(transformed[:, 0])),
            float(np.max(transformed[:, 1])),
        )

    @staticmethod
    def validate_bbox(
        bbox: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        width, height = image_size
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(x1, min(x2, width))
        y2 = max(y1, min(y2, height))
        return (x1, y1, x2, y2)


class CoordinateMapper:
    """Build / combine transformation matrices."""

    @staticmethod
    def create_resize_mapping(
        original_size: Tuple[int, int], target_size: Tuple[int, int]
    ) -> TransformationMatrix:
        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]
        matrix = np.array(
            [[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32
        )
        return TransformationMatrix(matrix, "affine")

    @staticmethod
    def create_crop_mapping(
        crop_region: Tuple[int, int, int, int]
    ) -> TransformationMatrix:
        x1, y1, _, _ = crop_region
        matrix = np.array([[1, 0, -x1], [0, 1, -y1]], dtype=np.float32)
        return TransformationMatrix(matrix, "affine")

    @staticmethod
    def create_rotation_mapping(
        angle: float, center: Tuple[float, float]
    ) -> TransformationMatrix:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return TransformationMatrix(matrix, "affine")

    @staticmethod
    def combine_transformations(
        transformations: List[TransformationMatrix],
    ) -> TransformationMatrix:
        if not transformations:
            raise ValueError("No transformations provided")
        if len(transformations) == 1:
            return transformations[0]

        combined = transformations[0].matrix
        for t in transformations[1:]:
            if combined.shape == (2, 3) and t.matrix.shape == (2, 3):
                combined = t.matrix @ combined
            else:
                if combined.shape == (2, 3):
                    combined = np.vstack([combined, [0, 0, 1]])
                if t.matrix.shape == (2, 3):
                    t_3x3 = np.vstack([t.matrix, [0, 0, 1]])
                else:
                    t_3x3 = t.matrix
                combined = t_3x3 @ combined

        if combined.shape == (3, 3) and np.allclose(combined[2, :], [0, 0, 1]):
            combined = combined[:2, :]
        return TransformationMatrix(combined, "custom")

    @staticmethod
    def map_coordinates(
        coordinates: Union[Tuple[float, float], np.ndarray],
        transformation: TransformationMatrix,
    ) -> Union[Tuple[float, float], np.ndarray]:
        if isinstance(coordinates, tuple):
            return transformation.transform_point(coordinates)
        if isinstance(coordinates, np.ndarray):
            return transformation.transform_points(coordinates)
        raise TypeError(f"Unsupported coordinates type: {type(coordinates)}")


class PredictionTransformer:
    """High-level helpers operating on prediction objects."""

    @staticmethod
    def transform_segmentation_prediction(
        prediction: SegmentationPrediction,
        transformation: TransformationMatrix,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> SegmentationPrediction:
        if target_size is not None:
            transformed_mask = MaskTransformer.resize_mask(prediction.mask, target_size)
        else:
            transformed_mask = prediction.mask

        original_image_shape_wh = (
            prediction.original_image_shape[::-1]
            if prediction.original_image_shape
            else (1, 1)
        )
        transformed_bbox = BoundingBoxTransformer.transform_bbox_for_resize(
            prediction.bounding_box,
            original_image_shape_wh,
            target_size[::-1] if target_size else (1, 1),
        )

        return SegmentationPrediction(
            mask=transformed_mask,
            confidence=prediction.confidence,
            bounding_box=transformed_bbox,
            class_id=prediction.class_id,
            class_name=prediction.class_name,
            original_image_shape=(
                target_size[::-1] if target_size else prediction.original_image_shape
            ),
            model_resolution=prediction.model_resolution,
            timestamp=prediction.timestamp,
        )

    @staticmethod
    def transform_pose_prediction(
        prediction: PosePrediction,
        transformation: TransformationMatrix,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> PosePrediction:
        transformed_keypoints = transformation.transform_points(prediction.keypoints)

        original_image_shape_wh = (
            prediction.original_image_shape[::-1]
            if prediction.original_image_shape
            else (1, 1)
        )
        transformed_bbox = BoundingBoxTransformer.transform_bbox_for_resize(
            prediction.bounding_box,
            original_image_shape_wh,
            target_size[::-1] if target_size else (1, 1),
        )

        return PosePrediction(
            keypoints=transformed_keypoints,
            bounding_box=transformed_bbox,
            confidence=prediction.confidence,
            class_id=prediction.class_id,
            class_name=prediction.class_name,
            original_image_shape=(
                target_size[::-1] if target_size else prediction.original_image_shape
            ),
            model_resolution=prediction.model_resolution,
            timestamp=prediction.timestamp,
        )

    @staticmethod
    def transform_prediction_bundle(
        bundle: PredictionBundle,
        transformation: TransformationMatrix,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> PredictionBundle:
        transformed_seg = (
            PredictionTransformer.transform_segmentation_prediction(
                bundle.segmentation, transformation, target_size
            )
            if bundle.segmentation
            else None
        )
        transformed_pose = (
            PredictionTransformer.transform_pose_prediction(
                bundle.pose, transformation, target_size
            )
            if bundle.pose
            else None
        )
        out = PredictionBundle(
            segmentation=transformed_seg,
            pose=transformed_pose,
            image_path=bundle.image_path,
            image_hash=bundle.image_hash,
            timestamp=bundle.timestamp,
            processing_steps=list(bundle.processing_steps),
            model_versions=dict(bundle.model_versions),
        )
        out.add_processing_step(f"Transformed with {transformation.matrix_type}")
        return out
