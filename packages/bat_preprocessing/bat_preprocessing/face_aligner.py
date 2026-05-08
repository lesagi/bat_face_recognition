"""Face alignment using YOLO segmentation masks and pose landmarks.

Given an input image plus a :class:`SegmentationPrediction` and (optionally)
a :class:`PosePrediction`, produce a square aligned crop of configurable
edge length.

Pipeline:

1. **Rotate** image so the line between leftmost/rightmost pose keypoints
   becomes horizontal (drop step if no pose).
2. **Crop** a square region centred on the segmentation mask, with
   ``margin_ratio`` padding around the mask's bounding box.
3. **Resize** to ``edge_length`` × ``edge_length``.

This is a numpy / OpenCV-only implementation — no TensorFlow.
"""

from __future__ import annotations

import cv2
import numpy as np

from .prediction_structures import PosePrediction, SegmentationPrediction


class FaceAligner:
    """Align and square-crop a bat face."""

    def __init__(
        self,
        edge_length: int = 224,
        margin_ratio: float = 0.5,
        interpolation: str = "bilinear",
        pad_color: tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        """Construct a :class:`FaceAligner`.

        Args:
            edge_length: Output square edge length, in pixels.
            margin_ratio: Margin around the mask bounding box, as a fraction
                of the largest mask dimension.
            interpolation: One of ``"bilinear"``, ``"nearest"``, ``"cubic"``,
                ``"area"``.
            pad_color: BGR fill colour used when the crop runs outside the
                source image.
        """
        if edge_length <= 0:
            raise ValueError(f"edge_length must be positive, got {edge_length}")
        if margin_ratio < 0:
            raise ValueError(f"margin_ratio must be >= 0, got {margin_ratio}")
        self.edge_length = int(edge_length)
        self.margin_ratio = float(margin_ratio)
        self.interpolation = interpolation
        self.pad_color = pad_color

    # ---- public API ----------------------------------------------------

    def align(
        self,
        image: np.ndarray,
        segmentation: SegmentationPrediction,
        pose: PosePrediction | None = None,
    ) -> np.ndarray | None:
        """Run the full align-and-crop pipeline.

        Args:
            image: Source image, BGR uint8 ``(H, W, 3)``.
            segmentation: Mask used for cropping.
            pose: Optional landmarks used for rotation alignment.

        Returns:
            Aligned ``(edge_length, edge_length, 3)`` image, or ``None`` if
            alignment fails (e.g. empty mask).
        """
        if image is None or image.ndim != 3:
            return None

        current = image
        current_mask = segmentation.mask

        # Step 1: rotation alignment via pose keypoints
        if pose is not None:
            rotated = self._rotate_to_align(current, pose)
            if rotated is not None:
                current = rotated
                # Rotate the mask in tandem so the crop step still lines up
                current_mask = self._rotate_mask(segmentation.mask, pose)

        # Step 2: square crop around the mask
        cropped = self._crop_square_around_mask(current, current_mask)
        if cropped is None:
            return None

        # Step 3: resize to target edge length
        return self._resize(cropped, self.edge_length)

    # ---- rotation -------------------------------------------------------

    @staticmethod
    def _rotation_params(
        pose: PosePrediction,
    ) -> tuple[tuple[float, float], float] | None:
        """Compute (center, angle_degrees) from pose keypoints, or None."""
        kpts = pose.keypoints
        if kpts is None or kpts.size == 0:
            return None
        valid = kpts[kpts[:, 2] > 0.0]
        if len(valid) < 2:
            return None

        left_idx = int(np.argmin(valid[:, 0]))
        right_idx = int(np.argmax(valid[:, 0]))
        left = valid[left_idx][:2]
        right = valid[right_idx][:2]

        if np.any(np.isnan(left)) or np.any(np.isnan(right)) or np.allclose(left, right):
            return None

        dy = float(right[1] - left[1])
        dx = float(right[0] - left[0])
        angle = float(np.degrees(np.arctan2(dy, dx)))
        center = (
            float((left[0] + right[0]) / 2.0),
            float((left[1] + right[1]) / 2.0),
        )
        return center, angle

    def _rotate_to_align(self, image: np.ndarray, pose: PosePrediction) -> np.ndarray | None:
        params = self._rotation_params(pose)
        if params is None:
            return None
        center, angle = params
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            rot,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _rotate_mask(self, mask: np.ndarray, pose: PosePrediction) -> np.ndarray:
        params = self._rotation_params(pose)
        if params is None:
            return mask
        center, angle = params
        rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            mask,
            rot,
            (mask.shape[1], mask.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return (rotated > 0).astype(np.uint8)

    # ---- crop -----------------------------------------------------------

    def _crop_square_around_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        binary = mask if mask.max() <= 1 else (mask > 127).astype(np.uint8)

        ys, xs = np.where(binary > 0)
        if xs.size == 0:
            return None

        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())

        mask_w = max_x - min_x + 1
        mask_h = max_y - min_y + 1
        max_dim = max(mask_w, mask_h)
        square_size = int(round(max_dim * (1.0 + 2.0 * self.margin_ratio)))
        if square_size <= 0:
            return None

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        # Ideal crop region (may extend beyond the source image)
        ideal_x1 = int(round(center_x - square_size / 2.0))
        ideal_y1 = int(round(center_y - square_size / 2.0))
        ideal_x2 = ideal_x1 + square_size
        ideal_y2 = ideal_y1 + square_size

        # Clamp source-side
        src_x1 = max(0, ideal_x1)
        src_y1 = max(0, ideal_y1)
        src_x2 = min(image.shape[1], ideal_x2)
        src_y2 = min(image.shape[0], ideal_y2)

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return None

        # Build a padded canvas and place the cropped region inside it
        canvas = np.full(
            (square_size, square_size, image.shape[2]),
            self.pad_color,
            dtype=image.dtype,
        )
        dst_x1 = src_x1 - ideal_x1
        dst_y1 = src_y1 - ideal_y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        return canvas

    # ---- resize ---------------------------------------------------------

    def _resize(self, image: np.ndarray, edge: int) -> np.ndarray:
        methods = {
            "bilinear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }
        method = methods.get(self.interpolation, cv2.INTER_LINEAR)
        return cv2.resize(image, (edge, edge), interpolation=method)
