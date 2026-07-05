"""Face alignment using YOLO segmentation masks and pose landmarks.

Given an input image plus a :class:`SegmentationPrediction` and (optionally)
a :class:`PosePrediction`, produce a square aligned crop of configurable
edge length. Two alignment modes are supported:

``"mask_crop"`` (default, back-compatible)
    1. **Rotate** image so the line between leftmost/rightmost pose keypoints
       becomes horizontal (drop step if no pose).
    2. **Crop** a square region centred on the segmentation *mask*, with
       ``margin_ratio`` padding around the mask's bounding box.
    3. **Resize** to ``edge_length`` × ``edge_length``.

``"eye_anchored"``
    Rotate so the **eye line** (keypoints ``eye_indices``) is horizontal — i.e.
    *straighten* the face — then crop a square **centred on the segmentation
    mask** (the head), sized from the mask bbox plus a ``margin_ratio`` margin.
    The eyes are used **only** to straighten; the crop is **not** eye-centred
    (eye-centred framing gave poor results). The nose only drives the 180°
    orientation guard. Returns the aligned crop plus the keypoints and mask
    warped into crop space, so callers can build background / ellipse variants
    without re-running models.

This is a numpy / OpenCV-only implementation — no TensorFlow.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from .prediction_structures import PosePrediction, SegmentationPrediction


class AlignedFace(NamedTuple):
    """Result of :meth:`FaceAligner.align_eye_anchored`."""

    image: np.ndarray  # (edge, edge, 3) BGR uint8 aligned crop
    keypoints: np.ndarray  # (K, 3) pose keypoints in crop pixel coords (x, y, conf)
    mask: np.ndarray  # (edge, edge) uint8 0/1 — segmentation mask warped into crop space


class FaceAligner:
    """Align and square-crop a bat face."""

    def __init__(
        self,
        edge_length: int = 224,
        margin_ratio: float = 0.5,
        interpolation: str = "bilinear",
        pad_color: tuple[int, int, int] = (128, 128, 128),
        mode: str = "mask_crop",
        eye_indices: tuple[int, int] = (0, 1),
        nose_index: int = 2,
        min_keypoint_confidence: float = 0.0,
        require_confident_eyes: bool = False,
        nose_roll_weight: float = 0.0,
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
            mode: ``"mask_crop"`` (default) or ``"eye_anchored"``.
            eye_indices: Indices of the left/right eye keypoints in the pose
                model output (this project's pose model: 0=left_eye, 1=right_eye).
            nose_index: Index of the nose keypoint (used only for the 180°
                orientation guard, not for positioning).
            min_keypoint_confidence: Eye keypoints below this confidence are
                treated as undetected (avoids rotating by a garbage angle when
                the pose model barely sees the eyes).
            require_confident_eyes: In ``eye_anchored`` mode, return ``None``
                (skip the frame) when both eyes are not confidently detected,
                instead of falling back to a mask-centroid crop.
            nose_roll_weight: Blend factor (0..1) for the roll implied by the
                eye-midpoint→nose axis. The eye line can be horizontal while
                the muzzle still leans sideways; 0 keeps eye-line-only
                rotation, 1 rotates the nose directly below the eye midpoint,
                0.5 splits the difference.
        """
        if edge_length <= 0:
            raise ValueError(f"edge_length must be positive, got {edge_length}")
        if margin_ratio < 0:
            raise ValueError(f"margin_ratio must be >= 0, got {margin_ratio}")
        if mode not in ("mask_crop", "eye_anchored"):
            raise ValueError(f"mode must be 'mask_crop' or 'eye_anchored', got {mode!r}")
        self.edge_length = int(edge_length)
        self.margin_ratio = float(margin_ratio)
        self.interpolation = interpolation
        self.pad_color = pad_color
        self.mode = mode
        self.eye_indices = (int(eye_indices[0]), int(eye_indices[1]))
        self.nose_index = int(nose_index)
        self.min_keypoint_confidence = float(min_keypoint_confidence)
        self.require_confident_eyes = bool(require_confident_eyes)
        self.nose_roll_weight = float(nose_roll_weight)

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

        if self.mode == "eye_anchored":
            result = self.align_eye_anchored(image, segmentation, pose)
            return None if result is None else result.image

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
        method = self._cv_interpolation()
        return cv2.resize(image, (edge, edge), interpolation=method)

    def _cv_interpolation(self) -> int:
        methods = {
            "bilinear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }
        return methods.get(self.interpolation, cv2.INTER_LINEAR)

    # ---- eye-anchored alignment ----------------------------------------

    def _eye_rotation_params(
        self, pose: PosePrediction | None
    ) -> tuple[tuple[float, float], float] | None:
        """(center, angle_degrees) from the two eye keypoints, or None.

        Uses the explicit ``eye_indices`` (left/right eye) rather than the
        leftmost/rightmost valid keypoint, so the nose can never be mistaken
        for an eye on a tilted head.
        """
        if pose is None:
            return None
        kpts = pose.keypoints
        i0, i1 = self.eye_indices
        if kpts is None or kpts.ndim != 2 or kpts.shape[0] <= max(i0, i1):
            return None
        left_kpt, right_kpt = kpts[i0], kpts[i1]
        # Require *both* eyes to be confidently detected. Below threshold the
        # keypoint position is unreliable and rotating by its angle tilts the
        # whole face (the 90°-ish failures seen on near-zero-confidence eyes).
        thr = self.min_keypoint_confidence
        if float(left_kpt[2]) < thr or float(right_kpt[2]) < thr:
            return None
        left = left_kpt[:2].astype(float)
        right = right_kpt[:2].astype(float)
        if np.any(np.isnan(left)) or np.any(np.isnan(right)) or np.allclose(left, right):
            return None
        dy = float(right[1] - left[1])
        dx = float(right[0] - left[0])
        angle = float(np.degrees(np.arctan2(dy, dx)))
        center = (float((left[0] + right[0]) / 2.0), float((left[1] + right[1]) / 2.0))

        # Orientation guard: if the nose is confidently detected and ends up
        # *above* the eye line after rotation, the face is upside down — add
        # 180° so the nose points down (a bat face's snout is below the eyes).
        nose = self._confident_nose(pose)
        if nose is not None:
            rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            nose_y = float(rot[1, 0] * nose[0] + rot[1, 1] * nose[1] + rot[1, 2])
            if nose_y < center[1]:
                angle = (angle + 180.0) % 360.0
            if self.nose_roll_weight > 0.0:
                # Blend in the roll implied by the eye-midpoint→nose axis: the
                # eye line can be horizontal while the muzzle still leans, so
                # pull the nose toward "directly below the eye midpoint".
                rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                nx = float(rot[0, 0] * nose[0] + rot[0, 1] * nose[1] + rot[0, 2])
                ny = float(rot[1, 0] * nose[0] + rot[1, 1] * nose[1] + rot[1, 2])
                vx, vy = nx - center[0], ny - center[1]
                if vy > 1e-6:  # guard already put the nose below the eyes
                    residual = -float(np.degrees(np.arctan2(vx, vy)))
                    angle += self.nose_roll_weight * residual
        return center, angle

    def _confident_nose(self, pose: PosePrediction | None) -> tuple[float, float] | None:
        """Nose keypoint xy if confidently detected, else ``None``."""
        if pose is None:
            return None
        kpts = pose.keypoints
        j = self.nose_index
        if kpts is None or kpts.ndim != 2 or kpts.shape[0] <= j:
            return None
        if float(kpts[j][2]) < self.min_keypoint_confidence:
            return None
        x, y = float(kpts[j][0]), float(kpts[j][1])
        if np.isnan(x) or np.isnan(y):
            return None
        return x, y

    @staticmethod
    def _transform_keypoints(kpts: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply a 2x3 affine ``matrix`` to keypoint xy, preserving confidence."""
        if kpts is None or kpts.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        out = kpts.astype(np.float32).copy()
        xy = kpts[:, :2].astype(np.float64)
        homog = np.hstack([xy, np.ones((len(xy), 1))])  # (K, 3)
        transformed = (matrix @ homog.T).T  # (K, 2)
        out[:, 0] = transformed[:, 0]
        out[:, 1] = transformed[:, 1]
        return out

    def align_eye_anchored(
        self,
        image: np.ndarray,
        segmentation: SegmentationPrediction,
        pose: PosePrediction | None = None,
    ) -> AlignedFace | None:
        """Eye-centred alignment (see module docstring).

        Returns an :class:`AlignedFace` (crop + keypoints + mask in crop space)
        or ``None`` if the mask is empty / degenerate.
        """
        if image is None or image.ndim != 3:
            return None

        h, w = image.shape[:2]
        mask = segmentation.mask
        if mask is None:
            return None
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = mask if mask.max() <= 1 else (mask > 127).astype(np.uint8)
        binary = (binary > 0).astype(np.uint8)

        # Rotation: eye line → horizontal. Without confidently-detected eyes we
        # either skip the frame (require_confident_eyes) or fall back to a
        # mask-centroid crop with no rotation.
        rot = self._eye_rotation_params(pose)
        if rot is not None:
            center, angle = rot
        elif self.require_confident_eyes:
            return None
        else:
            ys0, xs0 = np.where(binary > 0)
            if xs0.size == 0:
                return None
            center = (float(xs0.mean()), float(ys0.mean()))
            angle = 0.0

        rotation = cv2.getRotationMatrix2D(center, angle, 1.0)  # 2x3

        # Mask bbox in the rotated frame sizes the square.
        mask_rot = cv2.warpAffine(
            binary,
            rotation,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        ys, xs = np.where(mask_rot > 0)
        if xs.size == 0:
            return None
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())
        mask_w = max_x - min_x + 1.0
        mask_h = max_y - min_y + 1.0
        max_dim = max(mask_w, mask_h)

        # Center the crop on the segmentation mask (the head), NOT the eyes.
        # The eyes are used only to straighten the image (the rotation above);
        # eye-centred framing gave poor results in practice.
        mask_cx = (min_x + max_x) / 2.0
        mask_cy = (min_y + max_y) / 2.0
        margin = self.margin_ratio
        side = max_dim * (1.0 + 2.0 * margin)
        if side <= 1.0:
            return None

        edge = self.edge_length
        scale = edge / side
        top_left_x = mask_cx - side / 2.0
        top_left_y = mask_cy - side / 2.0

        # Compose rotate → translate-to-crop → scale into one affine.
        matrix = rotation.astype(np.float64) * scale
        matrix[0, 2] -= scale * top_left_x
        matrix[1, 2] -= scale * top_left_y

        aligned = cv2.warpAffine(
            image,
            matrix,
            (edge, edge),
            flags=self._cv_interpolation(),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.pad_color,
        )
        mask_in_crop = cv2.warpAffine(
            binary,
            matrix,
            (edge, edge),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask_in_crop = (mask_in_crop > 0).astype(np.uint8)

        kpts = pose.keypoints if pose is not None else np.zeros((0, 3), dtype=np.float32)
        kpts_in_crop = self._transform_keypoints(kpts, matrix)

        return AlignedFace(image=aligned, keypoints=kpts_in_crop, mask=mask_in_crop)

    def elliptical_face_mask(
        self,
        mask: np.ndarray | None,
        keypoints: np.ndarray | None,
        edge: int,
        expand: float = 1.05,
    ) -> np.ndarray:
        """Build a circular face mask from a segmentation ``mask``.

        Always a perfect circle (never a rotated/eccentric ellipse), so the
        variant's cut shape is identical across images: centred on the fitted
        head, base radius the geometric mean of the fitted semi-axes, then
        grown just enough that every valid landmark in ``keypoints`` lies
        inside, plus a small ``expand`` margin.

        Returns a ``(edge, edge)`` uint8 0/1 mask (1 = inside the circle).
        """
        out = np.zeros((edge, edge), dtype=np.uint8)
        if mask is None:
            return out
        binary = mask if mask.max() <= 1 else (mask > 127).astype(np.uint8)
        binary = (binary > 0).astype(np.uint8)

        cx = cy = ma = mb = None
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if len(largest) >= 5 and cv2.contourArea(largest) > 0:
                (cx, cy), (ma, mb), _ = cv2.fitEllipse(largest)

        if cx is None:
            # Fallback: circle from the mask bbox, else from the landmark bbox.
            ys, xs = np.where(binary > 0)
            if xs.size == 0:
                if keypoints is None or keypoints.size == 0:
                    return out
                valid = keypoints[keypoints[:, 2] > 0]
                if len(valid) == 0:
                    return out
                xs, ys = valid[:, 0], valid[:, 1]
            cx = float((xs.min() + xs.max()) / 2.0)
            cy = float((ys.min() + ys.max()) / 2.0)
            ma = float(xs.max() - xs.min() + 1)
            mb = float(ys.max() - ys.min() + 1)

        # Perfect circle: geometric-mean radius keeps the area comparable to
        # the fitted head while making the cut shape identical across images.
        r = max((float(ma) * float(mb)) ** 0.5 / 2.0, 1e-3)
        max_d = r
        if keypoints is not None and keypoints.size:
            for x, y, conf in keypoints:
                if conf <= 0.0:
                    continue
                max_d = max(max_d, float(np.hypot(float(x) - cx, float(y) - cy)))
        r = max_d * expand

        cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), 1, thickness=-1)
        return out
