"""High-level preprocessing pipeline for bat-face recognition inputs.

Composes :class:`YOLOSegmenter`, :class:`YOLOPoseEstimator`, :class:`FaceAligner`,
and the background utilities into the same multi-step pipeline that
``app/siamese_preprocessing/input_processor.py`` provided — with all
TensorFlow tensor calls replaced by numpy / PIL equivalents.

Pipeline (default):

    1. (optional) face alignment via YOLO pose
    2. YOLO segmentation
    3. square crop around the mask
    4. (optional) background replacement
    5. resize to ``target_size``
    6. normalise to [0, 1] (or any custom ``scale_factor``)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from .background import (
    BACKGROUND_GENERATORS,
    BackgroundGenerator,
    get_background_generator,
    replace_background,
)
from .face_aligner import FaceAligner
from .prediction_cache import CacheConfig, CacheManager
from .prediction_structures import (
    PosePrediction,
    PredictionBundle,
    SegmentationPrediction,
)
from .yolo_pose import YOLOPoseEstimator
from .yolo_segmenter import YOLOSegmenter


@dataclass
class PreprocessingConfig:
    """Configuration for the high-level preprocessing pipeline."""

    target_size: int = 224
    """Edge length of the output square image, in pixels."""

    scale_factor: float = 255.0
    """Divisor used when normalising pixels to floats."""

    face_outer_margin_ratio: float = 0.5
    """Margin around the segmentation mask when cropping."""

    background_enabled: bool = False
    background_type: str = "blur"
    """One of ``BACKGROUND_GENERATORS`` keys."""

    align_face: bool = True
    """If True, run YOLO pose for rotational alignment."""

    cache_enabled: bool = True

    segmenter_config: Dict[str, Any] = field(default_factory=dict)
    pose_config: Dict[str, Any] = field(default_factory=dict)
    cache_config: Dict[str, Any] = field(default_factory=dict)


class PreprocessingPipeline:
    """Orchestrates segmentation, pose, alignment, and normalisation steps."""

    def __init__(
        self,
        config: Optional[Union[PreprocessingConfig, Dict[str, Any]]] = None,
        *,
        segmenter: Optional[YOLOSegmenter] = None,
        pose_estimator: Optional[YOLOPoseEstimator] = None,
        aligner: Optional[FaceAligner] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> None:
        if isinstance(config, dict):
            config = PreprocessingConfig(**config)
        self.config: PreprocessingConfig = config or PreprocessingConfig()

        self._segmenter = segmenter
        self._pose = pose_estimator
        self._aligner = aligner or FaceAligner(
            edge_length=self.config.target_size,
            margin_ratio=self.config.face_outer_margin_ratio,
        )

        if self.config.background_enabled:
            self.background_generator: Optional[Callable[..., np.ndarray]] = (
                get_background_generator(self.config.background_type)
            )
        else:
            self.background_generator = None

        if cache_manager is not None:
            self.cache_manager: Optional[CacheManager] = cache_manager
        elif self.config.cache_enabled:
            self.cache_manager = CacheManager(CacheConfig(**self.config.cache_config))
        else:
            self.cache_manager = None

    # ---- model lazy-instantiation --------------------------------------

    @property
    def segmenter(self) -> YOLOSegmenter:
        if self._segmenter is None:
            self._segmenter = YOLOSegmenter(self.config.segmenter_config)
        return self._segmenter

    @property
    def pose_estimator(self) -> YOLOPoseEstimator:
        if self._pose is None:
            self._pose = YOLOPoseEstimator(self.config.pose_config)
        return self._pose

    # ---- cache helpers -------------------------------------------------

    def _get_cached(self, key: Optional[str]) -> Optional[PredictionBundle]:
        if not key or self.cache_manager is None:
            return None
        try:
            return self.cache_manager.get_predictions(key, "both")
        except Exception:
            return None

    def _cache_predictions(
        self,
        key: Optional[str],
        seg: Optional[SegmentationPrediction],
        pose: Optional[PosePrediction],
    ) -> None:
        if not key or self.cache_manager is None:
            return
        if seg is None and pose is None:
            return
        bundle = PredictionBundle(
            segmentation=seg, pose=pose, image_path=key
        )
        try:
            self.cache_manager.cache_predictions(key, "both", bundle)
        except Exception:
            pass

    # ---- pipeline steps ------------------------------------------------

    def normalize_image(
        self,
        image: np.ndarray,
        scale: Optional[float] = None,
        mean_subtract: Optional[Union[float, List[float]]] = None,
        std_divide: Optional[Union[float, List[float]]] = None,
    ) -> np.ndarray:
        """Normalise a uint8 image with optional mean/std adjustment.

        Replaces the TF-tensor code in the original input_processor with a
        plain numpy implementation.
        """
        scale = float(scale if scale is not None else self.config.scale_factor)
        out = image.astype(np.float32) / scale
        if mean_subtract is not None:
            if isinstance(mean_subtract, (list, tuple)):
                for i, m in enumerate(mean_subtract):
                    out[..., i] -= float(m)
            else:
                out -= float(mean_subtract)
        if std_divide is not None:
            if isinstance(std_divide, (list, tuple)):
                for i, s in enumerate(std_divide):
                    out[..., i] /= float(s)
            else:
                out /= float(std_divide)
        return out

    def resize_image(
        self,
        image: np.ndarray,
        target_size: Optional[int] = None,
        interpolation: str = "bilinear",
    ) -> np.ndarray:
        """Resize ``image`` to a square of ``target_size`` using PIL.

        PIL handles uint8 RGB/BGR images consistently across platforms; this
        keeps the implementation framework-agnostic (no TF, no torchvision).
        """
        target = int(target_size if target_size is not None else self.config.target_size)
        modes = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "cubic": Image.BICUBIC,
            "area": Image.LANCZOS,
        }
        pil_mode = modes.get(interpolation, Image.BILINEAR)
        # PIL expects RGB ordering; the rest of the pipeline is BGR. Swap
        # before/after to avoid silent colour distortion.
        if image.ndim == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            resized = pil_img.resize((target, target), pil_mode)
            arr = np.array(resized)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        pil_img = Image.fromarray(image)
        resized = pil_img.resize((target, target), pil_mode)
        return np.array(resized)

    def preprocess_single_image(
        self,
        image: np.ndarray,
        cache_key: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Run the full preprocessing pipeline on a single image.

        Returns the normalised float32 image of shape
        ``(target_size, target_size, 3)`` with values in ``[0, 1]``, or
        ``None`` if any required step fails.
        """
        if image is None or image.ndim != 3:
            return None

        cached = self._get_cached(cache_key)
        cached_pose = cached.pose if cached else None
        cached_seg = cached.segmentation if cached else None

        # Step 1: pose-based alignment (optional)
        pose_pred = cached_pose
        if self.config.align_face and pose_pred is None:
            try:
                pose_pred = self.pose_estimator.predict(image)
            except Exception:
                pose_pred = None

        # Step 2: segmentation
        seg_pred = cached_seg
        if seg_pred is None:
            try:
                seg_pred = self.segmenter.predict(image)
            except Exception:
                seg_pred = None
        if seg_pred is None:
            return None

        # Cache the freshly computed predictions
        self._cache_predictions(cache_key, seg_pred, pose_pred)

        # Step 3+4: align, crop and (optionally) replace background
        aligned = self._aligner.align(image, seg_pred, pose_pred)
        if aligned is None:
            return None

        if self.config.background_enabled and self.background_generator is not None:
            # We need a mask aligned to the cropped output. The aligner has
            # already operated on the source mask; recompute a mask for the
            # aligned image using the same segmenter for consistency.
            try:
                aligned_seg = self.segmenter.predict(aligned)
            except Exception:
                aligned_seg = None
            if aligned_seg is not None:
                aligned = replace_background(
                    aligned, aligned_seg.mask, self.background_generator
                )

        # Step 5: resize to target size (the aligner already produced a square
        # at target_size, but we keep this step for parity).
        resized = self.resize_image(aligned, self.config.target_size, "bilinear")

        # Step 6: normalise
        return self.normalize_image(resized)

    def preprocess_single_image_from_path(
        self, image_path: str
    ) -> Optional[np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            return None
        return self.preprocess_single_image(image, cache_key=image_path)


__all__ = [
    "BACKGROUND_GENERATORS",
    "BackgroundGenerator",
    "PreprocessingConfig",
    "PreprocessingPipeline",
]
