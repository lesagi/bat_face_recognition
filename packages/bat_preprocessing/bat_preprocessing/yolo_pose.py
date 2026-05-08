"""YOLO-based bat-face landmark / pose estimation.

Wraps an ``ultralytics.YOLO`` pose model. Mirrors :mod:`yolo_segmenter` —
weights path is provided via a config dictionary; nothing is hard-coded.

Default config:

    {"weights": "models/preprocessing/face_pose.pt",
     "confidence_threshold": 0.3,
     "device": None}
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .prediction_structures import PosePrediction

DEFAULT_POSE_CONFIG: dict[str, Any] = {
    "weights": "models/preprocessing/face_pose.pt",
    "confidence_threshold": 0.3,
    "device": None,
}


class YOLOPoseEstimator:
    """Wrapper around an ultralytics YOLO pose-estimation model."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        model: Any = None,
    ) -> None:
        """Construct a pose estimator.

        Args:
            config: Mapping with keys ``weights`` (path to a ``.pt`` file),
                ``confidence_threshold``, and optionally ``device``. Falls back
                to :data:`DEFAULT_POSE_CONFIG`.
            model: An already-instantiated YOLO model. Useful for tests; if
                provided, ``config['weights']`` is ignored.
        """
        cfg = {**DEFAULT_POSE_CONFIG, **(config or {})}
        self.weights: str = cfg["weights"]
        self.confidence_threshold: float = float(cfg["confidence_threshold"])
        self.device = cfg.get("device")

        if model is not None:
            self._model = model
        else:
            self._model = self._load_model(self.weights)

    @staticmethod
    def _load_model(weights: str) -> Any:
        try:
            from ultralytics import YOLO  # local import keeps deps optional
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for YOLOPoseEstimator; install with `pip install ultralytics`"
            ) from exc
        return YOLO(weights)

    @property
    def model(self) -> Any:
        return self._model

    def predict(
        self,
        image: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> PosePrediction | None:
        """Run pose estimation and return the first detection.

        Args:
            image: BGR ``np.ndarray`` of shape ``(H, W, 3)``.
            confidence_threshold: Override the configured threshold.

        Returns:
            A :class:`PosePrediction` for the first detection, or ``None``.
        """
        conf = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        kwargs: dict[str, Any] = {"source": image, "conf": conf, "verbose": False, "save": False}
        if self.device is not None:
            kwargs["device"] = self.device

        results = self._model.predict(**kwargs)
        if not results or len(results) == 0:
            return None

        result = results[0]
        keypoints_attr = getattr(result, "keypoints", None)
        if keypoints_attr is None:
            return None

        boxes_attr = getattr(result, "boxes", None)
        if boxes_attr is None or len(boxes_attr) == 0:
            return None

        kpts_data = self._to_numpy(keypoints_attr.data)
        if kpts_data is None or len(kpts_data) == 0:
            return None

        # First detection
        kpts3 = kpts_data[0]  # shape (K, 3)
        if kpts3.ndim != 2 or kpts3.shape[1] != 3:
            return None

        # Bounding box: take first detection's xyxy
        try:
            box_xyxy = self._to_numpy(boxes_attr.xyxy)[0]  # (4,)
        except Exception:
            box_xyxy = None

        orig_h, orig_w = image.shape[:2]
        if box_xyxy is not None:
            bbox = (
                float(box_xyxy[0]) / orig_w,
                float(box_xyxy[1]) / orig_h,
                float(box_xyxy[2]) / orig_w,
                float(box_xyxy[3]) / orig_h,
            )
        else:
            bbox = (0.0, 0.0, 1.0, 1.0)

        # Per-detection confidence: best-effort
        try:
            box_conf = self._to_numpy(boxes_attr.conf)[0]
            det_conf = float(box_conf)
        except Exception:
            det_conf = conf

        return PosePrediction(
            keypoints=kpts3.astype(np.float32),
            bounding_box=bbox,
            confidence=max(0.0, min(1.0, det_conf)),
            class_id=0,
            class_name="bat_face",
            original_image_shape=(orig_h, orig_w),
            model_resolution=(orig_h, orig_w),
        )

    @staticmethod
    def _to_numpy(data: Any) -> np.ndarray | None:
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data
        cpu = getattr(data, "cpu", None)
        if cpu is not None:
            data = cpu()
        numpy_fn = getattr(data, "numpy", None)
        if numpy_fn is not None:
            return numpy_fn()
        try:
            return np.asarray(data)
        except Exception:
            return None
