"""YOLO-based bat-face segmentation.

Wraps an ``ultralytics.YOLO`` model so the rest of the pipeline can call a
plain Python interface and stay decoupled from the framework. The weights
path is supplied via a config dictionary; nothing is hard-coded at the class
level.

Default config:

    {"weights": "models/preprocessing/face_seg.pt",
     "confidence_threshold": 0.3,
     "device": None}
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .prediction_structures import SegmentationPrediction


DEFAULT_SEGMENTER_CONFIG: Dict[str, Any] = {
    "weights": "models/preprocessing/face_seg.pt",
    "confidence_threshold": 0.3,
    "device": None,
}


class YOLOSegmenter:
    """Wrapper around an ultralytics YOLO segmentation model."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        model: Any = None,
    ) -> None:
        """Construct a segmenter.

        Args:
            config: Mapping with keys ``weights`` (path to a ``.pt`` file),
                ``confidence_threshold``, and optionally ``device``. Falls back
                to :data:`DEFAULT_SEGMENTER_CONFIG`.
            model: An already-instantiated YOLO model. Useful for tests; if
                provided, ``config['weights']`` is ignored.
        """
        cfg = {**DEFAULT_SEGMENTER_CONFIG, **(config or {})}
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
        except ImportError as exc:  # pragma: no cover - exercised in env without ultralytics
            raise ImportError(
                "ultralytics is required for YOLOSegmenter; install with `pip install ultralytics`"
            ) from exc
        return YOLO(weights)

    @property
    def model(self) -> Any:
        return self._model

    def predict(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> Optional[SegmentationPrediction]:
        """Run segmentation and return the highest-area mask.

        Args:
            image: BGR ``np.ndarray`` of shape ``(H, W, 3)``.
            confidence_threshold: Override the configured threshold.

        Returns:
            A :class:`SegmentationPrediction` for the largest detected mask,
            or ``None`` if no mask is found.
        """
        conf = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        kwargs: Dict[str, Any] = {"source": image, "conf": conf, "verbose": False, "save": False}
        if self.device is not None:
            kwargs["device"] = self.device

        results = self._model.predict(**kwargs)
        if not results or len(results) == 0:
            return None

        result = results[0]
        masks = getattr(result, "masks", None)
        if masks is None or len(masks) == 0:
            return None

        # Some ultralytics builds expose masks.data as torch tensor; convert via cpu().numpy()
        masks_np = self._masks_to_numpy(masks.data)
        if masks_np is None or len(masks_np) == 0:
            return None

        areas = [int(np.sum(m)) for m in masks_np]
        idx = int(np.argmax(areas))
        mask = masks_np[idx].astype(np.uint8)

        orig_h, orig_w = image.shape[:2]
        if mask.shape[0] != orig_h or mask.shape[1] != orig_w:
            import cv2

            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)

        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            return None
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())

        bbox = (
            min_x / orig_w,
            min_y / orig_h,
            max_x / orig_w,
            max_y / orig_h,
        )

        return SegmentationPrediction(
            mask=mask,
            confidence=conf,
            bounding_box=bbox,
            class_id=0,
            class_name="bat_face",
            original_image_shape=(orig_h, orig_w),
            model_resolution=(int(mask.shape[0]), int(mask.shape[1])),
        )

    @staticmethod
    def _masks_to_numpy(data: Any) -> Optional[np.ndarray]:
        """Coerce a YOLO masks tensor (torch.Tensor or np.ndarray) to numpy."""
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
