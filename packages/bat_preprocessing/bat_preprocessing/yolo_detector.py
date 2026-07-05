"""YOLO-based bat-face detection (bounding box only).

Wraps an ``ultralytics.YOLO`` *detect* model. Mirrors :mod:`yolo_segmenter` —
weights path comes from a config dictionary; nothing is hard-coded.

Used to localise the face on a full frame and crop a square chip around it
(the mauritius segmentation model was trained on square images and only works
on such crops). The segmenter/pose wrappers can't stand in here: a detect model
exposes no masks, so :class:`YOLOSegmenter` would return ``None``.

Default config::

    {"weights": "models/preprocessing/face_detect.pt",
     "confidence_threshold": 0.3,
     "device": None}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_DETECTOR_CONFIG: dict[str, Any] = {
    "weights": "models/preprocessing/face_detect.pt",
    "confidence_threshold": 0.3,
    "device": None,
}


@dataclass
class DetectionPrediction:
    """A single YOLO detection (the top-confidence box)."""

    bounding_box: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalised [0, 1]
    confidence: float
    class_id: int | None = None
    class_name: str | None = None
    original_image_shape: tuple[int, int] | None = None  # (h, w)

    def pixel_box(self, height: int, width: int) -> tuple[int, int, int, int]:
        """Return the box in pixel coords ``(x1, y1, x2, y2)`` for an H×W image."""
        x1, y1, x2, y2 = self.bounding_box
        return (
            int(round(x1 * width)),
            int(round(y1 * height)),
            int(round(x2 * width)),
            int(round(y2 * height)),
        )


class YOLODetector:
    """Wrapper around an ultralytics YOLO detection model."""

    def __init__(self, config: dict[str, Any] | None = None, *, model: Any = None) -> None:
        """Construct a detector.

        Args:
            config: Mapping with keys ``weights`` (path to a ``.pt`` file),
                ``confidence_threshold``, and optionally ``device``. Falls back
                to :data:`DEFAULT_DETECTOR_CONFIG`.
            model: An already-instantiated YOLO model (tests); if provided,
                ``config['weights']`` is ignored.
        """
        cfg = {**DEFAULT_DETECTOR_CONFIG, **(config or {})}
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
        except ImportError as exc:  # pragma: no cover - env without ultralytics
            raise ImportError(
                "ultralytics is required for YOLODetector; install with `pip install ultralytics`"
            ) from exc
        return YOLO(weights)

    @property
    def model(self) -> Any:
        return self._model

    def predict(
        self,
        image: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> DetectionPrediction | None:
        """Run detection and return the highest-confidence box.

        Args:
            image: BGR ``np.ndarray`` of shape ``(H, W, 3)``.
            confidence_threshold: Override the configured threshold.

        Returns:
            A :class:`DetectionPrediction` for the most confident detection, or
            ``None`` if nothing is detected.
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
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = self._to_numpy(boxes.xyxy)
        confs = self._to_numpy(boxes.conf)
        if xyxy is None or confs is None or len(xyxy) == 0:
            return None

        idx = int(np.argmax(confs))
        box = xyxy[idx]
        det_conf = float(confs[idx])

        orig_h, orig_w = image.shape[:2]
        bbox = (
            float(box[0]) / orig_w,
            float(box[1]) / orig_h,
            float(box[2]) / orig_w,
            float(box[3]) / orig_h,
        )

        class_id = None
        try:
            cls = self._to_numpy(boxes.cls)
            class_id = int(cls[idx]) if cls is not None else None
        except Exception:
            class_id = None
        class_name = None
        names = getattr(self._model, "names", None)
        if names is not None and class_id is not None:
            try:
                class_name = names[class_id]
            except Exception:
                class_name = None

        return DetectionPrediction(
            bounding_box=bbox,
            confidence=max(0.0, min(1.0, det_conf)),
            class_id=class_id,
            class_name=class_name,
            original_image_shape=(orig_h, orig_w),
        )

    @staticmethod
    def _to_numpy(data: Any) -> np.ndarray | None:
        """Coerce a YOLO tensor (torch.Tensor or np.ndarray) to numpy."""
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
