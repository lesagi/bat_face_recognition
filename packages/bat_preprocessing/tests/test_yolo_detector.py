"""Tests for YOLODetector — uses a mocked ultralytics model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from bat_preprocessing.yolo_detector import (
    DEFAULT_DETECTOR_CONFIG,
    DetectionPrediction,
    YOLODetector,
)


class _StubTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> _StubTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _StubBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = _StubTensor(xyxy)
        self.conf = _StubTensor(conf)
        self.cls = _StubTensor(cls)

    def __len__(self) -> int:
        return int(self.xyxy._arr.shape[0])


class _StubResult:
    def __init__(self, boxes: _StubBoxes | None) -> None:
        self.boxes = boxes


class _StubModel:
    def __init__(self, boxes: _StubBoxes | None) -> None:
        self._boxes = boxes
        self.names = {0: "bat"}
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[_StubResult]:
        self.calls.append(kwargs)
        return [_StubResult(self._boxes)]


def test_detector_returns_top_confidence_box() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)  # H=100, W=200
    boxes = _StubBoxes(
        xyxy=np.array([[10, 20, 50, 60], [40, 40, 160, 90]], dtype=np.float32),
        conf=np.array([0.4, 0.9], dtype=np.float32),
        cls=np.array([0, 0], dtype=np.float32),
    )
    stub = _StubModel(boxes)
    det = YOLODetector(model=stub)
    out = det.predict(image)
    assert isinstance(out, DetectionPrediction)
    # Picks the conf=0.9 box (index 1), normalised by W,H.
    assert out.confidence == pytest.approx(0.9)
    assert out.bounding_box == pytest.approx((40 / 200, 40 / 100, 160 / 200, 90 / 100))
    assert out.class_name == "bat"
    assert out.original_image_shape == (100, 200)  # (H, W)
    # pixel_box round-trips back to the source resolution.
    assert out.pixel_box(100, 200) == (40, 40, 160, 90)
    assert stub.calls and stub.calls[0]["source"] is image


def test_detector_returns_none_when_no_boxes() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert YOLODetector(model=_StubModel(None)).predict(image) is None
    empty = _StubBoxes(
        np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.float32)
    )
    assert YOLODetector(model=_StubModel(empty)).predict(image) is None


def test_detector_default_config() -> None:
    assert "weights" in DEFAULT_DETECTOR_CONFIG
    assert DEFAULT_DETECTOR_CONFIG["confidence_threshold"] == 0.3
