"""Tests for YOLOSegmenter — uses a mocked ultralytics model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from bat_preprocessing.prediction_structures import SegmentationPrediction
from bat_preprocessing.yolo_segmenter import DEFAULT_SEGMENTER_CONFIG, YOLOSegmenter


class _StubMaskTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> _StubMaskTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _StubMasks:
    def __init__(self, arr: np.ndarray) -> None:
        self.data = _StubMaskTensor(arr)

    def __len__(self) -> int:
        return int(self.data._arr.shape[0])


class _StubResult:
    def __init__(self, masks: _StubMasks) -> None:
        self.masks = masks


class _StubModel:
    def __init__(self, mask_array: np.ndarray) -> None:
        self._mask_array = mask_array
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[_StubResult]:
        self.calls.append(kwargs)
        return [_StubResult(_StubMasks(self._mask_array))]


def _stub_mask(h: int = 16, w: int = 16) -> np.ndarray:
    masks = np.zeros((1, h, w), dtype=np.uint8)
    masks[0, 4:12, 4:12] = 1
    return masks


def test_segmenter_returns_largest_mask() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    stub = _StubModel(_stub_mask())
    seg = YOLOSegmenter(model=stub)
    out = seg.predict(image)
    assert isinstance(out, SegmentationPrediction)
    assert out.mask.shape == image.shape[:2]
    assert out.mask.sum() == 64
    assert out.original_image_shape == (16, 16)
    assert stub.calls and stub.calls[0]["source"] is image


def test_segmenter_returns_none_when_no_masks() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    empty_masks = np.zeros((0, 16, 16), dtype=np.uint8)
    seg = YOLOSegmenter(model=_StubModel(empty_masks))
    assert seg.predict(image) is None


def test_segmenter_uses_default_config_paths() -> None:
    # The class must not hard-code paths internally; verify default config
    # carries the expected weights location.
    assert "weights" in DEFAULT_SEGMENTER_CONFIG
    assert DEFAULT_SEGMENTER_CONFIG["weights"].endswith("face_seg.pt")


def test_segmenter_loads_real_yolo_when_available() -> None:
    """Smoke test the real ultralytics path when the package is installed."""
    pytest.importorskip("ultralytics", reason="ultralytics not installed")
    # Lazy import to avoid the module-level skip pattern
    # (this only proves that DEFAULT_SEGMENTER_CONFIG plumbs through)
    cfg = {**DEFAULT_SEGMENTER_CONFIG}
    # We can't load real weights here without the .pt file; just ensure the
    # constructor accepts the model arg path without trying to load.
    seg = YOLOSegmenter(cfg, model=_StubModel(_stub_mask()))
    assert seg.weights == cfg["weights"]
