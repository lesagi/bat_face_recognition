"""Tests for the face aligner."""

from __future__ import annotations

import numpy as np

from bat_preprocessing.face_aligner import FaceAligner
from bat_preprocessing.prediction_structures import (
    PosePrediction,
    SegmentationPrediction,
)


def _synthetic_image(size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # white square in centre
    img[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
    return img


def _synthetic_seg(size: int = 64) -> SegmentationPrediction:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1
    return SegmentationPrediction(
        mask=mask,
        confidence=0.9,
        bounding_box=(0.25, 0.25, 0.75, 0.75),
        original_image_shape=(size, size),
    )


def _synthetic_pose(size: int = 64) -> PosePrediction:
    cx = size // 2
    cy = size // 2
    kpts = np.array(
        [
            [cx - 8, cy - 4, 0.95],
            [cx + 8, cy - 4, 0.95],
            [cx, cy + 4, 0.85],
        ],
        dtype=np.float32,
    )
    return PosePrediction(
        keypoints=kpts,
        bounding_box=(0.25, 0.25, 0.75, 0.75),
        confidence=0.9,
        original_image_shape=(size, size),
    )


def test_face_aligner_output_shape_with_pose() -> None:
    aligner = FaceAligner(edge_length=128, margin_ratio=0.25)
    img = _synthetic_image(64)
    seg = _synthetic_seg(64)
    pose = _synthetic_pose(64)
    out = aligner.align(img, seg, pose)
    assert out is not None
    assert out.shape == (128, 128, 3)
    assert out.dtype == img.dtype


def test_face_aligner_output_shape_without_pose() -> None:
    aligner = FaceAligner(edge_length=64, margin_ratio=0.0)
    img = _synthetic_image(48)
    seg = _synthetic_seg(48)
    out = aligner.align(img, seg, None)
    assert out is not None
    assert out.shape == (64, 64, 3)


def test_face_aligner_handles_empty_mask() -> None:
    aligner = FaceAligner(edge_length=32)
    img = _synthetic_image(32)
    empty_mask = np.zeros((32, 32), dtype=np.uint8)
    # pre-bake the prediction with at least one pixel so __post_init__ accepts it,
    # then zero it out before passing it into align()
    seg = SegmentationPrediction(
        mask=empty_mask.copy(),
        confidence=0.5,
        bounding_box=(0.0, 0.0, 1.0, 1.0),
        original_image_shape=(32, 32),
    )
    out = aligner.align(img, seg, None)
    assert out is None


def test_face_aligner_validates_constructor() -> None:
    try:
        FaceAligner(edge_length=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for edge_length=0")

    try:
        FaceAligner(margin_ratio=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative margin_ratio")
