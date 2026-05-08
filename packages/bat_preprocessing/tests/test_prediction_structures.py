"""Tests for prediction dataclasses."""

from __future__ import annotations

import numpy as np

from bat_preprocessing.prediction_structures import (
    PosePrediction,
    PredictionBundle,
    SegmentationPrediction,
)


def _make_seg(area: int = 25) -> SegmentationPrediction:
    mask = np.zeros((10, 10), dtype=np.uint8)
    side = int(np.sqrt(area))
    mask[:side, :side] = 1
    return SegmentationPrediction(
        mask=mask,
        confidence=0.9,
        bounding_box=(0.0, 0.0, side / 10.0, side / 10.0),
        original_image_shape=(10, 10),
    )


def _make_pose() -> PosePrediction:
    kpts = np.array(
        [[1.0, 1.0, 0.9], [9.0, 1.0, 0.85], [5.0, 9.0, 0.8]], dtype=np.float32
    )
    return PosePrediction(
        keypoints=kpts,
        bounding_box=(0.1, 0.1, 0.9, 0.9),
        confidence=0.85,
        original_image_shape=(10, 10),
    )


def test_segmentation_prediction_roundtrip() -> None:
    pred = _make_seg(area=16)
    data = pred.to_dict()
    restored = SegmentationPrediction.from_dict(data)
    assert restored.mask.shape == pred.mask.shape
    assert restored.confidence == pred.confidence
    assert restored.bounding_box == pred.bounding_box
    assert pred.mask_area == 16
    cx, cy = pred.mask_center
    assert 0 <= cx < 10 and 0 <= cy < 10


def test_segmentation_prediction_validates_inputs() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    try:
        SegmentationPrediction(mask=mask, confidence=2.0, bounding_box=(0, 0, 1, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad confidence")


def test_pose_prediction_basics() -> None:
    pose = _make_pose()
    assert pose.num_keypoints == 3
    assert pose.valid_keypoints.shape == (3, 3)
    centers = pose.keypoint_centers
    assert len(centers) == 3
    serialised = pose.to_dict()
    restored = PosePrediction.from_dict(serialised)
    assert restored.keypoints.shape == pose.keypoints.shape
    np.testing.assert_allclose(restored.keypoints, pose.keypoints)


def test_prediction_bundle_roundtrip() -> None:
    bundle = PredictionBundle(
        segmentation=_make_seg(),
        pose=_make_pose(),
        image_path="/tmp/foo.jpg",
    )
    bundle.add_processing_step("test")
    bundle.add_model_version("seg", "1.0")
    assert bundle.has_predictions
    assert bundle.prediction_count == 2
    data = bundle.to_dict()
    restored = PredictionBundle.from_dict(data)
    assert restored.image_path == bundle.image_path
    assert restored.prediction_count == 2
    assert restored.processing_steps == bundle.processing_steps
