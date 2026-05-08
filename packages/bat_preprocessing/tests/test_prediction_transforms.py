"""Tests for prediction transformation helpers."""

from __future__ import annotations

import numpy as np

from bat_preprocessing.prediction_transforms import (
    BoundingBoxTransformer,
    CoordinateMapper,
    KeypointTransformer,
    MaskTransformer,
)


def test_resize_mask_preserves_binary() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    out = MaskTransformer.resize_mask(mask, target_size=(10, 10))
    assert out.shape == (10, 10)
    assert set(np.unique(out)).issubset({0, 1})


def test_crop_mask_shape() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    cropped = MaskTransformer.crop_mask(mask, (5, 5, 15, 15))
    assert cropped.shape == (10, 10)
    assert cropped.sum() == 100


def test_keypoint_resize_scales_coords() -> None:
    kpts = np.array(
        [[10.0, 10.0, 1.0], [20.0, 20.0, 0.5]], dtype=np.float32
    )
    out = KeypointTransformer.transform_keypoints_for_resize(
        kpts, original_size=(40, 40), target_size=(20, 20)
    )
    np.testing.assert_allclose(out[:, :2], [[5.0, 5.0], [10.0, 10.0]])
    np.testing.assert_allclose(out[:, 2], kpts[:, 2])


def test_keypoint_crop_filters_out_of_bounds() -> None:
    kpts = np.array(
        [[5.0, 5.0, 1.0], [15.0, 15.0, 1.0]], dtype=np.float32
    )
    out = KeypointTransformer.transform_keypoints_for_crop(kpts, (10, 10, 30, 30))
    # First kpt at (5,5) → (-5,-5) which is filtered out;
    # second at (15,15) → (5,5) which is inside the 20x20 crop window.
    assert out.shape[0] == 1
    np.testing.assert_allclose(out[0, :2], [5.0, 5.0])


def test_bbox_transform_for_crop_clamps() -> None:
    out = BoundingBoxTransformer.transform_bbox_for_crop(
        bbox=(5.0, 5.0, 25.0, 25.0), crop_region=(10, 10, 20, 20)
    )
    assert out == (0.0, 0.0, 10.0, 10.0)


def test_coordinate_mapper_resize_then_apply() -> None:
    tm = CoordinateMapper.create_resize_mapping((100, 100), (50, 50))
    pt = tm.transform_point((40.0, 80.0))
    np.testing.assert_allclose(pt, (20.0, 40.0))
