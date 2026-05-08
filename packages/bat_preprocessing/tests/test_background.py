"""Tests for the background generation + replacement utilities."""

from __future__ import annotations

import numpy as np
from bat_preprocessing.background import (
    BackgroundGenerator,
    create_blur_image,
    get_background_generator,
    replace_background,
    replace_background_with_alpha,
)


def test_solid_color_generator_shape_and_color() -> None:
    bg = BackgroundGenerator.solid_color(8, 12, color=(0, 255, 0))
    assert bg.shape == (8, 12, 3)
    assert (bg[..., 1] == 255).all()


def test_blur_generator_shape() -> None:
    bg = create_blur_image(16, 16)
    assert bg.shape == (16, 16, 3)
    assert bg.dtype == np.uint8


def test_get_background_generator_lookup() -> None:
    gen = get_background_generator("noise")
    assert gen is BackgroundGenerator.noise
    try:
        get_background_generator("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown generator")


def test_replace_background_uses_mask() -> None:
    image = np.full((10, 10, 3), 50, dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 1
    bg = np.full((10, 10, 3), 200, dtype=np.uint8)
    out = replace_background(image, mask, bg)
    # foreground square retained
    assert (out[3:7, 3:7] == 50).all()
    # background outside the mask
    assert (out[0, 0] == 200).all()


def test_replace_background_resizes_mask_to_image() -> None:
    image = np.full((20, 20, 3), 0, dtype=np.uint8)
    small_mask = np.zeros((10, 10), dtype=np.uint8)
    small_mask[3:7, 3:7] = 1
    bg = np.full((20, 20, 3), 250, dtype=np.uint8)
    out = replace_background(image, small_mask, bg)
    # Output shape must match image
    assert out.shape == image.shape


def test_replace_background_with_alpha_preserves_alpha() -> None:
    rgb = np.full((10, 10, 3), 60, dtype=np.uint8)
    alpha = np.full((10, 10, 1), 200, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=2)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    bg = np.full((10, 10, 3), 10, dtype=np.uint8)
    out = replace_background_with_alpha(rgba, mask, bg)
    assert out.shape == (10, 10, 4)
    # Alpha channel passed through unchanged
    np.testing.assert_array_equal(out[..., 3:], alpha)


def test_replace_background_accepts_callable() -> None:
    image = np.full((6, 6, 3), 90, dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:5, 1:5] = 1

    def gen(h: int, w: int) -> np.ndarray:
        return np.full((h, w, 3), 5, dtype=np.uint8)

    out = replace_background(image, mask, gen)
    assert out.shape == image.shape
    # outside mask is filled by generator
    assert int(out[0, 0, 0]) == 5
    # inside mask is foreground
    assert int(out[2, 2, 0]) == 90
