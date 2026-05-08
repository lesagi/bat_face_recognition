"""Tests for the 5-identities-per-page ``make_composite`` layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from bat_core.types import SaliencyImage  # noqa: E402
from bat_interpretability import make_composite  # noqa: E402
from bat_interpretability.composite import composite_from_saliency_images  # noqa: E402


def _write_fake_image(path: Path, edge: int = 32) -> Path:
    img = (np.random.rand(edge, edge, 3) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    return path


def test_make_composite_5_rows_returns_pil_image(tmp_path: Path) -> None:
    images = [_write_fake_image(tmp_path / f"img_{i}.png") for i in range(5)]
    saliencies = [np.random.rand(32, 32).astype(np.float32) for _ in range(5)]
    identities = [f"B{i}" for i in range(5)]

    composite = make_composite(images, saliencies, identities, tile_size=128)

    assert isinstance(composite, Image.Image)
    # 2 columns x 5 rows of 128 px tiles, no header.
    assert composite.size == (256, 640)


def test_make_composite_with_title_adds_header(tmp_path: Path) -> None:
    images = [_write_fake_image(tmp_path / f"img_{i}.png") for i in range(5)]
    saliencies = [np.random.rand(16, 16).astype(np.float32) for _ in range(5)]
    identities = [f"B{i}" for i in range(5)]

    composite = make_composite(
        images,
        saliencies,
        identities,
        tile_size=64,
        title="Test page",
    )
    # Header is 28 px above the 5x64=320 px tile column.
    w, h = composite.size
    assert w == 128
    assert h == 28 + 5 * 64


def test_make_composite_wrong_count_raises(tmp_path: Path) -> None:
    images = [_write_fake_image(tmp_path / "i.png")]
    saliencies = [np.random.rand(8, 8).astype(np.float32)]
    identities = ["B1"]
    with pytest.raises(ValueError, match="exactly 5"):
        make_composite(images, saliencies, identities)


def test_make_composite_length_mismatch_raises(tmp_path: Path) -> None:
    images = [_write_fake_image(tmp_path / f"i{i}.png") for i in range(5)]
    saliencies = [np.random.rand(8, 8).astype(np.float32) for _ in range(4)]
    identities = ["B1"] * 5
    with pytest.raises(ValueError, match="same length"):
        make_composite(images, saliencies, identities)


def test_composite_from_saliency_images(tmp_path: Path) -> None:
    items = [
        SaliencyImage(
            identity=f"B{i}",
            image_path=_write_fake_image(tmp_path / f"img_{i}.png"),
            saliency=np.random.rand(16, 16).astype(np.float32),
            method="vanilla",
        )
        for i in range(6)  # provide more than rows; helper trims.
    ]
    composite = composite_from_saliency_images(items, rows=5, tile_size=64)
    assert isinstance(composite, Image.Image)
    assert composite.size == (128, 5 * 64)
