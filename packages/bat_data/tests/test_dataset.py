"""Basic dataset indexing test (torch-dependent — skipped when missing)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic")
torch = pytest.importorskip("torch")

from bat_core import ImageRecord, Manifest  # noqa: E402
from bat_data.dataset import BatDataset  # noqa: E402


def _record(idx: int, identity: str, split: str = "train") -> ImageRecord:
    return ImageRecord(
        path=Path(f"/fake/img_{idx}.png"),
        identity=identity,
        species="rousettus",
        background="random",
        source="video",
        augmented=False,
        split=split,
        quality=1.0,
    )


def _stub_loader(_: Path) -> torch.Tensor:
    return torch.zeros(3, 8, 8, dtype=torch.float32)


def test_default_image_loader_imagenet_normalization(tmp_path: Path) -> None:
    """``normalize='imagenet'`` applies (x/255 - mean)/std; default keeps [0,1]."""
    np = pytest.importorskip("numpy")
    pil_image = pytest.importorskip("PIL.Image")

    from bat_data.dataset import IMAGENET_MEAN, IMAGENET_STD, default_image_loader

    # Solid mid-grey image: every pixel is 128 -> 128/255 in [0,1].
    arr = np.full((16, 16, 3), 128, dtype="uint8")
    img_path = tmp_path / "grey.png"
    pil_image.fromarray(arr).save(img_path)

    raw = default_image_loader(img_path, image_size=8)
    assert raw.shape == (3, 8, 8)
    assert torch.allclose(raw, torch.full_like(raw, 128.0 / 255.0), atol=1e-3)

    norm = default_image_loader(img_path, image_size=8, normalize="imagenet")
    base = 128.0 / 255.0
    for c in range(3):
        expected = (base - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        assert torch.allclose(norm[c], torch.full((8, 8), expected), atol=1e-3)


def test_default_image_loader_rejects_unknown_normalize(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pil_image = pytest.importorskip("PIL.Image")

    from bat_data.dataset import default_image_loader

    img_path = tmp_path / "x.png"
    pil_image.fromarray(np.zeros((4, 4, 3), dtype="uint8")).save(img_path)
    with pytest.raises(ValueError, match="unknown normalize"):
        default_image_loader(img_path, image_size=4, normalize="zscore")


def test_basic_indexing() -> None:
    records = [
        _record(0, "A", "train"),
        _record(1, "B", "train"),
        _record(2, "A", "train"),
        _record(3, "C", "val"),  # filtered out
    ]
    manifest = Manifest.from_records(records)
    ds = BatDataset(manifest, split="train", loader=_stub_loader)

    assert len(ds) == 3
    assert ds.num_classes == 2  # only A and B in the train split
    tensor, identity_int, record = ds[0]
    assert tuple(tensor.shape) == (3, 8, 8)
    assert isinstance(identity_int, int)
    # Stable mapping: A < B alphabetically
    assert ds.identity_to_int == {"A": 0, "B": 1}
    assert ds.identity_for(identity_int) == record.identity


def test_empty_split_rejected() -> None:
    records = [_record(0, "A", "train")]
    manifest = Manifest.from_records(records)
    with pytest.raises(ValueError):
        BatDataset(manifest, split="val", loader=_stub_loader)


def test_invalid_split_value() -> None:
    records = [_record(0, "A", "train")]
    manifest = Manifest.from_records(records)
    with pytest.raises(ValueError):
        BatDataset(manifest, split="holdout", loader=_stub_loader)  # type: ignore[arg-type]
