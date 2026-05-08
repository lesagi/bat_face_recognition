"""Manifest CSV round-trip preserves hash + records."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pandas")

from bat_core import ImageRecord, Manifest  # noqa: E402
from bat_core.exceptions import InvalidManifestError  # noqa: E402
from bat_data.manifest import manifest_from_csv, manifest_to_csv  # noqa: E402


def _make_manifest() -> Manifest:
    records = [
        ImageRecord(
            path=Path("/data/r--W--001.png"),
            identity="W",
            species="rousettus",
            background="random",
            source="video",
            augmented=False,
            split="train",
            quality=12.5,
        ),
        ImageRecord(
            path=Path("/data/r--X--002--aug001.png"),
            identity="X",
            species="rousettus",
            background="green",
            source="video",
            augmented=True,
            split="val",
            quality=88.2,
        ),
        ImageRecord(
            path=Path("/data/r--Y--003.png"),
            identity="Y",
            species="rousettus",
            background="original",
            source="still",
            augmented=False,
            split="test",
            quality=4.0,
        ),
    ]
    return Manifest.from_records(records)


def test_manifest_csv_round_trip(tmp_path) -> None:
    m = _make_manifest()
    csv_path = tmp_path / "manifest.csv"
    out = manifest_to_csv(m, csv_path)
    assert out.exists()
    sidecar = out.with_suffix(out.suffix + ".hash")
    assert sidecar.exists()

    loaded = manifest_from_csv(csv_path)
    assert loaded.manifest_hash == m.manifest_hash
    assert len(loaded.records) == len(m.records)
    for original, restored in zip(m.records, loaded.records):
        assert restored.identity == original.identity
        assert restored.species == original.species
        assert restored.background == original.background
        assert restored.source == original.source
        assert restored.split == original.split
        assert restored.augmented == original.augmented
        assert pytest.approx(restored.quality, rel=1e-6) == original.quality
        assert str(restored.path) == str(original.path)


def test_manifest_method_helpers(tmp_path) -> None:
    """``Manifest.to_csv`` / ``Manifest.from_csv`` thin shortcuts."""
    m = _make_manifest()
    csv_path = tmp_path / "manifest.csv"
    m.to_csv(csv_path)
    loaded = Manifest.from_csv(csv_path)
    assert loaded.manifest_hash == m.manifest_hash


def test_manifest_csv_detects_hash_tampering(tmp_path) -> None:
    m = _make_manifest()
    csv_path = tmp_path / "manifest.csv"
    manifest_to_csv(m, csv_path)
    sidecar = csv_path.with_suffix(csv_path.suffix + ".hash")
    sidecar.write_text("0" * 64, encoding="utf-8")
    with pytest.raises(InvalidManifestError):
        manifest_from_csv(csv_path)


def test_manifest_csv_missing_columns_rejected(tmp_path) -> None:
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text("path,identity\n/x.png,A\n", encoding="utf-8")
    with pytest.raises(InvalidManifestError):
        manifest_from_csv(csv_path)
