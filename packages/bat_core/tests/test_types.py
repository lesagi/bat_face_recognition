from pathlib import Path

import pytest
from pydantic import ValidationError

from bat_core import (
    ImageRecord,
    InvalidManifestError,
    Manifest,
    Predictions,
)


def _record(identity: str, split: str, path: str = "data/foo.png") -> ImageRecord:
    return ImageRecord(
        path=Path(path),
        identity=identity,
        species="rousettus",
        background="random",
        source="video",
        augmented=False,
        split=split,  # type: ignore[arg-type]
        quality=0.5,
    )


def test_image_record_rejects_empty_identity() -> None:
    with pytest.raises(ValidationError):
        _record(identity="  ", split="train")


def test_image_record_rejects_negative_quality() -> None:
    with pytest.raises(ValidationError):
        ImageRecord(
            path=Path("a.png"),
            identity="W",
            species="rousettus",
            background="random",
            source="video",
            split="train",
            quality=-0.1,
        )


def test_image_record_rejects_unknown_species() -> None:
    with pytest.raises(ValidationError):
        ImageRecord(
            path=Path("a.png"),
            identity="W",
            species="elephant",  # type: ignore[arg-type]
            background="random",
            source="video",
            split="train",
            quality=0.5,
        )


def test_manifest_hash_is_deterministic() -> None:
    a = Manifest.from_records([_record("W", "train"), _record("H", "test")])
    b = Manifest.from_records([_record("W", "train"), _record("H", "test")])
    assert a.manifest_hash == b.manifest_hash


def test_manifest_hash_changes_when_records_change() -> None:
    a = Manifest.from_records([_record("W", "train")])
    b = Manifest.from_records([_record("H", "train")])
    assert a.manifest_hash != b.manifest_hash


def test_manifest_filter_split() -> None:
    m = Manifest.from_records(
        [
            _record("W", "train"),
            _record("H", "val"),
            _record("R", "test"),
        ]
    )
    assert {r.identity for r in m.filter_split("train")} == {"W"}
    assert {r.identity for r in m.filter_split("val")} == {"H"}
    assert m.identities("test") == {"R"}


def test_assert_identity_disjoint_passes_when_disjoint() -> None:
    m = Manifest.from_records(
        [
            _record("W", "train"),
            _record("H", "val"),
            _record("R", "test"),
        ]
    )
    m.assert_identity_disjoint()  # should not raise


def test_assert_identity_disjoint_raises_on_overlap() -> None:
    m = Manifest.from_records(
        [
            _record("W", "train"),
            _record("W", "val"),
        ]
    )
    with pytest.raises(InvalidManifestError):
        m.assert_identity_disjoint()


def test_predictions_length_mismatch() -> None:
    with pytest.raises(ValueError):
        Predictions(y_true=(0, 1), y_score=(0.1, 0.2, 0.3))
