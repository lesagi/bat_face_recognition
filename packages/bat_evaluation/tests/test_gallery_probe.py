"""Gallery / probe split tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from bat_core import ImageRecord, Manifest
from bat_evaluation import split_gallery_probe, split_manifest_for_identification


def _rec(identity: str, idx: int, split: str = "test") -> ImageRecord:
    return ImageRecord(
        path=Path(f"data/{identity}_{idx}.png"),
        identity=identity,
        species="rousettus",
        background="random",
        source="video",
        augmented=False,
        split=split,  # type: ignore[arg-type]
        quality=0.5,
    )


def test_split_uses_first_image_per_identity():
    records = [
        _rec("alpha", 0),
        _rec("alpha", 1),
        _rec("alpha", 2),
        _rec("beta", 0),
        _rec("beta", 1),
    ]
    split = split_gallery_probe(records, gallery_size=1, min_probe_size=1)
    assert len(split.gallery_records) == 2
    assert len(split.probe_records) == 3
    # Gallery picks lowest-path entry per identity (sorted by path).
    gallery_ids = {r.identity for r in split.gallery_records}
    assert gallery_ids == {"alpha", "beta"}
    assert split.skipped_identities == ()


def test_split_skips_identities_with_too_few_images():
    records = [
        _rec("alpha", 0),
        _rec("alpha", 1),
        _rec("singleton", 0),  # only one image -> can't satisfy gallery+probe
    ]
    split = split_gallery_probe(records, gallery_size=1, min_probe_size=1)
    assert {"singleton"} == set(split.skipped_identities)
    assert all(r.identity == "alpha" for r in split.gallery_records + split.probe_records)


def test_split_invalid_args():
    with pytest.raises(ValueError):
        split_gallery_probe([], gallery_size=0)
    with pytest.raises(ValueError):
        split_gallery_probe([], min_probe_size=0)


def test_split_manifest_for_identification_filters_split():
    records = [
        _rec("alpha", 0, split="train"),
        _rec("alpha", 1, split="train"),
        _rec("beta", 0, split="test"),
        _rec("beta", 1, split="test"),
    ]
    manifest = Manifest.from_records(records)
    split = split_manifest_for_identification(manifest, split="test")
    seen_ids = {r.identity for r in split.gallery_records + split.probe_records}
    assert seen_ids == {"beta"}
