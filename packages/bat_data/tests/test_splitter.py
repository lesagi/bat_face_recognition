"""Identity-disjoint guarantee + reproducibility under seed."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from bat_core import ImageRecord, Manifest  # noqa: E402
from bat_core.exceptions import InvalidManifestError  # noqa: E402
from bat_data.splitter import IdentitySplitter  # noqa: E402


def _make_record(idx: int, identity: str, *, split: str = "train") -> ImageRecord:
    return ImageRecord(
        path=Path(f"/data/img_{idx}.png"),
        identity=identity,
        species="rousettus",
        background="random",
        source="video",
        augmented=False,
        split=split,
        quality=10.0 + idx,
    )


def _make_manifest(num_identities: int, per_id: int = 4) -> Manifest:
    records = []
    n = 0
    for i in range(num_identities):
        ident = f"bat_{i:03d}"
        for _ in range(per_id):
            records.append(_make_record(n, ident))
            n += 1
    return Manifest.from_records(records)


def test_splitter_produces_identity_disjoint_partition() -> None:
    manifest = _make_manifest(num_identities=20)
    splitter = IdentitySplitter(val_fraction=0.2, test_fraction=0.2, seed=7)
    out = splitter.split(manifest)
    out.assert_identity_disjoint()
    train_ids = out.identities("train")
    val_ids = out.identities("val")
    test_ids = out.identities("test")
    # Cover all identities, no leak across splits
    assert (train_ids | val_ids | test_ids) == manifest.identities()
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)


def test_splitter_reproducible_under_same_seed() -> None:
    manifest = _make_manifest(num_identities=15)
    a = IdentitySplitter(val_fraction=0.2, test_fraction=0.2, seed=42).split(manifest)
    b = IdentitySplitter(val_fraction=0.2, test_fraction=0.2, seed=42).split(manifest)
    assert a.identities("train") == b.identities("train")
    assert a.identities("val") == b.identities("val")
    assert a.identities("test") == b.identities("test")
    assert a.manifest_hash == b.manifest_hash


def test_splitter_different_seeds_change_partition() -> None:
    # Strong probabilistic guarantee with N=30 identities.
    manifest = _make_manifest(num_identities=30)
    a = IdentitySplitter(val_fraction=0.2, test_fraction=0.2, seed=1).split(manifest)
    b = IdentitySplitter(val_fraction=0.2, test_fraction=0.2, seed=2).split(manifest)
    assert a.identities("test") != b.identities("test")


def test_splitter_records_inherit_split_from_identity() -> None:
    manifest = _make_manifest(num_identities=12)
    out = IdentitySplitter(val_fraction=0.25, test_fraction=0.25, seed=3).split(manifest)
    # Every record's split must match the bucket its identity landed in.
    train_ids = out.identities("train")
    val_ids = out.identities("val")
    test_ids = out.identities("test")
    for r in out.records:
        if r.split == "train":
            assert r.identity in train_ids
        elif r.split == "val":
            assert r.identity in val_ids
        else:
            assert r.identity in test_ids


def test_splitter_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError):
        IdentitySplitter(val_fraction=0.0, test_fraction=0.2)
    with pytest.raises(ValueError):
        IdentitySplitter(val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError):
        IdentitySplitter(val_fraction=0.2, test_fraction=-0.1)


def test_splitter_raises_when_too_few_identities() -> None:
    manifest = _make_manifest(num_identities=2)
    splitter = IdentitySplitter(val_fraction=0.2, test_fraction=0.2, seed=0)
    with pytest.raises(InvalidManifestError):
        splitter.split(manifest)


def test_split_manifest_assert_disjoint_passes() -> None:
    manifest = _make_manifest(num_identities=12)
    out = IdentitySplitter(val_fraction=0.2, test_fraction=0.2, seed=4).split(manifest)
    # Should not raise.
    out.assert_identity_disjoint()
