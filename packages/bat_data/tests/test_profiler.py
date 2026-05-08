"""Profiler emits expected aggregations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from bat_core import ImageRecord, Manifest  # noqa: E402
from bat_data.profiler import profile, save_profile  # noqa: E402


def _record(name: str, identity: str, *, split: str, q: float = 10.0) -> ImageRecord:
    return ImageRecord(
        path=Path(f"/data/{name}.png"),
        identity=identity,
        species="rousettus",
        background="random",
        source="video",
        augmented=name.endswith("aug001"),
        split=split,
        quality=q,
    )


def _manifest() -> Manifest:
    records = [
        _record("r--A--001", "A", split="train", q=5.0),
        _record("r--A--002", "A", split="train", q=10.0),
        _record("r--A--002--aug001", "A", split="train", q=15.0),
        _record("r--B--010", "B", split="val", q=20.0),
        _record("r--C--020", "C", split="test", q=25.0),
        _record("r--C--021", "C", split="test", q=30.0),
    ]
    return Manifest.from_records(records)


def _stub_dim_reader(_: Path) -> tuple[int, int]:
    return 224, 224


def test_profile_aggregates(tmp_path) -> None:
    m = _manifest()
    p = profile(m, dim_reader=_stub_dim_reader, image_sample_cap=10)

    assert p["total_records"] == 6
    assert p["per_class_counts"] == {"A": 3, "B": 1, "C": 2}
    assert p["identity_counts_per_split"] == {"train": 1, "val": 1, "test": 1}
    assert p["records_per_split"] == {"train": 3, "val": 1, "test": 2}
    assert p["augmented_count"] == 1
    assert p["background_breakdown"] == {"random": 6}
    assert p["source_breakdown"] == {"video": 6}

    # Image dim stats — every stub returns (224, 224)
    assert p["image_dim_stats"]["sampled"] == 6
    assert p["image_dim_stats"]["height"]["mean"] == 224.0
    assert p["image_dim_stats"]["height"]["std"] == 0.0

    q = p["quality"]
    assert q["min"] == 5.0
    assert q["max"] == 30.0
    assert pytest.approx(q["mean"], rel=1e-6) == (5 + 10 + 15 + 20 + 25 + 30) / 6


def test_save_profile_writes_json(tmp_path) -> None:
    m = _manifest()
    p = profile(m, dim_reader=_stub_dim_reader)
    out = save_profile(p, tmp_path / "profile.json")
    assert out.exists()

    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["total_records"] == 6
    assert reloaded["manifest_hash"] == m.manifest_hash
