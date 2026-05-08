"""Manifest profiler — emits per-class image counts, mean/std image
dimensions, identity counts per split, and basic quality stats.

Saved as JSON via :func:`save_profile` so trainers can attach the
artifact to MLflow runs.
"""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from bat_core import Manifest

__all__ = ["profile", "save_profile"]


def _image_dim_stats(
    paths: list[Path],
    *,
    dim_reader=None,
    sample_cap: int = 256,
) -> dict[str, Any]:
    """Read the first ``sample_cap`` images and report H/W mean+std.

    ``dim_reader`` is a ``Path -> (h, w)`` callable, defaulting to
    OpenCV. Tests override this to avoid the cv2 dependency.
    """
    if not paths:
        return {
            "sampled": 0,
            "height": {"mean": 0.0, "std": 0.0},
            "width": {"mean": 0.0, "std": 0.0},
        }

    if dim_reader is None:

        def dim_reader(p: Path) -> tuple[int, int] | None:
            import cv2

            img = cv2.imread(os.fspath(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            h, w = img.shape[:2]
            return int(h), int(w)

    sampled = paths[:sample_cap]
    heights: list[int] = []
    widths: list[int] = []
    for p in sampled:
        try:
            dims = dim_reader(p)
        except Exception:  # pragma: no cover - defensive
            dims = None
        if dims is None:
            continue
        h, w = dims
        heights.append(h)
        widths.append(w)

    def _mean_std(vals: list[int]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "std": 0.0}
        if len(vals) == 1:
            return {"mean": float(vals[0]), "std": 0.0}
        return {
            "mean": float(statistics.mean(vals)),
            "std": float(statistics.pstdev(vals)),
        }

    return {
        "sampled": len(heights),
        "height": _mean_std(heights),
        "width": _mean_std(widths),
    }


def _quality_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    if len(values) == 1:
        v = float(values[0])
        return {"min": v, "max": v, "mean": v, "std": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
    }


def profile(
    manifest: Manifest,
    *,
    dim_reader=None,
    image_sample_cap: int = 256,
) -> dict[str, Any]:
    """Return a profile dict for ``manifest``.

    Keys:
        manifest_hash: str
        total_records: int
        per_class_counts: dict[identity, int]
        identity_counts_per_split: dict[split, int]
        records_per_split: dict[split, int]
        image_dim_stats: {sampled, height: {mean, std}, width: {mean, std}}
        quality: {min, max, mean, std}
        augmented_count: int
        background_breakdown: dict[background, int]
        source_breakdown: dict[source, int]
    """
    records = manifest.records

    per_class = Counter(r.identity for r in records)
    backgrounds = Counter(r.background for r in records)
    sources = Counter(r.source for r in records)
    splits = Counter(r.split for r in records)
    identity_counts_per_split: dict[str, int] = {}
    for split in ("train", "val", "test"):
        identity_counts_per_split[split] = len({r.identity for r in records if r.split == split})

    paths = [Path(r.path) for r in records]
    qualities = [float(r.quality) for r in records]

    return {
        "manifest_hash": manifest.manifest_hash,
        "total_records": len(records),
        "per_class_counts": dict(sorted(per_class.items())),
        "identity_counts_per_split": identity_counts_per_split,
        "records_per_split": {
            "train": int(splits.get("train", 0)),
            "val": int(splits.get("val", 0)),
            "test": int(splits.get("test", 0)),
        },
        "image_dim_stats": _image_dim_stats(
            paths, dim_reader=dim_reader, sample_cap=image_sample_cap
        ),
        "quality": _quality_stats(qualities),
        "augmented_count": sum(1 for r in records if r.augmented),
        "background_breakdown": dict(sorted(backgrounds.items())),
        "source_breakdown": dict(sorted(sources.items())),
    }


def save_profile(profile_dict: dict[str, Any], path: str | os.PathLike[str]) -> Path:
    """Persist a profile dict as JSON. Returns the written path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(profile_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out
