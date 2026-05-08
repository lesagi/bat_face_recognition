"""Tests for the prediction cache."""

from __future__ import annotations

import numpy as np
import pytest
from bat_preprocessing.prediction_cache import CacheConfig, CacheManager, PredictionCache
from bat_preprocessing.prediction_structures import PredictionBundle, SegmentationPrediction


@pytest.fixture
def bundle() -> PredictionBundle:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    seg = SegmentationPrediction(
        mask=mask,
        confidence=0.7,
        bounding_box=(0.25, 0.25, 0.75, 0.75),
        original_image_shape=(8, 8),
    )
    return PredictionBundle(segmentation=seg, image_path="/tmp/sample.jpg")


def test_cache_get_set_roundtrip(bundle: PredictionBundle) -> None:
    cache = PredictionCache(CacheConfig(enable_auto_cleanup=False))
    assert cache.get("/tmp/sample.jpg", "both") is None
    assert cache.put("/tmp/sample.jpg", "both", bundle) is True
    fetched = cache.get("/tmp/sample.jpg", "both")
    assert fetched is bundle
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    cache.close()


def test_cache_eviction_on_size(bundle: PredictionBundle) -> None:
    cache = PredictionCache(
        CacheConfig(max_cache_size=2, enable_auto_cleanup=False, max_memory_mb=10**6)
    )
    cache.put("/a", "both", bundle)
    cache.put("/b", "both", bundle)
    cache.put("/c", "both", bundle)
    assert len(cache) == 2
    cache.close()


def test_cache_manager_is_cached_and_or_cache(bundle: PredictionBundle) -> None:
    manager = CacheManager(CacheConfig(enable_auto_cleanup=False))
    assert not manager.is_cached("/x", "both")
    calls = {"n": 0}

    def gen() -> PredictionBundle:
        calls["n"] += 1
        return bundle

    out = manager.get_or_cache_predictions("/x", "both", gen)
    assert out is bundle
    assert calls["n"] == 1
    out2 = manager.get_or_cache_predictions("/x", "both", gen)
    assert out2 is bundle
    assert calls["n"] == 1, "second call should hit cache, not regenerate"
    manager.close()
