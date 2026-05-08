"""Prediction caching for the bat-face preprocessing pipeline.

Ports ``app/siamese_preprocessing/prediction_cache.py``. Pure Python; no
framework dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from .prediction_structures import (
    PredictionBundle,
    PredictionMetadata,
)


@dataclass
class CacheConfig:
    """Configuration for the prediction cache."""

    max_cache_size: int = 1000
    max_memory_mb: int = 512
    cache_ttl_hours: int = 24

    enable_persistence: bool = False
    cache_dir: str = "cache/predictions"
    persistence_format: str = "json"  # "json" or "pickle"

    cleanup_interval_seconds: int = 3600
    enable_auto_cleanup: bool = True

    enable_compression: bool = False
    enable_async_io: bool = True


class CacheEntry:
    """A single cache entry: predictions plus metadata."""

    def __init__(
        self,
        key: str,
        predictions: PredictionBundle,
        metadata: PredictionMetadata,
    ) -> None:
        self.key = key
        self.predictions = predictions
        self.metadata = metadata
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0

    def update_access(self) -> None:
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.metadata.update_access()

    def get_age_seconds(self) -> float:
        return (datetime.now() - self.created_at).total_seconds()

    def is_expired(self, ttl_hours: int) -> bool:
        return self.get_age_seconds() > (ttl_hours * 3600)

    def get_size_bytes(self) -> int:
        size = 0
        if self.predictions.segmentation is not None:
            size += int(self.predictions.segmentation.mask.nbytes) + 1000
        if self.predictions.pose is not None:
            size += int(self.predictions.pose.keypoints.nbytes) + 500
        size += len(self.key) + 1000
        return size


class PredictionCache:
    """In-memory (optionally persistent) cache for prediction bundles."""

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        self.config = config or CacheConfig()
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()

        self._stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0,
        }

        if self.config.enable_persistence:
            os.makedirs(self.config.cache_dir, exist_ok=True)

        if self.config.enable_auto_cleanup:
            self._start_cleanup_thread()

    # ---- context manager / lifecycle -----------------------------------

    def __enter__(self) -> "PredictionCache":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5.0)

        if self.config.enable_persistence:
            self._save_cache_to_disk()

    def _start_cleanup_thread(self) -> None:
        def cleanup_worker() -> None:
            while not self._stop_cleanup.wait(self.config.cleanup_interval_seconds):
                try:
                    self._cleanup_expired_entries()
                except Exception as exc:  # pragma: no cover - background task
                    print(f"Cache cleanup error: {exc}")

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    # ---- core API -------------------------------------------------------

    def _generate_cache_key(self, image_path: str, model_type: str) -> str:
        return hashlib.md5(f"{image_path}:{model_type}".encode()).hexdigest()

    def get(self, image_path: str, model_type: str) -> Optional[PredictionBundle]:
        with self._lock:
            self._stats["total_requests"] += 1
            cache_key = self._generate_cache_key(image_path, model_type)

            if cache_key in self._cache:
                entry = self._cache[cache_key]
                if entry.is_expired(self.config.cache_ttl_hours):
                    del self._cache[cache_key]
                    self._stats["misses"] += 1
                    return None
                entry.update_access()
                self._stats["hits"] += 1
                return entry.predictions

            self._stats["misses"] += 1
            return None

    def put(
        self,
        image_path: str,
        model_type: str,
        predictions: PredictionBundle,
    ) -> bool:
        with self._lock:
            try:
                cache_key = self._generate_cache_key(image_path, model_type)
                metadata = PredictionMetadata(
                    prediction_id=cache_key,
                    source_image_path=image_path,
                )
                entry = CacheEntry(cache_key, predictions, metadata)

                if len(self._cache) >= self.config.max_cache_size:
                    self._evict_oldest_entries(1)
                if self._get_cache_memory_mb() > self.config.max_memory_mb:
                    self._evict_largest_entries(1)

                self._cache[cache_key] = entry

                if self.config.enable_persistence:
                    self._save_entry_to_disk(cache_key, entry)
                return True
            except Exception as exc:
                print(f"Failed to cache predictions: {exc}")
                return False

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    # ---- eviction & cleanup --------------------------------------------

    def _evict_oldest_entries(self, count: int) -> None:
        if not self._cache:
            return
        sorted_entries = sorted(
            self._cache.items(), key=lambda kv: kv[1].created_at
        )
        for key, _ in sorted_entries[: max(0, count)]:
            del self._cache[key]
            self._stats["evictions"] += 1
            if self.config.enable_persistence:
                self._remove_entry_from_disk(key)

    def _evict_largest_entries(self, count: int) -> None:
        if not self._cache:
            return
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda kv: kv[1].get_size_bytes(),
            reverse=True,
        )
        for key, _ in sorted_entries[: max(0, count)]:
            del self._cache[key]
            self._stats["evictions"] += 1
            if self.config.enable_persistence:
                self._remove_entry_from_disk(key)

    def _get_cache_memory_mb(self) -> float:
        total = sum(entry.get_size_bytes() for entry in self._cache.values())
        return total / (1024 * 1024)

    def _cleanup_expired_entries(self) -> None:
        with self._lock:
            expired = [
                key
                for key, entry in self._cache.items()
                if entry.is_expired(self.config.cache_ttl_hours)
            ]
            for key in expired:
                del self._cache[key]
                self._stats["evictions"] += 1
                if self.config.enable_persistence:
                    self._remove_entry_from_disk(key)

    # ---- persistence ----------------------------------------------------

    def _save_entry_to_disk(self, key: str, entry: CacheEntry) -> None:
        try:
            filepath = os.path.join(
                self.config.cache_dir, f"{key}.{self.config.persistence_format}"
            )
            if self.config.persistence_format == "json":
                with open(filepath, "w") as f:
                    json.dump(entry.predictions.to_dict(), f, indent=2)
            elif self.config.persistence_format == "pickle":
                with open(filepath, "wb") as f:
                    pickle.dump(entry.predictions, f)
            entry.metadata.is_persistent = True
            entry.metadata.cache_size_bytes = entry.get_size_bytes()
        except Exception as exc:
            print(f"Failed to save cache entry to disk: {exc}")

    def _remove_entry_from_disk(self, key: str) -> None:
        try:
            filepath = os.path.join(
                self.config.cache_dir, f"{key}.{self.config.persistence_format}"
            )
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as exc:
            print(f"Failed to remove cache entry from disk: {exc}")

    def _save_cache_to_disk(self) -> None:
        if not self.config.enable_persistence:
            return
        try:
            for key, entry in self._cache.items():
                self._save_entry_to_disk(key, entry)
        except Exception as exc:
            print(f"Failed to save cache to disk: {exc}")

    def load_from_disk(self) -> None:
        if not self.config.enable_persistence:
            return
        try:
            cache_dir = self.config.cache_dir
            if not os.path.exists(cache_dir):
                return
            for filename in os.listdir(cache_dir):
                ext = f".{self.config.persistence_format}"
                if not filename.endswith(ext):
                    continue
                key = filename[: -len(ext)]
                filepath = os.path.join(cache_dir, filename)
                try:
                    if self.config.persistence_format == "json":
                        with open(filepath, "r") as f:
                            data = json.load(f)
                        predictions = PredictionBundle.from_dict(data)
                    elif self.config.persistence_format == "pickle":
                        with open(filepath, "rb") as f:
                            predictions = pickle.load(f)
                    else:
                        continue

                    metadata = PredictionMetadata(
                        prediction_id=key,
                        source_image_path=predictions.image_path or "unknown",
                    )
                    self._cache[key] = CacheEntry(key, predictions, metadata)
                except Exception as exc:
                    print(f"Failed to load cache entry {filename}: {exc}")
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
        except Exception as exc:
            print(f"Failed to load cache from disk: {exc}")

    # ---- introspection --------------------------------------------------

    def clear(self) -> None:
        with self._lock:
            if self.config.enable_persistence:
                for key in list(self._cache.keys()):
                    self._remove_entry_from_disk(key)
            self._cache.clear()
            self._stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "total_requests": 0,
            }

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats: Dict[str, Any] = dict(self._stats)
            stats["cache_size"] = len(self._cache)
            stats["memory_usage_mb"] = self._get_cache_memory_mb()
            stats["hit_rate"] = (
                stats["hits"] / stats["total_requests"]
                if stats["total_requests"] > 0
                else 0.0
            )
            return stats

    def get_cache_info(self) -> Dict[str, Any]:
        with self._lock:
            info: Dict[str, Any] = {
                "config": {
                    "max_cache_size": self.config.max_cache_size,
                    "max_memory_mb": self.config.max_memory_mb,
                    "cache_ttl_hours": self.config.cache_ttl_hours,
                    "enable_persistence": self.config.enable_persistence,
                    "persistence_format": self.config.persistence_format,
                },
                "current_state": {
                    "cache_size": len(self._cache),
                    "memory_usage_mb": self._get_cache_memory_mb(),
                    "oldest_entry_age_hours": 0.0,
                    "newest_entry_age_hours": 0.0,
                },
                "statistics": self.get_stats(),
            }
            if self._cache:
                now = datetime.now()
                ages = [
                    (now - entry.created_at).total_seconds() / 3600
                    for entry in self._cache.values()
                ]
                info["current_state"]["oldest_entry_age_hours"] = max(ages)
                info["current_state"]["newest_entry_age_hours"] = min(ages)
            return info


class CacheManager:
    """High-level cache management interface for the preprocessing pipeline."""

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        self.config = config or CacheConfig()
        self._cache = PredictionCache(self.config)
        if self.config.enable_persistence:
            self._cache.load_from_disk()

    def get_predictions(
        self, image_path: str, model_type: str = "both"
    ) -> Optional[PredictionBundle]:
        return self._cache.get(image_path, model_type)

    def cache_predictions(
        self, image_path: str, model_type: str, predictions: PredictionBundle
    ) -> bool:
        return self._cache.put(image_path, model_type, predictions)

    def is_cached(self, image_path: str, model_type: str = "both") -> bool:
        key = self._cache._generate_cache_key(image_path, model_type)
        return key in self._cache

    def get_or_cache_predictions(
        self,
        image_path: str,
        model_type: str,
        prediction_generator: Callable[[], Optional[PredictionBundle]],
    ) -> Optional[PredictionBundle]:
        cached = self.get_predictions(image_path, model_type)
        if cached is not None:
            return cached
        try:
            predictions = prediction_generator()
            if predictions is not None:
                self.cache_predictions(image_path, model_type, predictions)
                return predictions
        except Exception as exc:
            print(f"Failed to generate predictions: {exc}")
        return None

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        return self._cache.get_stats()

    def get_cache_info(self) -> Dict[str, Any]:
        return self._cache.get_cache_info()

    def close(self) -> None:
        self._cache.close()

    def __enter__(self) -> "CacheManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
