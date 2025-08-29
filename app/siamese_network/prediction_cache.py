"""
Prediction cache mechanism for Siamese Network preprocessing pipeline.

This module provides caching functionality for model predictions to avoid
redundant model inference calls when processing augmented images.
"""

import os
import json
import pickle
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
import time

from .prediction_structures import (
    SegmentationPrediction, 
    PosePrediction, 
    PredictionBundle,
    PredictionMetadata
)


@dataclass
class CacheConfig:
    """Configuration for prediction caching."""
    
    # Cache settings
    max_cache_size: int = 1000  # Maximum number of cached predictions
    max_memory_mb: int = 512    # Maximum memory usage in MB
    cache_ttl_hours: int = 24   # Time-to-live for cached predictions in hours
    
    # Persistence settings
    enable_persistence: bool = False
    cache_dir: str = "cache/predictions"
    persistence_format: str = "json"  # "json" or "pickle"
    
    # Cleanup settings
    cleanup_interval_seconds: int = 3600  # Cleanup interval in seconds
    enable_auto_cleanup: bool = True
    
    # Performance settings
    enable_compression: bool = False
    enable_async_io: bool = True


class CacheEntry:
    """Single cache entry containing predictions and metadata."""
    
    def __init__(
        self, 
        key: str, 
        predictions: PredictionBundle,
        metadata: PredictionMetadata
    ):
        self.key = key
        self.predictions = predictions
        self.metadata = metadata
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.metadata.update_access()
    
    def get_age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def is_expired(self, ttl_hours: int) -> bool:
        """Check if cache entry is expired."""
        return self.get_age_seconds() > (ttl_hours * 3600)
    
    def get_size_bytes(self) -> int:
        """Estimate size of cache entry in bytes."""
        # Rough estimation based on prediction data
        size = 0
        
        if self.predictions.segmentation:
            mask_size = self.predictions.segmentation.mask.nbytes
            size += mask_size + 1000  # Add overhead for metadata
        
        if self.predictions.pose:
            keypoints_size = self.predictions.pose.keypoints.nbytes
            size += keypoints_size + 500  # Add overhead for metadata
        
        # Add metadata size
        size += len(self.key) + 1000
        
        return size


class PredictionCache:
    """Main prediction cache implementation."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize prediction cache.
        
        Args:
            config: Cache configuration. If None, uses default config.
        """
        self.config = config or CacheConfig()
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Initialize cache directory if persistence enabled
        if self.config.enable_persistence:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Start cleanup thread if enabled
        if self.config.enable_auto_cleanup:
            self._start_cleanup_thread()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def close(self):
        """Close cache and cleanup resources."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5.0)
        
        # Save cache to disk if persistence enabled
        if self.config.enable_persistence:
            self._save_cache_to_disk()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while not self._stop_cleanup.wait(self.config.cleanup_interval_seconds):
                try:
                    self._cleanup_expired_entries()
                except Exception as e:
                    print(f"⚠️ Cache cleanup error: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _generate_cache_key(self, image_path: str, model_type: str) -> str:
        """Generate cache key for image and model type."""
        # Create hash from image path and model type
        key_string = f"{image_path}:{model_type}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(
        self, 
        image_path: str, 
        model_type: str
    ) -> Optional[PredictionBundle]:
        """Get predictions from cache.
        
        Args:
            image_path: Path to the image
            model_type: Type of model ("segmentation", "pose", or "both")
        
        Returns:
            Cached predictions if available, None otherwise
        """
        with self._lock:
            self._stats['total_requests'] += 1
            
            # Generate cache key
            cache_key = self._generate_cache_key(image_path, model_type)
            
            # Check if entry exists
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                # Check if expired
                if entry.is_expired(self.config.cache_ttl_hours):
                    del self._cache[cache_key]
                    self._stats['misses'] += 1
                    return None
                
                # Update access statistics
                entry.update_access()
                self._stats['hits'] += 1
                
                return entry.predictions
            
            self._stats['misses'] += 1
            return None
    
    def put(
        self, 
        image_path: str, 
        model_type: str, 
        predictions: PredictionBundle
    ) -> bool:
        """Store predictions in cache.
        
        Args:
            image_path: Path to the image
            model_type: Type of model ("segmentation", "pose", or "both")
            predictions: Prediction bundle to cache
        
        Returns:
            True if successfully cached, False otherwise
        """
        with self._lock:
            try:
                # Generate cache key
                cache_key = self._generate_cache_key(image_path, model_type)
                
                # Create metadata
                metadata = PredictionMetadata(
                    prediction_id=cache_key,
                    source_image_path=image_path
                )
                
                # Create cache entry
                entry = CacheEntry(cache_key, predictions, metadata)
                
                # Check cache size limits
                if len(self._cache) >= self.config.max_cache_size:
                    self._evict_oldest_entries(1)
                
                # Check memory limits
                if self._get_cache_memory_mb() > self.config.max_memory_mb:
                    self._evict_largest_entries(1)
                
                # Store entry
                self._cache[cache_key] = entry
                
                # Save to disk if persistence enabled
                if self.config.enable_persistence:
                    self._save_entry_to_disk(cache_key, entry)
                
                return True
                
            except Exception as e:
                print(f"⚠️ Failed to cache predictions: {e}")
                return False
    
    def _evict_oldest_entries(self, count: int):
        """Evict oldest cache entries."""
        if not self._cache:
            return
        
        # Sort by creation time and remove oldest
        sorted_entries = sorted(
            self._cache.items(), 
            key=lambda x: x[1].created_at
        )
        
        for i in range(min(count, len(sorted_entries))):
            key, entry = sorted_entries[i]
            del self._cache[key]
            self._stats['evictions'] += 1
            
            # Remove from disk if persistence enabled
            if self.config.enable_persistence:
                self._remove_entry_from_disk(key)
    
    def _evict_largest_entries(self, count: int):
        """Evict largest cache entries by memory usage."""
        if not self._cache:
            return
        
        # Sort by size and remove largest
        sorted_entries = sorted(
            self._cache.items(), 
            key=lambda x: x[1].get_size_bytes(),
            reverse=True
        )
        
        for i in range(min(count, len(sorted_entries))):
            key, entry = sorted_entries[i]
            del self._cache[key]
            self._stats['evictions'] += 1
            
            # Remove from disk if persistence enabled
            if self.config.enable_persistence:
                self._remove_entry_from_disk(key)
    
    def _get_cache_memory_mb(self) -> float:
        """Get current cache memory usage in MB."""
        total_bytes = sum(entry.get_size_bytes() for entry in self._cache.values())
        return total_bytes / (1024 * 1024)
    
    def _cleanup_expired_entries(self):
        """Remove expired cache entries."""
        with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.is_expired(self.config.cache_ttl_hours):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                self._stats['evictions'] += 1
                
                # Remove from disk if persistence enabled
                if self.config.enable_persistence:
                    self._remove_entry_from_disk(key)
    
    def _save_entry_to_disk(self, key: str, entry: CacheEntry):
        """Save cache entry to disk."""
        try:
            filepath = os.path.join(self.config.cache_dir, f"{key}.{self.config.persistence_format}")
            
            if self.config.persistence_format == "json":
                with open(filepath, 'w') as f:
                    json.dump(entry.predictions.to_dict(), f, indent=2)
            elif self.config.persistence_format == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(entry.predictions, f)
            
            # Update metadata
            entry.metadata.is_persistent = True
            entry.metadata.cache_size_bytes = entry.get_size_bytes()
            
        except Exception as e:
            print(f"⚠️ Failed to save cache entry to disk: {e}")
    
    def _remove_entry_from_disk(self, key: str):
        """Remove cache entry from disk."""
        try:
            filepath = os.path.join(self.config.cache_dir, f"{key}.{self.config.persistence_format}")
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"⚠️ Failed to remove cache entry from disk: {e}")
    
    def _save_cache_to_disk(self):
        """Save entire cache to disk."""
        if not self.config.enable_persistence:
            return
        
        try:
            for key, entry in self._cache.items():
                self._save_entry_to_disk(key, entry)
        except Exception as e:
            print(f"⚠️ Failed to save cache to disk: {e}")
    
    def load_from_disk(self):
        """Load cache entries from disk."""
        if not self.config.enable_persistence:
            return
        
        try:
            cache_dir = self.config.cache_dir
            if not os.path.exists(cache_dir):
                return
            
            for filename in os.listdir(cache_dir):
                if not filename.endswith(f".{self.config.persistence_format}"):
                    continue
                
                key = filename[:-len(f".{self.config.persistence_format}") - 1]
                filepath = os.path.join(cache_dir, filename)
                
                try:
                    if self.config.persistence_format == "json":
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        predictions = PredictionBundle.from_dict(data)
                    elif self.config.persistence_format == "pickle":
                        with open(filepath, 'rb') as f:
                            predictions = pickle.load(f)
                    else:
                        continue
                    
                    # Create metadata
                    metadata = PredictionMetadata(
                        prediction_id=key,
                        source_image_path=predictions.image_path or "unknown"
                    )
                    
                    # Create cache entry
                    entry = CacheEntry(key, predictions, metadata)
                    
                    # Store in memory cache
                    self._cache[key] = entry
                    
                except Exception as e:
                    print(f"⚠️ Failed to load cache entry {filename}: {e}")
                    # Remove corrupted file
                    try:
                        os.remove(filepath)
                    except:
                        pass
                        
        except Exception as e:
            print(f"⚠️ Failed to load cache from disk: {e}")
    
    def clear(self):
        """Clear all cached predictions."""
        with self._lock:
            # Remove from disk if persistence enabled
            if self.config.enable_persistence:
                for key in list(self._cache.keys()):
                    self._remove_entry_from_disk(key)
            
            # Clear memory cache
            self._cache.clear()
            
            # Reset statistics
            self._stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'total_requests': 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats = self._stats.copy()
            
            # Add current cache state
            stats['cache_size'] = len(self._cache)
            stats['memory_usage_mb'] = self._get_cache_memory_mb()
            stats['hit_rate'] = (
                stats['hits'] / stats['total_requests'] 
                if stats['total_requests'] > 0 else 0.0
            )
            
            return stats
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        with self._lock:
            info = {
                'config': {
                    'max_cache_size': self.config.max_cache_size,
                    'max_memory_mb': self.config.max_memory_mb,
                    'cache_ttl_hours': self.config.cache_ttl_hours,
                    'enable_persistence': self.config.enable_persistence,
                    'persistence_format': self.config.persistence_format
                },
                'current_state': {
                    'cache_size': len(self._cache),
                    'memory_usage_mb': self._get_cache_memory_mb(),
                    'oldest_entry_age_hours': 0,
                    'newest_entry_age_hours': 0
                },
                'statistics': self.get_stats()
            }
            
            # Calculate age information
            if self._cache:
                now = datetime.now()
                ages = [
                    (now - entry.created_at).total_seconds() / 3600 
                    for entry in self._cache.values()
                ]
                info['current_state']['oldest_entry_age_hours'] = max(ages)
                info['current_state']['newest_entry_age_hours'] = min(ages)
            
            return info
    
    def __len__(self) -> int:
        """Get number of cached entries."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache


class CacheManager:
    """High-level cache management interface."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager.
        
        Args:
            config: Cache configuration. If None, uses default config.
        """
        self.config = config or CacheConfig()
        self._cache = PredictionCache(self.config)
        
        # Load existing cache from disk
        if self.config.enable_persistence:
            self._cache.load_from_disk()
    
    def get_predictions(
        self, 
        image_path: str, 
        model_type: str = "both"
    ) -> Optional[PredictionBundle]:
        """Get predictions from cache.
        
        Args:
            image_path: Path to the image
            model_type: Type of model ("segmentation", "pose", or "both")
        
        Returns:
            Cached predictions if available, None otherwise
        """
        return self._cache.get(image_path, model_type)
    
    def cache_predictions(
        self, 
        image_path: str, 
        model_type: str, 
        predictions: PredictionBundle
    ) -> bool:
        """Cache predictions.
        
        Args:
            image_path: Path to the image
            model_type: Type of model ("segmentation", "pose", or "both")
            predictions: Prediction bundle to cache
        
        Returns:
            True if successfully cached, False otherwise
        """
        return self._cache.put(image_path, model_type, predictions)
    
    def get_or_cache_predictions(
        self, 
        image_path: str, 
        model_type: str,
        prediction_generator: callable
    ) -> Optional[PredictionBundle]:
        """Get predictions from cache or generate and cache them.
        
        Args:
            image_path: Path to the image
            model_type: Type of model ("segmentation", "pose", or "both")
            prediction_generator: Function to generate predictions if not cached
        
        Returns:
            Predictions from cache or newly generated
        """
        # Try to get from cache first
        cached = self.get_predictions(image_path, model_type)
        if cached is not None:
            return cached
        
        # Generate predictions
        try:
            predictions = prediction_generator()
            if predictions is not None:
                # Cache the predictions
                self.cache_predictions(image_path, model_type, predictions)
                return predictions
        except Exception as e:
            print(f"⚠️ Failed to generate predictions: {e}")
        
        return None
    
    def clear_cache(self):
        """Clear all cached predictions."""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        return self._cache.get_cache_info()
    
    def close(self):
        """Close cache manager and cleanup resources."""
        self._cache.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
