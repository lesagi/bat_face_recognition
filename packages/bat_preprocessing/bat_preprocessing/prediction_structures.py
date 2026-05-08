"""Prediction data structures for the bat-face preprocessing pipeline.

This module defines standardised dataclasses for storing and managing model
predictions (segmentation masks, pose keypoints, bounding boxes) so they can
be cached and reused across augmented copies of the same input image.

Ported from ``app/siamese_preprocessing/prediction_structures.py`` — pure
Python / numpy, no framework dependencies.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SegmentationPrediction:
    """A single YOLO segmentation prediction."""

    mask: np.ndarray  # 2-D binary mask (0/1)
    confidence: float  # detection confidence in [0, 1]
    bounding_box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)

    class_id: Optional[int] = None
    class_name: Optional[str] = None

    original_image_shape: Optional[Tuple[int, int]] = None  # (h, w)
    model_resolution: Optional[Tuple[int, int]] = None  # (h, w)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if self.mask.max() > 1:
            self.mask = (self.mask > 127).astype(np.uint8)

        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        if len(self.bounding_box) != 4:
            raise ValueError(
                f"Bounding box must have 4 values, got {len(self.bounding_box)}"
            )

        if self.mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got {self.mask.ndim}D")

    @property
    def mask_area(self) -> int:
        return int(np.sum(self.mask))

    @property
    def mask_center(self) -> Tuple[float, float]:
        if self.mask_area == 0:
            return (0.0, 0.0)
        ys, xs = np.where(self.mask > 0)
        return (float(np.mean(xs)), float(np.mean(ys)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask": self.mask.tolist(),
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "original_image_shape": self.original_image_shape,
            "model_resolution": self.model_resolution,
            "timestamp": self.timestamp.isoformat(),
            "mask_area": self.mask_area,
            "mask_center": self.mask_center,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentationPrediction":
        mask = np.array(data["mask"], dtype=np.uint8)
        timestamp = datetime.fromisoformat(data["timestamp"])
        return cls(
            mask=mask,
            confidence=data["confidence"],
            bounding_box=tuple(data["bounding_box"]),
            class_id=data.get("class_id"),
            class_name=data.get("class_name"),
            original_image_shape=data.get("original_image_shape"),
            model_resolution=data.get("model_resolution"),
            timestamp=timestamp,
        )

    def __repr__(self) -> str:
        return (
            f"SegmentationPrediction(mask_shape={self.mask.shape}, "
            f"confidence={self.confidence:.3f}, "
            f"bbox={self.bounding_box}, area={self.mask_area})"
        )


@dataclass
class PosePrediction:
    """A single YOLO pose-estimation prediction."""

    keypoints: np.ndarray  # (N, 3) where columns are (x, y, conf)
    bounding_box: Tuple[float, float, float, float]
    confidence: float

    class_id: Optional[int] = None
    class_name: Optional[str] = None

    original_image_shape: Optional[Tuple[int, int]] = None
    model_resolution: Optional[Tuple[int, int]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if self.keypoints.ndim != 2 or self.keypoints.shape[1] != 3:
            raise ValueError(
                f"Keypoints must have shape (N, 3), got {self.keypoints.shape}"
            )

        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        if len(self.bounding_box) != 4:
            raise ValueError(
                f"Bounding box must have 4 values, got {len(self.bounding_box)}"
            )

    @property
    def num_keypoints(self) -> int:
        return int(len(self.keypoints))

    @property
    def valid_keypoints(self) -> np.ndarray:
        return self.keypoints[self.keypoints[:, 2] > 0.0]

    @property
    def keypoint_centers(self) -> List[Tuple[float, float]]:
        return [(float(x), float(y)) for x, y, _ in self.valid_keypoints]

    def get_keypoint(self, index: int) -> Optional[Tuple[float, float, float]]:
        if 0 <= index < len(self.keypoints):
            x, y, conf = self.keypoints[index]
            return (float(x), float(y), float(conf))
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keypoints": self.keypoints.tolist(),
            "bounding_box": self.bounding_box,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "original_image_shape": self.original_image_shape,
            "model_resolution": self.model_resolution,
            "timestamp": self.timestamp.isoformat(),
            "num_keypoints": self.num_keypoints,
            "keypoint_centers": self.keypoint_centers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PosePrediction":
        keypoints = np.array(data["keypoints"], dtype=np.float32)
        timestamp = datetime.fromisoformat(data["timestamp"])
        return cls(
            keypoints=keypoints,
            bounding_box=tuple(data["bounding_box"]),
            confidence=data["confidence"],
            class_id=data.get("class_id"),
            class_name=data.get("class_name"),
            original_image_shape=data.get("original_image_shape"),
            model_resolution=data.get("model_resolution"),
            timestamp=timestamp,
        )

    def __repr__(self) -> str:
        return (
            f"PosePrediction(keypoints={self.num_keypoints}, "
            f"confidence={self.confidence:.3f}, bbox={self.bounding_box})"
        )


@dataclass
class PredictionBundle:
    """Container for all predictions associated with a single image."""

    segmentation: Optional[SegmentationPrediction] = None
    pose: Optional[PosePrediction] = None

    image_path: Optional[str] = None
    image_hash: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    processing_steps: List[str] = field(default_factory=list)
    model_versions: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image_path and not self.image_hash:
            self.image_hash = hashlib.md5(self.image_path.encode()).hexdigest()

    @property
    def has_predictions(self) -> bool:
        return self.segmentation is not None or self.pose is not None

    @property
    def prediction_count(self) -> int:
        return int(self.segmentation is not None) + int(self.pose is not None)

    def add_processing_step(self, step: str) -> None:
        self.processing_steps.append(f"{step}: {datetime.now().isoformat()}")

    def add_model_version(self, model_type: str, version: str) -> None:
        self.model_versions[model_type] = version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segmentation": self.segmentation.to_dict() if self.segmentation else None,
            "pose": self.pose.to_dict() if self.pose else None,
            "image_path": self.image_path,
            "image_hash": self.image_hash,
            "timestamp": self.timestamp.isoformat(),
            "processing_steps": self.processing_steps,
            "model_versions": self.model_versions,
            "has_predictions": self.has_predictions,
            "prediction_count": self.prediction_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionBundle":
        seg = (
            SegmentationPrediction.from_dict(data["segmentation"])
            if data.get("segmentation")
            else None
        )
        pose = PosePrediction.from_dict(data["pose"]) if data.get("pose") else None
        timestamp = datetime.fromisoformat(data["timestamp"])
        return cls(
            segmentation=seg,
            pose=pose,
            image_path=data.get("image_path"),
            image_hash=data.get("image_hash"),
            timestamp=timestamp,
            processing_steps=list(data.get("processing_steps", [])),
            model_versions=dict(data.get("model_versions", {})),
        )

    def save_to_file(self, filepath: str, format: str = "json") -> None:
        if format.lower() == "json":
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif format.lower() == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")

    @classmethod
    def load_from_file(cls, filepath: str, format: str = "json") -> "PredictionBundle":
        if format.lower() == "json":
            with open(filepath, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        if format.lower() == "pickle":
            with open(filepath, "rb") as f:
                obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise ValueError(f"Pickle did not contain a {cls.__name__}")
            return obj
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")

    def __repr__(self) -> str:
        h = self.image_hash[:8] if self.image_hash else "None"
        return (
            f"PredictionBundle(predictions={self.prediction_count}, "
            f"has_segmentation={self.segmentation is not None}, "
            f"has_pose={self.pose is not None}, image_hash={h})"
        )


@dataclass
class PredictionMetadata:
    """Bookkeeping for a cached prediction entry."""

    prediction_id: str
    source_image_path: str

    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    segmentation_model_version: Optional[str] = None
    pose_model_version: Optional[str] = None

    cache_size_bytes: int = 0
    is_persistent: bool = False

    def update_access(self) -> None:
        self.last_accessed = datetime.now()
        self.access_count += 1

    def get_age_seconds(self) -> float:
        return (datetime.now() - self.created_at).total_seconds()

    def get_access_frequency(self) -> float:
        age = self.get_age_seconds()
        return self.access_count / age if age > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "source_image_path": self.source_image_path,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "segmentation_model_version": self.segmentation_model_version,
            "pose_model_version": self.pose_model_version,
            "cache_size_bytes": self.cache_size_bytes,
            "is_persistent": self.is_persistent,
            "age_seconds": self.get_age_seconds(),
            "access_frequency": self.get_access_frequency(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionMetadata":
        return cls(
            prediction_id=data["prediction_id"],
            source_image_path=data["source_image_path"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=int(data["access_count"]),
            segmentation_model_version=data.get("segmentation_model_version"),
            pose_model_version=data.get("pose_model_version"),
            cache_size_bytes=int(data.get("cache_size_bytes", 0)),
            is_persistent=bool(data.get("is_persistent", False)),
        )

    def __repr__(self) -> str:
        return (
            f"PredictionMetadata(id={self.prediction_id[:8]}, "
            f"accesses={self.access_count}, "
            f"age={self.get_age_seconds():.1f}s, "
            f"size={self.cache_size_bytes} bytes)"
        )
