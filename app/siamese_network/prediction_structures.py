"""
Prediction data structures for Siamese Network preprocessing pipeline.

This module defines standardized data structures for storing and managing
model predictions (segmentation masks, pose keypoints, bounding boxes)
to enable caching and reuse across augmented image processing.
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
import hashlib


@dataclass
class SegmentationPrediction:
    """Data structure for YOLO segmentation model predictions."""
    
    # Core prediction data
    mask: np.ndarray  # Binary mask (0s and 1s)
    confidence: float  # Detection confidence (0.0-1.0)
    bounding_box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in normalized coordinates
    
    # Metadata
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    
    # Processing info
    original_image_shape: Optional[Tuple[int, int]] = None  # (height, width)
    model_resolution: Optional[Tuple[int, int]] = None  # (height, width)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate and preprocess data after initialization."""
        # Ensure mask is binary
        if self.mask.max() > 1:
            self.mask = (self.mask > 127).astype(np.uint8)
        
        # Validate confidence range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        # Validate bounding box format
        if len(self.bounding_box) != 4:
            raise ValueError(f"Bounding box must have 4 values, got {len(self.bounding_box)}")
        
        # Ensure mask is 2D
        if self.mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got {self.mask.ndim}D")
    
    @property
    def mask_area(self) -> int:
        """Calculate the area of the mask."""
        return int(np.sum(self.mask))
    
    @property
    def mask_center(self) -> Tuple[float, float]:
        """Calculate the center point of the mask."""
        if self.mask_area == 0:
            return (0.0, 0.0)
        
        y_coords, x_coords = np.where(self.mask > 0)
        center_x = float(np.mean(x_coords))
        center_y = float(np.mean(y_coords))
        return (center_x, center_y)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'mask': self.mask.tolist(),
            'confidence': self.confidence,
            'bounding_box': self.bounding_box,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'original_image_shape': self.original_image_shape,
            'model_resolution': self.model_resolution,
            'timestamp': self.timestamp.isoformat(),
            'mask_area': self.mask_area,
            'mask_center': self.mask_center
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SegmentationPrediction':
        """Create instance from dictionary."""
        # Convert mask back to numpy array
        mask = np.array(data['mask'], dtype=np.uint8)
        
        # Parse timestamp
        timestamp = datetime.fromisoformat(data['timestamp'])
        
        return cls(
            mask=mask,
            confidence=data['confidence'],
            bounding_box=tuple(data['bounding_box']),
            class_id=data.get('class_id'),
            class_name=data.get('class_name'),
            original_image_shape=data.get('original_image_shape'),
            model_resolution=data.get('model_resolution'),
            timestamp=timestamp
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"SegmentationPrediction(mask_shape={self.mask.shape}, "
                f"confidence={self.confidence:.3f}, "
                f"bbox={self.bounding_box}, "
                f"area={self.mask_area})")


@dataclass
class PosePrediction:
    """Data structure for YOLO pose model predictions."""
    
    # Core prediction data
    keypoints: np.ndarray  # Shape: (N, 3) where 3 = (x, y, confidence)
    bounding_box: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in normalized coordinates
    confidence: float  # Overall detection confidence (0.0-1.0)
    
    # Metadata
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    
    # Processing info
    original_image_shape: Optional[Tuple[int, int]] = None  # (height, width)
    model_resolution: Optional[Tuple[int, int]] = None  # (height, width)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate and preprocess data after initialization."""
        # Validate keypoints format
        if self.keypoints.ndim != 2 or self.keypoints.shape[1] != 3:
            raise ValueError(f"Keypoints must have shape (N, 3), got {self.keypoints.shape}")
        
        # Validate confidence range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        # Validate bounding box format
        if len(self.bounding_box) != 4:
            raise ValueError(f"Bounding box must have 4 values, got {len(self.bounding_box)}")
    
    @property
    def num_keypoints(self) -> int:
        """Number of keypoints."""
        return len(self.keypoints)
    
    @property
    def valid_keypoints(self) -> np.ndarray:
        """Get keypoints with confidence above threshold."""
        return self.keypoints[self.keypoints[:, 2] > 0.0]
    
    @property
    def keypoint_centers(self) -> List[Tuple[float, float]]:
        """Get centers of all valid keypoints."""
        valid_kpts = self.valid_keypoints
        return [(float(x), float(y)) for x, y, _ in valid_kpts]
    
    def get_keypoint(self, index: int) -> Optional[Tuple[float, float, float]]:
        """Get specific keypoint by index."""
        if 0 <= index < len(self.keypoints):
            x, y, conf = self.keypoints[index]
            return (float(x), float(y), float(conf))
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'keypoints': self.keypoints.tolist(),
            'bounding_box': self.bounding_box,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'original_image_shape': self.original_image_shape,
            'model_resolution': self.model_resolution,
            'timestamp': self.timestamp.isoformat(),
            'num_keypoints': self.num_keypoints,
            'keypoint_centers': self.keypoint_centers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PosePrediction':
        """Create instance from dictionary."""
        # Convert keypoints back to numpy array
        keypoints = np.array(data['keypoints'], dtype=np.float32)
        
        # Parse timestamp
        timestamp = datetime.fromisoformat(data['timestamp'])
        
        return cls(
            keypoints=keypoints,
            bounding_box=tuple(data['bounding_box']),
            confidence=data['confidence'],
            class_id=data.get('class_id'),
            class_name=data.get('class_name'),
            original_image_shape=data.get('original_image_shape'),
            model_resolution=data.get('model_resolution'),
            timestamp=timestamp
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"PosePrediction(keypoints={self.num_keypoints}, "
                f"confidence={self.confidence:.3f}, "
                f"bbox={self.bounding_box})")


@dataclass
class PredictionBundle:
    """Container for all predictions related to a single image."""
    
    # Core predictions
    segmentation: Optional[SegmentationPrediction] = None
    pose: Optional[PosePrediction] = None
    
    # Metadata
    image_path: Optional[str] = None
    image_hash: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Processing info
    processing_steps: List[str] = field(default_factory=list)
    model_versions: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate image hash if image path is provided."""
        if self.image_path and not self.image_hash:
            self.image_hash = self._generate_image_hash()
    
    def _generate_image_hash(self) -> str:
        """Generate hash for the image path."""
        return hashlib.md5(self.image_path.encode()).hexdigest()
    
    @property
    def has_predictions(self) -> bool:
        """Check if any predictions are available."""
        return self.segmentation is not None or self.pose is not None
    
    @property
    def prediction_count(self) -> int:
        """Count of available predictions."""
        count = 0
        if self.segmentation:
            count += 1
        if self.pose:
            count += 1
        return count
    
    def add_processing_step(self, step: str):
        """Add a processing step to the history."""
        self.processing_steps.append(f"{step}: {datetime.now().isoformat()}")
    
    def add_model_version(self, model_type: str, version: str):
        """Add model version information."""
        self.model_versions[model_type] = version
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'segmentation': self.segmentation.to_dict() if self.segmentation else None,
            'pose': self.pose.to_dict() if self.pose else None,
            'image_path': self.image_path,
            'image_hash': self.image_hash,
            'timestamp': self.timestamp.isoformat(),
            'processing_steps': self.processing_steps,
            'model_versions': self.model_versions,
            'has_predictions': self.has_predictions,
            'prediction_count': self.prediction_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionBundle':
        """Create instance from dictionary."""
        # Parse predictions
        segmentation = None
        if data.get('segmentation'):
            segmentation = SegmentationPrediction.from_dict(data['segmentation'])
        
        pose = None
        if data.get('pose'):
            pose = PosePrediction.from_dict(data['pose'])
        
        # Parse timestamp
        timestamp = datetime.fromisoformat(data['timestamp'])
        
        return cls(
            segmentation=segmentation,
            pose=pose,
            image_path=data.get('image_path'),
            image_hash=data.get('image_hash'),
            timestamp=timestamp,
            processing_steps=data.get('processing_steps', []),
            model_versions=data.get('model_versions', {})
        )
    
    def save_to_file(self, filepath: str, format: str = 'json'):
        """Save predictions to file."""
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif format.lower() == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")
    
    @classmethod
    def load_from_file(cls, filepath: str, format: str = 'json') -> 'PredictionBundle':
        """Load predictions from file."""
        if format.lower() == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        elif format.lower() == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"PredictionBundle(predictions={self.prediction_count}, "
                f"has_segmentation={self.segmentation is not None}, "
                f"has_pose={self.pose is not None}, "
                f"image_hash={self.image_hash[:8] if self.image_hash else 'None'})")


@dataclass
class PredictionMetadata:
    """Metadata for prediction processing and caching."""
    
    # Identification
    prediction_id: str
    source_image_path: str
    
    # Processing info
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # Model info
    segmentation_model_version: Optional[str] = None
    pose_model_version: Optional[str] = None
    
    # Cache info
    cache_size_bytes: int = 0
    is_persistent: bool = False
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def get_age_seconds(self) -> float:
        """Get age of prediction in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def get_access_frequency(self) -> float:
        """Get access frequency (accesses per second)."""
        age = self.get_age_seconds()
        return self.access_count / age if age > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'prediction_id': self.prediction_id,
            'source_image_path': self.source_image_path,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'segmentation_model_version': self.segmentation_model_version,
            'pose_model_version': self.pose_model_version,
            'cache_size_bytes': self.cache_size_bytes,
            'is_persistent': self.is_persistent,
            'age_seconds': self.get_age_seconds(),
            'access_frequency': self.get_access_frequency()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionMetadata':
        """Create instance from dictionary."""
        # Parse timestamps
        created_at = datetime.fromisoformat(data['created_at'])
        last_accessed = datetime.fromisoformat(data['last_accessed'])
        
        return cls(
            prediction_id=data['prediction_id'],
            source_image_path=data['source_image_path'],
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=data['access_count'],
            segmentation_model_version=data.get('segmentation_model_version'),
            pose_model_version=data.get('pose_model_version'),
            cache_size_bytes=data.get('cache_size_bytes', 0),
            is_persistent=data.get('is_persistent', False)
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"PredictionMetadata(id={self.prediction_id[:8]}, "
                f"accesses={self.access_count}, "
                f"age={self.get_age_seconds():.1f}s, "
                f"size={self.cache_size_bytes} bytes)")
