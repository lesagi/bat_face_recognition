"""bat_preprocessing — YOLO seg/pose + face alignment + background pipelines.

Public API:

* :class:`PreprocessingPipeline`, :class:`PreprocessingConfig`
* :class:`YOLOSegmenter`, :class:`YOLOPoseEstimator`
* :class:`FaceAligner`
* :class:`BackgroundGenerator`, :func:`replace_background`,
  :func:`replace_green_background`, :func:`replace_background_with_alpha`
* :class:`VideoExtractor`, :class:`VideoExtractionConfig`
* Prediction structures: :class:`SegmentationPrediction`,
  :class:`PosePrediction`, :class:`PredictionBundle`
* Cache: :class:`CacheManager`, :class:`CacheConfig`, :class:`PredictionCache`
"""

from .background import (
    BACKGROUND_GENERATORS,
    BackgroundGenerator,
    create_blur_image,
    create_green_image,
    get_background_generator,
    get_random_cropped_image,
    replace_background,
    replace_background_with_alpha,
    replace_green_background,
)
from .face_aligner import FaceAligner
from .input_processor import PreprocessingConfig, PreprocessingPipeline
from .prediction_cache import CacheConfig, CacheEntry, CacheManager, PredictionCache
from .prediction_structures import (
    PosePrediction,
    PredictionBundle,
    PredictionMetadata,
    SegmentationPrediction,
)
from .prediction_transforms import (
    BoundingBoxTransformer,
    CoordinateMapper,
    KeypointTransformer,
    MaskTransformer,
    PredictionTransformer,
    TransformationMatrix,
)
from .video_extractor import VideoExtractionConfig, VideoExtractor
from .yolo_pose import DEFAULT_POSE_CONFIG, YOLOPoseEstimator
from .yolo_segmenter import DEFAULT_SEGMENTER_CONFIG, YOLOSegmenter

__all__ = [
    # background
    "BACKGROUND_GENERATORS",
    "BackgroundGenerator",
    "create_blur_image",
    "create_green_image",
    "get_background_generator",
    "get_random_cropped_image",
    "replace_background",
    "replace_background_with_alpha",
    "replace_green_background",
    # face alignment
    "FaceAligner",
    # high-level pipeline
    "PreprocessingConfig",
    "PreprocessingPipeline",
    # caching
    "CacheConfig",
    "CacheEntry",
    "CacheManager",
    "PredictionCache",
    # structures
    "PosePrediction",
    "PredictionBundle",
    "PredictionMetadata",
    "SegmentationPrediction",
    # transforms
    "BoundingBoxTransformer",
    "CoordinateMapper",
    "KeypointTransformer",
    "MaskTransformer",
    "PredictionTransformer",
    "TransformationMatrix",
    # video extraction
    "VideoExtractionConfig",
    "VideoExtractor",
    # YOLO wrappers
    "DEFAULT_POSE_CONFIG",
    "DEFAULT_SEGMENTER_CONFIG",
    "YOLOPoseEstimator",
    "YOLOSegmenter",
]
