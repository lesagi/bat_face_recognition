"""Video → still image extraction for bat-face training data.

Reads a video frame-by-frame, runs each frame through the YOLO segmenter and
(optionally) the pose estimator + face aligner, then writes per-frame still
images to disk and returns a list of :class:`bat_core.ImageRecord` objects
suitable for ``bat_data.build_manifest`` to consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from bat_core import ImageRecord
from bat_core.types import Background, Source, Species, Split

from .face_aligner import FaceAligner
from .yolo_pose import YOLOPoseEstimator
from .yolo_segmenter import YOLOSegmenter


def _laplacian_quality(image: np.ndarray) -> float:
    """Estimate sharpness via the variance of the Laplacian.

    Quality value is normalised to ``[0, 1]`` via a soft cap at 1000 so the
    output fits :class:`bat_core.ImageRecord`'s ``quality >= 0`` constraint.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(min(var / 1000.0, 1.0))


@dataclass
class VideoExtractionConfig:
    """Configuration for :class:`VideoExtractor`."""

    output_dir: Path
    """Directory where extracted frames will be written."""

    identity: str
    """Identity label (the bat ID) for every extracted frame."""

    species: Species = "rousettus"
    background: Background = "original"
    split: Split = "train"
    source: Source = "video"
    augmented: bool = False

    frame_stride: int = 1
    """Process every Nth frame."""

    max_frames: Optional[int] = None
    """Stop after this many successfully extracted frames."""

    min_quality: float = 0.0
    """Reject frames whose Laplacian quality is below this threshold."""

    edge_length: int = 224
    margin_ratio: float = 0.5

    require_pose: bool = False
    """If True, drop frames where pose estimation fails."""

    filename_prefix: str = ""
    """Prefix prepended to each output file name."""

    segmenter_config: Dict[str, Any] = field(default_factory=dict)
    pose_config: Dict[str, Any] = field(default_factory=dict)


class VideoExtractor:
    """Extract aligned still images from a video using YOLO seg + pose."""

    def __init__(
        self,
        config: VideoExtractionConfig,
        *,
        segmenter: Optional[YOLOSegmenter] = None,
        pose_estimator: Optional[YOLOPoseEstimator] = None,
        aligner: Optional[FaceAligner] = None,
    ) -> None:
        self.config = config
        self._segmenter = segmenter
        self._pose = pose_estimator
        self._aligner = aligner or FaceAligner(
            edge_length=config.edge_length, margin_ratio=config.margin_ratio
        )

    @property
    def segmenter(self) -> YOLOSegmenter:
        if self._segmenter is None:
            self._segmenter = YOLOSegmenter(self.config.segmenter_config)
        return self._segmenter

    @property
    def pose_estimator(self) -> YOLOPoseEstimator:
        if self._pose is None:
            self._pose = YOLOPoseEstimator(self.config.pose_config)
        return self._pose

    def extract(self, video_path: Union[str, Path]) -> List[ImageRecord]:
        """Process ``video_path`` and write extracted frames to disk.

        Returns:
            A list of :class:`bat_core.ImageRecord` for every successfully
            written frame.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")

        records: List[ImageRecord] = []
        frame_idx = 0
        kept = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame_idx % max(self.config.frame_stride, 1) != 0:
                    frame_idx += 1
                    continue

                record = self._process_frame(frame, frame_idx, video_path)
                frame_idx += 1
                if record is None:
                    continue

                records.append(record)
                kept += 1
                if (
                    self.config.max_frames is not None
                    and kept >= self.config.max_frames
                ):
                    break
        finally:
            cap.release()

        return records

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        video_path: Path,
    ) -> Optional[ImageRecord]:
        try:
            seg = self.segmenter.predict(frame)
        except Exception:
            return None
        if seg is None:
            return None

        pose = None
        try:
            pose = self.pose_estimator.predict(frame)
        except Exception:
            pose = None
        if self.config.require_pose and pose is None:
            return None

        aligned = self._aligner.align(frame, seg, pose)
        if aligned is None:
            return None

        quality = _laplacian_quality(aligned)
        if quality < self.config.min_quality:
            return None

        stem = video_path.stem
        prefix = self.config.filename_prefix or ""
        filename = f"{prefix}{stem}_frame{frame_idx:06d}.png"
        out_path = Path(self.config.output_dir) / filename
        if not cv2.imwrite(str(out_path), aligned):
            return None

        return ImageRecord(
            path=out_path,
            identity=self.config.identity,
            species=self.config.species,
            background=self.config.background,
            source=self.config.source,
            augmented=self.config.augmented,
            split=self.config.split,
            quality=quality,
        )


__all__ = [
    "VideoExtractionConfig",
    "VideoExtractor",
]
