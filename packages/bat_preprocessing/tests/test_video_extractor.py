"""Tests for the video extractor.

Uses synthetic frames written to disk via OpenCV; YOLO segmenter and pose
estimator are stubbed out so the test does not depend on weight files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv not installed")


from bat_preprocessing.face_aligner import FaceAligner  # noqa: E402
from bat_preprocessing.prediction_structures import (  # noqa: E402
    PosePrediction,
    SegmentationPrediction,
)
from bat_preprocessing.video_extractor import VideoExtractionConfig, VideoExtractor  # noqa: E402


class _StubSegmenter:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image: np.ndarray, **_: Any):
        self.calls += 1
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1
        return SegmentationPrediction(
            mask=mask,
            confidence=0.9,
            bounding_box=(0.25, 0.25, 0.75, 0.75),
            original_image_shape=(h, w),
        )


class _StubPose:
    def predict(self, image: np.ndarray, **_: Any):
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        kpts = np.array(
            [[cx - 6, cy, 0.95], [cx + 6, cy, 0.95], [cx, cy + 6, 0.85]],
            dtype=np.float32,
        )
        return PosePrediction(
            keypoints=kpts,
            bounding_box=(0.25, 0.25, 0.75, 0.75),
            confidence=0.9,
            original_image_shape=(h, w),
        )


def _write_synthetic_video(path: Path, frames: int = 5, size: int = 64) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (size, size))
    if not writer.isOpened():
        # Some CI environments lack the mp4v codec; fall back to MJPG/AVI.
        writer = cv2.VideoWriter(
            str(path.with_suffix(".avi")),
            cv2.VideoWriter_fourcc(*"MJPG"),
            5.0,
            (size, size),
        )
        path = path.with_suffix(".avi")
    if not writer.isOpened():
        pytest.skip("no usable video codec available in this environment")
    for i in range(frames):
        frame = np.full((size, size, 3), (10 + i * 20) % 256, dtype=np.uint8)
        frame[16:48, 16:48] = 255
        writer.write(frame)
    writer.release()


def test_video_extractor_emits_image_records(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    _write_synthetic_video(video, frames=4, size=64)
    # If the writer fell back to AVI, point to the new path
    if not video.exists():
        video = video.with_suffix(".avi")

    out_dir = tmp_path / "frames"
    cfg = VideoExtractionConfig(
        output_dir=out_dir,
        identity="X",
        species="rousettus",
        background="original",
        split="train",
        edge_length=48,
        margin_ratio=0.1,
        frame_stride=1,
        max_frames=2,
    )
    aligner = FaceAligner(edge_length=48, margin_ratio=0.1)
    extractor = VideoExtractor(
        cfg,
        segmenter=_StubSegmenter(),
        pose_estimator=_StubPose(),
        aligner=aligner,
    )
    records = extractor.extract(video)

    assert 1 <= len(records) <= 2, "should respect max_frames cap"
    for r in records:
        assert r.identity == "X"
        assert r.species == "rousettus"
        assert r.path.exists()
        assert r.path.suffix == ".png"
        assert 0.0 <= r.quality <= 1.0


def test_video_extractor_validates_missing_video(tmp_path: Path) -> None:
    cfg = VideoExtractionConfig(
        output_dir=tmp_path / "out",
        identity="Y",
    )
    extractor = VideoExtractor(cfg, segmenter=_StubSegmenter(), pose_estimator=_StubPose())
    try:
        extractor.extract(tmp_path / "no_such.mp4")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
