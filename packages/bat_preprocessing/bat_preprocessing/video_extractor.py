"""Video → still image extraction for bat-face training data.

Reads a video frame-by-frame, runs each frame through the YOLO segmenter and
(optionally) the pose estimator + face aligner, then writes per-frame still
images to disk and returns a list of :class:`bat_core.ImageRecord` objects
suitable for ``bat_data.build_manifest`` to consume.

Two output styles:

* ``align_mode="mask_crop"`` (default) — the original single-image-per-frame
  behaviour: one aligned crop written with ``config.background``.
* ``align_mode="eye_anchored"`` — eye-centred alignment plus, from the *same*
  seg + pose + align pass, multiple background **variants** per kept frame
  (original / green / random / elliptical face mask). Models run once per
  frame regardless of how many variants are requested.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from bat_core import ImageRecord
from bat_core.types import Background, Source, Species, Split

from .background import (
    BackgroundGenerator,
    create_blur_image,
    get_background_generator,
    get_random_cropped_image,
    replace_background,
)
from .face_aligner import FaceAligner
from .yolo_detector import YOLODetector
from .yolo_pose import YOLOPoseEstimator
from .yolo_segmenter import YOLOSegmenter

# Variant name → ImageRecord ``background`` value. ``face_ellipse`` is a
# face-shape variant (head silhouette removed) carried on the "original"
# background label; it is distinguished by its own output folder / manifest.
_VARIANT_BACKGROUND: dict[str, Background] = {
    "original_bg": "original",
    "green_bg": "green",
    "random_bg": "random",
    "face_ellipse": "original",
}


def _laplacian_quality(image: np.ndarray) -> float:
    """Estimate sharpness via the variance of the Laplacian.

    Quality value is normalised to ``[0, 1]`` via a soft cap at 1000 so the
    output fits :class:`bat_core.ImageRecord`'s ``quality >= 0`` constraint.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(min(var / 1000.0, 1.0))


@dataclass
class VideoExtractionConfig:
    """Configuration for :class:`VideoExtractor`."""

    output_dir: Path
    """Directory where extracted frames will be written (mask_crop mode), or
    the root under which per-variant subfolders are created (eye_anchored mode
    when ``variant_output_dirs`` is not given)."""

    identity: str
    """Identity label (the bat ID) for every extracted frame."""

    species: Species = "rousettus"
    background: Background = "original"
    split: Split = "train"
    source: Source = "video"
    augmented: bool = False

    frame_stride: int = 1
    """Process every Nth frame."""

    max_frames: int | None = None
    """Stop after this many successfully extracted frames."""

    min_quality: float = 0.0
    """Reject frames whose Laplacian quality is below this threshold."""

    edge_length: int = 224
    margin_ratio: float = 0.5

    require_pose: bool = False
    """If True, drop frames where pose estimation fails."""

    filename_prefix: str = ""
    """Prefix prepended to each output file name (mask_crop mode)."""

    segmenter_config: dict[str, Any] = field(default_factory=dict)
    pose_config: dict[str, Any] = field(default_factory=dict)

    # ---- eye_anchored / multi-variant options --------------------------
    align_mode: str = "mask_crop"
    """``"mask_crop"`` (default) or ``"eye_anchored"`` (multi-variant)."""

    variants: list[str] = field(
        default_factory=lambda: ["original_bg", "green_bg", "random_bg", "face_ellipse"]
    )
    """Which variants to emit in eye_anchored mode."""

    variant_output_dirs: dict[str, Path] = field(default_factory=dict)
    """Per-variant output directory. Defaults to ``output_dir / <variant>``."""

    name_template: str = "{prefix}{video}_frame{frame:06d}.png"
    """Output filename template. Available fields: ``identity``, ``video``
    (video stem), ``frame`` (frame index), ``prefix``. The rousettus-style
    convention is ``"m--{identity}--{video}.{frame}.jpg"``."""

    output_ext: str = ".png"
    jpeg_quality: int = 95

    random_bg_style: str = "picsum"
    """Background generator for ``random_bg``: ``picsum`` (natural images,
    pooled) or any key in ``BACKGROUND_GENERATORS`` (e.g. ``blur``)."""

    random_bg_pool_size: int = 40
    """Number of natural images to pre-fetch once for ``random_bg`` (picsum)."""

    green_color: tuple[int, int, int] = (0, 255, 0)
    ellipse_fill_color: tuple[int, int, int] = (0, 0, 0)
    ellipse_expand: float = 1.05
    random_seed: int = 42

    # ---- pose-confidence gating (eye_anchored) -------------------------
    min_keypoint_confidence: float = 0.0
    """Eye keypoints below this confidence are treated as undetected."""

    require_confident_eyes: bool = False
    """Skip frames whose eyes are not confidently detected (avoids the
    garbage-angle rotations seen on near-zero-confidence keypoints)."""

    resegment_crop: bool = False
    """If True, re-segment the 224 crop for the variant mask; otherwise use the
    segmentation mask warped into crop space (cleaner, the default)."""

    apply_orientation: bool = True
    """Honour the video's rotation metadata (``CAP_PROP_ORIENTATION_META``).
    Phone/portrait clips store frames landscape + a 90° flag that OpenCV does
    NOT apply by default, so every frame would otherwise be processed sideways.
    """


class VideoExtractor:
    """Extract aligned still images from a video using YOLO seg + pose."""

    def __init__(
        self,
        config: VideoExtractionConfig,
        *,
        segmenter: YOLOSegmenter | None = None,
        pose_estimator: YOLOPoseEstimator | None = None,
        aligner: FaceAligner | None = None,
        detector: YOLODetector | None = None,
    ) -> None:
        self.config = config
        self._segmenter = segmenter
        self._pose = pose_estimator
        # Optional face detector used as a presence gate: a frame is processed
        # only if the detector finds a face. Segmentation/pose/alignment run on
        # the full frame (the seg model is cleaner at full-frame scale than on a
        # tight crop). No detector → every frame is processed.
        self._detector = detector
        self._aligner = aligner or FaceAligner(
            edge_length=config.edge_length,
            margin_ratio=config.margin_ratio,
            mode=config.align_mode,
            min_keypoint_confidence=config.min_keypoint_confidence,
            require_confident_eyes=config.require_confident_eyes,
        )
        self._rng = np.random.RandomState(config.random_seed)
        self._random_bg_pool: list[np.ndarray] | None = None

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

    def extract(self, video_path: str | Path) -> list[ImageRecord]:
        """Process ``video_path`` and write extracted frames to disk.

        Returns:
            A list of :class:`bat_core.ImageRecord` for every successfully
            written frame (one per variant in eye_anchored mode).
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")

        # Apply the rotation metadata ourselves (disable OpenCV's auto-rotate so
        # behaviour is identical across builds, then rotate each decoded frame).
        rot_code = None
        if self.config.apply_orientation:
            with contextlib.suppress(Exception):
                cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0.0)
            rot_code = self._orientation_code(cap)

        records: list[ImageRecord] = []
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
                if rot_code is not None:
                    frame = cv2.rotate(frame, rot_code)

                frame_records = self._process_frame(frame, frame_idx, video_path)
                frame_idx += 1
                if not frame_records:
                    continue

                records.extend(frame_records)
                kept += 1
                if self.config.max_frames is not None and kept >= self.config.max_frames:
                    break
        finally:
            cap.release()

        return records

    @staticmethod
    def _orientation_code(cap: Any) -> int | None:
        """cv2.rotate code matching the capture's ORIENTATION_META, or None.

        ``ORIENTATION_META`` is the clockwise rotation (degrees) needed to display
        the frame upright. 90 → rotate CW, 180 → 180, 270 → rotate CCW.
        """
        try:
            meta = int(round(float(cap.get(cv2.CAP_PROP_ORIENTATION_META)))) % 360
        except Exception:
            return None
        return {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }.get(meta)

    # ---- per-frame processing ------------------------------------------

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        video_path: Path,
    ) -> list[ImageRecord]:
        # Detector is a presence gate only (no detection → skip the frame). The
        # segmenter, pose and aligner all run on the FULL frame: the seg model
        # produces clean head masks at full-frame scale (YOLO letterboxes to a
        # square) but over-segments / misses on a tight detection crop.
        if self._detector is not None:
            try:
                if self._detector.predict(frame) is None:
                    return []
            except Exception:
                return []

        try:
            seg = self.segmenter.predict(frame)
        except Exception:
            return []
        if seg is None:
            return []

        pose = None
        try:
            pose = self.pose_estimator.predict(frame)
        except Exception:
            pose = None
        if self.config.require_pose and pose is None:
            return []

        if self.config.align_mode == "eye_anchored":
            return self._process_frame_multivariant(frame, frame_idx, video_path, seg, pose)
        return self._process_frame_single(frame, frame_idx, video_path, seg, pose)

    def _process_frame_single(
        self,
        frame: np.ndarray,
        frame_idx: int,
        video_path: Path,
        seg: Any,
        pose: Any,
    ) -> list[ImageRecord]:
        aligned = self._aligner.align(frame, seg, pose)
        if aligned is None:
            return []
        quality = _laplacian_quality(aligned)
        if quality < self.config.min_quality:
            return []

        stem = video_path.stem
        prefix = self.config.filename_prefix or ""
        filename = f"{prefix}{stem}_frame{frame_idx:06d}.png"
        out_path = Path(self.config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / filename
        if not cv2.imwrite(str(out_path), aligned):
            return []

        return [
            ImageRecord(
                path=out_path,
                identity=self.config.identity,
                species=self.config.species,
                background=self.config.background,
                source=self.config.source,
                augmented=self.config.augmented,
                split=self.config.split,
                quality=quality,
            )
        ]

    def _process_frame_multivariant(
        self,
        frame: np.ndarray,
        frame_idx: int,
        video_path: Path,
        seg: Any,
        pose: Any,
    ) -> list[ImageRecord]:
        aligned_face = self._aligner.align_eye_anchored(frame, seg, pose)
        if aligned_face is None:
            return []
        aligned = aligned_face.image
        quality = _laplacian_quality(aligned)
        if quality < self.config.min_quality:
            return []

        variants = self._build_variants(aligned, aligned_face.keypoints, aligned_face.mask)
        if not variants:
            return []

        filename = self._frame_filename(video_path, frame_idx)
        records: list[ImageRecord] = []
        for name, image in variants.items():
            out_dir = self._variant_dir(name)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / filename
            params = (
                [cv2.IMWRITE_JPEG_QUALITY, int(self.config.jpeg_quality)]
                if self.config.output_ext.lower() in (".jpg", ".jpeg")
                else []
            )
            if not cv2.imwrite(str(out_path), image, params):
                continue
            records.append(
                ImageRecord(
                    path=out_path,
                    identity=self.config.identity,
                    species=self.config.species,
                    background=_VARIANT_BACKGROUND.get(name, "original"),
                    source=self.config.source,
                    augmented=self.config.augmented,
                    split=self.config.split,
                    quality=quality,
                )
            )
        return records

    # ---- variant generation --------------------------------------------

    def _build_variants(
        self, aligned: np.ndarray, kpts_crop: np.ndarray, fallback_mask: np.ndarray
    ) -> dict[str, np.ndarray]:
        edge = aligned.shape[0]
        # Default: use the full-frame segmentation mask warped into crop space
        # (precise, from the high-res frame). Optionally re-segment the 224 crop.
        mask = fallback_mask
        if self.config.resegment_crop:
            try:
                seg2 = self.segmenter.predict(aligned)
                if seg2 is not None and int(seg2.mask.sum()) > 0:
                    mask = seg2.mask
            except Exception:
                mask = fallback_mask

        variants: dict[str, np.ndarray] = {}
        wanted = self.config.variants
        if "original_bg" in wanted:
            variants["original_bg"] = aligned
        if "green_bg" in wanted:
            variants["green_bg"] = replace_background(
                aligned, mask, BackgroundGenerator.solid_color, color=self.config.green_color
            )
        if "random_bg" in wanted:
            variants["random_bg"] = replace_background(aligned, mask, self._random_background(edge))
        if "face_ellipse" in wanted:
            ellipse = self._aligner.elliptical_face_mask(
                mask, kpts_crop, edge, expand=self.config.ellipse_expand
            )
            variants["face_ellipse"] = replace_background(
                aligned,
                ellipse,
                BackgroundGenerator.solid_color,
                color=self.config.ellipse_fill_color,
            )
        return variants

    def _random_background(self, edge: int) -> np.ndarray:
        """A natural-ish random background image of size ``(edge, edge, 3)``.

        For ``picsum`` we pre-fetch a pool once and sample from it (avoids a
        slow network round-trip per frame); on failure we fall back to a local
        blurred-noise background. Other styles defer to the generator factory.
        """
        if self.config.random_bg_style == "picsum":
            pool = self._ensure_random_bg_pool(edge)
            if pool:
                return pool[int(self._rng.randint(len(pool)))].copy()
            return create_blur_image(edge, edge)
        try:
            return get_background_generator(self.config.random_bg_style)(edge, edge)
        except Exception:
            return create_blur_image(edge, edge)

    def _ensure_random_bg_pool(self, edge: int) -> list[np.ndarray]:
        if self._random_bg_pool is not None:
            return self._random_bg_pool
        pool: list[np.ndarray] = []
        for _ in range(max(1, self.config.random_bg_pool_size)):
            try:
                img = get_random_cropped_image(edge, edge, fallback_to_generated=False)
            except Exception:
                break  # network unavailable — stop trying, use local fallback
            pool.append(img)
        self._random_bg_pool = pool
        return pool

    # ---- naming / paths ------------------------------------------------

    def _frame_filename(self, video_path: Path, frame_idx: int) -> str:
        name = self.config.name_template.format(
            prefix=self.config.filename_prefix or "",
            identity=self.config.identity,
            video=video_path.stem,
            frame=frame_idx,
        )
        # Allow templates that omit the extension.
        if not Path(name).suffix:
            name = f"{name}{self.config.output_ext}"
        return name

    def _variant_dir(self, variant: str) -> Path:
        if variant in self.config.variant_output_dirs:
            return Path(self.config.variant_output_dirs[variant])
        return Path(self.config.output_dir) / variant


__all__ = [
    "VideoExtractionConfig",
    "VideoExtractor",
]
