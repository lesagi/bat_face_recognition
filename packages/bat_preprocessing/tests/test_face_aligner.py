"""Tests for the face aligner."""

from __future__ import annotations

import numpy as np
from bat_preprocessing.face_aligner import AlignedFace, FaceAligner
from bat_preprocessing.prediction_structures import PosePrediction, SegmentationPrediction


def _synthetic_image(size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # white square in centre
    img[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
    return img


def _synthetic_seg(size: int = 64) -> SegmentationPrediction:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1
    return SegmentationPrediction(
        mask=mask,
        confidence=0.9,
        bounding_box=(0.25, 0.25, 0.75, 0.75),
        original_image_shape=(size, size),
    )


def _synthetic_pose(size: int = 64) -> PosePrediction:
    cx = size // 2
    cy = size // 2
    kpts = np.array(
        [
            [cx - 8, cy - 4, 0.95],
            [cx + 8, cy - 4, 0.95],
            [cx, cy + 4, 0.85],
        ],
        dtype=np.float32,
    )
    return PosePrediction(
        keypoints=kpts,
        bounding_box=(0.25, 0.25, 0.75, 0.75),
        confidence=0.9,
        original_image_shape=(size, size),
    )


def test_face_aligner_output_shape_with_pose() -> None:
    aligner = FaceAligner(edge_length=128, margin_ratio=0.25)
    img = _synthetic_image(64)
    seg = _synthetic_seg(64)
    pose = _synthetic_pose(64)
    out = aligner.align(img, seg, pose)
    assert out is not None
    assert out.shape == (128, 128, 3)
    assert out.dtype == img.dtype


def test_face_aligner_output_shape_without_pose() -> None:
    aligner = FaceAligner(edge_length=64, margin_ratio=0.0)
    img = _synthetic_image(48)
    seg = _synthetic_seg(48)
    out = aligner.align(img, seg, None)
    assert out is not None
    assert out.shape == (64, 64, 3)


def test_face_aligner_handles_empty_mask() -> None:
    aligner = FaceAligner(edge_length=32)
    img = _synthetic_image(32)
    empty_mask = np.zeros((32, 32), dtype=np.uint8)
    # pre-bake the prediction with at least one pixel so __post_init__ accepts it,
    # then zero it out before passing it into align()
    seg = SegmentationPrediction(
        mask=empty_mask.copy(),
        confidence=0.5,
        bounding_box=(0.0, 0.0, 1.0, 1.0),
        original_image_shape=(32, 32),
    )
    out = aligner.align(img, seg, None)
    assert out is None


def test_face_aligner_validates_constructor() -> None:
    try:
        FaceAligner(edge_length=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for edge_length=0")

    try:
        FaceAligner(margin_ratio=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative margin_ratio")

    try:
        FaceAligner(mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid mode")


# ---- eye-anchored mode -------------------------------------------------


def test_eye_anchored_returns_aligned_face() -> None:
    aligner = FaceAligner(edge_length=128, margin_ratio=0.03, mode="eye_anchored")
    img = _synthetic_image(64)
    seg = _synthetic_seg(64)
    pose = _synthetic_pose(64)
    result = aligner.align_eye_anchored(img, seg, pose)
    assert isinstance(result, AlignedFace)
    assert result.image.shape == (128, 128, 3)
    assert result.keypoints.shape == (3, 3)
    assert result.mask.shape == (128, 128)
    # align() in eye_anchored mode returns just the crop
    assert aligner.align(img, seg, pose).shape == (128, 128, 3)


def test_eye_anchored_centers_the_mask_not_eyes() -> None:
    edge = 128
    aligner = FaceAligner(edge_length=edge, margin_ratio=0.03, mode="eye_anchored")
    img = _synthetic_image(64)
    seg = _synthetic_seg(64)
    pose = _synthetic_pose(64)
    result = aligner.align_eye_anchored(img, seg, pose)
    # The crop is centred on the segmentation MASK (head), not the eyes:
    # the warped mask's centroid lands at the image centre.
    ys, xs = np.where(result.mask > 0)
    assert xs.size > 0
    assert abs(float(xs.mean()) - edge / 2) <= 4.0
    assert abs(float(ys.mean()) - edge / 2) <= 4.0


def test_eye_anchored_makes_eye_line_horizontal() -> None:
    edge = 128
    aligner = FaceAligner(edge_length=edge, margin_ratio=0.1, mode="eye_anchored")
    img = _synthetic_image(80)
    seg = _synthetic_seg(80)
    # Deliberately tilted eye line (left eye higher than right).
    cx, cy = 40, 40
    kpts = np.array(
        [[cx - 10, cy - 6, 0.95], [cx + 10, cy + 6, 0.95], [cx, cy + 10, 0.8]],
        dtype=np.float32,
    )
    pose = PosePrediction(
        keypoints=kpts,
        bounding_box=(0.25, 0.25, 0.75, 0.75),
        confidence=0.9,
        original_image_shape=(80, 80),
    )
    result = aligner.align_eye_anchored(img, seg, pose)
    left_eye, right_eye = result.keypoints[0], result.keypoints[1]
    # After alignment the two eyes share (approximately) the same row.
    assert abs(float(left_eye[1]) - float(right_eye[1])) <= 1.5


def test_eye_anchored_empty_mask_returns_none() -> None:
    aligner = FaceAligner(edge_length=64, mode="eye_anchored")
    img = _synthetic_image(32)
    seg = SegmentationPrediction(
        mask=np.zeros((32, 32), dtype=np.uint8),
        confidence=0.5,
        bounding_box=(0.0, 0.0, 1.0, 1.0),
        original_image_shape=(32, 32),
    )
    assert aligner.align_eye_anchored(img, seg, _synthetic_pose(32)) is None


# ---- elliptical face mask ----------------------------------------------


def test_elliptical_face_mask_includes_landmarks() -> None:
    aligner = FaceAligner(edge_length=128, mode="eye_anchored")
    edge = 128
    mask = np.zeros((edge, edge), dtype=np.uint8)
    mask[24:104, 24:104] = 1  # central square "head"
    kpts = np.array(
        [[52, 56, 0.95], [76, 56, 0.95], [64, 76, 0.85]],
        dtype=np.float32,
    )
    ellipse = aligner.elliptical_face_mask(mask, kpts, edge)
    assert ellipse.shape == (edge, edge)
    assert ellipse.sum() > 0
    # Every valid landmark must be inside the ellipse.
    for x, y, _ in kpts:
        assert ellipse[int(round(y)), int(round(x))] == 1
    # Corners should be outside (head silhouette / corners removed).
    assert ellipse[0, 0] == 0 and ellipse[-1, -1] == 0


def test_elliptical_face_mask_empty_mask_uses_landmarks() -> None:
    aligner = FaceAligner(edge_length=64, mode="eye_anchored")
    edge = 64
    mask = np.zeros((edge, edge), dtype=np.uint8)
    kpts = np.array(
        [[26, 30, 0.95], [38, 30, 0.95], [32, 40, 0.85]],
        dtype=np.float32,
    )
    ellipse = aligner.elliptical_face_mask(mask, kpts, edge)
    assert ellipse.shape == (edge, edge)
    assert ellipse.sum() > 0


# ---- pose-confidence gating + orientation guard ------------------------


def _pose_with(kpts: np.ndarray, size: int = 64) -> PosePrediction:
    return PosePrediction(
        keypoints=kpts.astype(np.float32),
        bounding_box=(0.1, 0.1, 0.9, 0.9),
        confidence=0.9,
        original_image_shape=(size, size),
    )


def test_low_confidence_eyes_skip_frame() -> None:
    # Both eyes below threshold → require_confident_eyes drops the frame
    # (this is the fix for the garbage-angle 90° rotations).
    aligner = FaceAligner(
        edge_length=64,
        margin_ratio=0.03,
        mode="eye_anchored",
        min_keypoint_confidence=0.5,
        require_confident_eyes=True,
    )
    img = _synthetic_image(64)
    seg = _synthetic_seg(64)
    low = _pose_with(np.array([[24, 30, 0.02], [40, 30, 0.03], [32, 40, 0.9]]))
    assert aligner.align_eye_anchored(img, seg, low) is None
    # Confident eyes → frame is kept.
    good = _pose_with(np.array([[24, 30, 0.9], [40, 30, 0.9], [32, 40, 0.9]]))
    assert aligner.align_eye_anchored(img, seg, good) is not None


def test_low_confidence_eyes_fallback_when_not_required() -> None:
    # Without require_confident_eyes, weak eyes fall back to a no-rotation
    # mask-centroid crop rather than being dropped.
    aligner = FaceAligner(
        edge_length=64,
        margin_ratio=0.1,
        mode="eye_anchored",
        min_keypoint_confidence=0.5,
        require_confident_eyes=False,
    )
    weak = _pose_with(np.array([[24, 30, 0.01], [40, 30, 0.01], [32, 40, 0.01]]))
    out = aligner.align_eye_anchored(_synthetic_image(64), _synthetic_seg(64), weak)
    assert out is not None and out.image.shape == (64, 64, 3)


def test_nose_roll_weight_pulls_nose_below_eye_midpoint() -> None:
    # Eyes horizontal but the muzzle leans sideways (nose left of the eye
    # midpoint). With nose_roll_weight=1 the rotation must land the nose
    # directly below the eye midpoint.
    aligner = FaceAligner(
        edge_length=96,
        margin_ratio=0.2,
        mode="eye_anchored",
        min_keypoint_confidence=0.4,
        require_confident_eyes=True,
        nose_roll_weight=1.0,
    )
    img = _synthetic_image(64)
    seg = _synthetic_seg(64)
    leaning = _pose_with(np.array([[24, 30, 0.9], [40, 30, 0.9], [26, 42, 0.9]]))
    out = aligner.align_eye_anchored(img, seg, leaning)
    assert out is not None
    eye_mid = out.keypoints[:2, :2].mean(axis=0)
    nose = out.keypoints[2, :2]
    assert nose[1] > eye_mid[1]  # still upright
    assert abs(float(nose[0]) - float(eye_mid[0])) <= 1.0  # nose axis vertical


def test_nose_orientation_guard_flips_upside_down() -> None:
    # Eyes horizontal but nose ABOVE them (face upside down). The guard should
    # add 180° so the nose ends up below the eye-midpoint in the output.
    aligner = FaceAligner(
        edge_length=96,
        margin_ratio=0.2,
        mode="eye_anchored",
        min_keypoint_confidence=0.4,
        require_confident_eyes=True,
    )
    img = _synthetic_image(64)
    seg = _synthetic_seg(64)
    upside_down = _pose_with(np.array([[24, 34, 0.9], [40, 34, 0.9], [32, 16, 0.9]]))
    out = aligner.align_eye_anchored(img, seg, upside_down)
    assert out is not None
    eye_mid_y = out.keypoints[:2, 1].mean()
    nose_y = out.keypoints[2, 1]
    assert nose_y > eye_mid_y  # nose now below the eyes
