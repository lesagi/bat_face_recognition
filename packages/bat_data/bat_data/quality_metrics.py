"""Per-image quality battery for cross-species dataset comparison.

The manifest carries a single quality proxy — Laplacian variance
(:func:`bat_data.manifest.compute_quality`). Aggregated per identity it says the
two species' datasets differ strongly (median 571.7 vs 134.3 on green, Cliff's
delta ~0.8), but Laplacian variance conflates three different things: real
sharpness, effective resolution (an upscaled crop is smooth), and image content
(fur texture, contrast). This module separates them so a reviewer's "is there an
SNR-like measure showing the species are comparable?" can be answered per
dimension instead of with one confounded number.

Metric groups
-------------
sharpness     ``laplacian_var``, ``tenengrad``, ``gradient_energy``
noise         ``noise_sigma_immerkaer``, ``noise_sigma_mad``
signal/noise  ``snr``  (mean / std of ROI luminance)
contrast      ``rms_contrast``, ``michelson_contrast``
exposure      ``mean_luminance``
information   ``shannon_entropy``
colour        ``colorfulness``  (Hasler & Suesstrunk 2003)
no-reference  ``brisque``  (optional — requires ``piq``)

All metrics accept an optional boolean ROI mask. Gradient-based measures are
computed on the full image and *then* restricted to the ROI, so masking never
introduces a synthetic edge; the mask is eroded first (:data:`MASK_ERODE_PX`) so
the face/background boundary of the green-background variant cannot contaminate
a sharpness reading.

Effective resolution is deliberately *not* here: it must be measured on the
source frame before the aligner's warp, which needs segmentation. See
``scripts/measure_native_boxes.py``.

Deviation from the plan: BRISQUE is the only no-reference IQA implemented. NIQE
and PIQE are not available in ``piq`` and would each need their own pristine
model parameters; they were dropped rather than approximated. BRISQUE itself is
exploratory here — it is trained on human opinion scores of natural photographs
with synthetic distortions, so tight animal-face crops on a flat green
background are out of its training distribution. Treat the interpretable
sharpness / noise / contrast metrics as primary.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

__all__ = [
    "BATTERY_METRICS",
    "MASK_ERODE_PX",
    "brisque_score",
    "chromaticity_g",
    "chromaticity_r",
    "colorfulness",
    "compute_battery",
    "face_mask_from_green",
    "gradient_energy",
    "hue_mean_wb",
    "laplacian_var",
    "mean_luminance",
    "michelson_contrast",
    "noise_sigma_immerkaer",
    "noise_sigma_mad",
    "rms_contrast",
    "saturation_wb",
    "shannon_entropy",
    "snr",
    "tenengrad",
]

# Erosion applied to a supplied ROI mask before any pixel selection, in pixels.
# Two erosion steps of a 3x3 kernel remove the feathered mask boundary the
# background-replacement step leaves behind (``--feather 4`` at build time).
MASK_ERODE_PX = 4

# Immerkaer (1996) fast noise-variance kernel.
_IMMERKAER_KERNEL = np.array(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
    dtype=np.float64,
)

# Metrics computed by :func:`compute_battery`, in report order. ``brisque`` is
# appended only when requested and available.
BATTERY_METRICS: tuple[str, ...] = (
    "laplacian_var",
    "tenengrad",
    "gradient_energy",
    "noise_sigma_immerkaer",
    "noise_sigma_mad",
    "snr",
    "rms_contrast",
    "michelson_contrast",
    "mean_luminance",
    "shannon_entropy",
    "colorfulness",
    # Illumination-invariant colour: separates fur pigment from lighting.
    "chromaticity_r",
    "chromaticity_g",
    "hue_mean_wb",
    "saturation_wb",
)


# ---------------------------------------------------------------------------
# Loading / masking helpers
# ---------------------------------------------------------------------------


def _cv2() -> Any:
    import cv2  # local import: opencv-python is heavy

    return cv2


def load_bgr(image_path: str | os.PathLike[str]) -> np.ndarray | None:
    """Read an image as BGR ``uint8``, or ``None`` if unreadable."""
    return _cv2().imread(os.fspath(image_path), _cv2().IMREAD_COLOR)


def _gray(bgr: np.ndarray) -> np.ndarray:
    """Luminance as ``float64`` in [0, 255]."""
    if bgr.ndim == 2:
        return bgr.astype(np.float64)
    return _cv2().cvtColor(bgr, _cv2().COLOR_BGR2GRAY).astype(np.float64)


def face_mask_from_green(
    bgr: np.ndarray,
    *,
    tol: int = 40,
    min_area_frac: float = 0.01,
) -> np.ndarray | None:
    """Recover the face ROI from a green-background variant.

    The build paints the background pure green ``(0, 255, 0)``, so "not green"
    is the face. Returns a boolean mask, or ``None`` when the image is not a
    green-background variant (no plausible green region, or almost everything
    is green).
    """
    if bgr.ndim != 3:
        return None
    b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
    green = (g > 255 - tol) & (b < tol) & (r < tol)
    green_frac = float(green.mean())
    # A green-background crop is mostly-but-not-entirely green.
    if green_frac < min_area_frac or green_frac > 1.0 - min_area_frac:
        return None
    return ~green


def _prepare_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Erode and validate an ROI mask; ``None`` means "use every pixel"."""
    if mask is None:
        return None
    m = mask.astype(np.uint8)
    if m.shape[:2] != shape:
        m = _cv2().resize(m, (shape[1], shape[0]), interpolation=_cv2().INTER_NEAREST)
    if MASK_ERODE_PX > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        m = _cv2().erode(m, kernel, iterations=MASK_ERODE_PX)
    out = m > 0
    # An over-eroded mask would make every metric a single-pixel statistic.
    if int(out.sum()) < 64:
        return None
    return out


def _select(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Flatten *values*, keeping only ROI pixels when a mask is given."""
    if mask is None:
        return values.reshape(-1)
    return values[mask]


# ---------------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------------


def laplacian_var(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian — the manifest's existing focus measure."""
    lap = _cv2().Laplacian(gray, _cv2().CV_64F)
    return float(np.var(_select(lap, mask)))


def tenengrad(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean squared Sobel gradient magnitude (Tenenbaum focus measure)."""
    cv2 = _cv2()
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag2 = gx * gx + gy * gy
    return float(np.mean(_select(mag2, mask)))


def gradient_energy(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean squared first difference — a scale-sensitive sharpness proxy.

    Unlike :func:`tenengrad` this uses a 1-pixel difference, so it responds to
    the finest detail present and drops sharply on interpolated (upscaled)
    images.
    """
    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)
    dx[:, :-1] = np.diff(gray, axis=1)
    dy[:-1, :] = np.diff(gray, axis=0)
    return float(np.mean(_select(dx * dx + dy * dy, mask)))


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def noise_sigma_immerkaer(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Immerkaer (1996) fast noise standard-deviation estimate.

    Convolves with a 3x3 mask whose response to a locally-planar signal is
    zero, so the remaining energy is noise.
    """
    conv = _cv2().filter2D(gray, _cv2().CV_64F, _IMMERKAER_KERNEL)
    # Drop the 1-px border, where filter2D extrapolates.
    interior = conv[1:-1, 1:-1]
    interior_mask = None if mask is None else mask[1:-1, 1:-1]
    vals = _select(np.abs(interior), interior_mask)
    if vals.size == 0:
        return 0.0
    return float(np.sqrt(np.pi / 2.0) * vals.mean() / 6.0)


def noise_sigma_mad(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Robust noise sigma from the median absolute deviation of a high-pass residual.

    ``sigma = MAD / 0.6745`` is the standard consistent estimator for Gaussian
    noise; using a median makes it insensitive to genuine edges, which a plain
    residual variance would count as noise.
    """
    blurred = _cv2().GaussianBlur(gray, (0, 0), 1.0)
    residual = _select(gray - blurred, mask)
    if residual.size == 0:
        return 0.0
    mad = float(np.median(np.abs(residual - np.median(residual))))
    return mad / 0.6745


# ---------------------------------------------------------------------------
# Signal-to-noise, contrast, exposure
# ---------------------------------------------------------------------------


def snr(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Luminance SNR: mean / standard deviation over the ROI.

    The textbook image SNR. Note it rewards *flat* regions, so on its own it is
    not a quality measure for a textured subject — read it alongside
    :func:`noise_sigma_mad`, which isolates the noise term.
    """
    vals = _select(gray, mask)
    if vals.size == 0:
        return 0.0
    sd = float(vals.std())
    if sd <= 1e-12:
        return 0.0
    return float(vals.mean()) / sd


def rms_contrast(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """RMS contrast — standard deviation of luminance normalised to [0, 1]."""
    vals = _select(gray, mask)
    if vals.size == 0:
        return 0.0
    return float(vals.std() / 255.0)


def michelson_contrast(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Michelson contrast on the 5th/95th luminance percentiles.

    Percentiles rather than min/max: a single hot pixel would otherwise pin the
    result near 1.0 for every image.
    """
    vals = _select(gray, mask)
    if vals.size == 0:
        return 0.0
    lo, hi = np.percentile(vals, [5.0, 95.0])
    if hi + lo <= 1e-12:
        return 0.0
    return float((hi - lo) / (hi + lo))


def mean_luminance(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean luminance in [0, 1] — exposure level."""
    vals = _select(gray, mask)
    if vals.size == 0:
        return 0.0
    return float(vals.mean() / 255.0)


def shannon_entropy(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Shannon entropy (bits) of the 256-bin luminance histogram."""
    vals = _select(gray, mask)
    if vals.size == 0:
        return 0.0
    hist, _ = np.histogram(vals, bins=256, range=(0.0, 255.0))
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p /= total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def colorfulness(bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric.

    Strongly illumination-dependent: the same fur under warm daylight and dim
    indoor light scores very differently. Read it alongside the chromaticity and
    white-balanced hue measures below, which are built to separate pigment from
    lighting.
    """
    if bgr.ndim != 3:
        return 0.0
    b, g, r = (bgr[:, :, i].astype(np.float64) for i in range(3))
    rg = _select(r - g, mask)
    yb = _select(0.5 * (r + g) - b, mask)
    if rg.size == 0:
        return 0.0
    std_root = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
    mean_root = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return std_root + 0.3 * mean_root


# ---------------------------------------------------------------------------
# Illumination-invariant colour
# ---------------------------------------------------------------------------
#
# Colourfulness is the second-largest species difference of the whole battery
# (Hedges' g = +2.12), which raises a question it cannot answer on its own: is
# that the bats' fur, or the light it was filmed under? mauritius was shot
# outdoors in daylight, rousettus indoors under dim artificial light.
#
# The separation works because illumination scales all three channels roughly
# together. Dividing each channel by their sum cancels that common factor, so
# chromaticity survives a change of lamp while saturation does not. Grey-world
# white balance removes the residual colour *cast* before hue is measured.
#
# Pre-declared interpretation, so it cannot be chosen after seeing the answer:
# a species difference that SURVIVES these measures is pigmentation, a
# biological finding; one that COLLAPSES was lighting, a third confound.


def _white_balance(bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Grey-world correction: scale each channel so the ROI averages neutral."""
    out = bgr.astype(np.float64)
    means = np.array(
        [float(np.mean(_select(out[:, :, i], mask))) for i in range(3)], dtype=np.float64
    )
    if np.any(means <= 1e-6):
        return out
    return np.clip(out * (means.mean() / means), 0.0, 255.0)


def chromaticity_r(bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean red chromaticity ``r / (r + g + b)`` — largely illumination-invariant."""
    return _chromaticity(bgr, mask, channel=2)


def chromaticity_g(bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean green chromaticity ``g / (r + g + b)``."""
    return _chromaticity(bgr, mask, channel=1)


def _chromaticity(bgr: np.ndarray, mask: np.ndarray | None, *, channel: int) -> float:
    if bgr.ndim != 3:
        return 0.0
    arr = bgr.astype(np.float64)
    total = arr.sum(axis=2)
    total[total < 1e-6] = np.nan
    values = _select(arr[:, :, channel] / total, mask)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(values.mean())


def hue_mean_wb(bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Circular-mean hue in degrees, after grey-world white balance.

    Hue is measured circularly (0 and 360 are the same colour), so a plain mean
    would be wrong for anything spanning red.
    """
    if bgr.ndim != 3:
        return 0.0
    balanced = _white_balance(bgr, mask).astype(np.uint8)
    hsv = _cv2().cvtColor(balanced, _cv2().COLOR_BGR2HSV)
    # OpenCV packs hue into 0-179 for 8-bit images.
    hues = _select(hsv[:, :, 0].astype(np.float64), mask) * 2.0
    if hues.size == 0:
        return 0.0
    radians = np.radians(hues)
    return float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360.0)


def saturation_wb(bgr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean saturation after white balance, normalised to [0, 1]."""
    if bgr.ndim != 3:
        return 0.0
    balanced = _white_balance(bgr, mask).astype(np.uint8)
    hsv = _cv2().cvtColor(balanced, _cv2().COLOR_BGR2HSV)
    values = _select(hsv[:, :, 1].astype(np.float64), mask)
    if values.size == 0:
        return 0.0
    return float(values.mean() / 255.0)


# ---------------------------------------------------------------------------
# Optional no-reference IQA
# ---------------------------------------------------------------------------


def brisque_score(bgr: np.ndarray) -> float | None:
    """BRISQUE score via ``piq`` (lower is better), or ``None`` if unavailable.

    Exploratory only — see the module docstring on distribution mismatch.
    """
    try:
        import piq
        import torch
    except ImportError:
        return None

    rgb = bgr[:, :, ::-1].copy() if bgr.ndim == 3 else np.stack([bgr] * 3, axis=-1)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    try:
        return float(piq.brisque(tensor, data_range=1.0).item())
    except Exception:
        # piq raises on degenerate inputs (e.g. a uniform patch); a missing
        # score is more useful than a crashed batch.
        return None


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def compute_battery(
    image_path: str | os.PathLike[str],
    *,
    roi_from_green: bool = True,
    roi_mask: np.ndarray | None = None,
    include_nriqa: bool = False,
) -> dict[str, float] | None:
    """Compute every battery metric for one image.

    Args:
        image_path: Image to measure.
        roi_from_green: Derive the face ROI via :func:`face_mask_from_green`
            when *roi_mask* is not supplied. Silently falls back to the whole
            image for non-green variants.
        roi_mask: Explicit boolean ROI mask, overriding *roi_from_green*.
        include_nriqa: Also compute :func:`brisque_score`.

    Returns:
        Metric name -> value, plus ``roi_pixels`` (pixels actually measured)
        and ``roi_source`` (``"mask"`` / ``"green"`` / ``"full"``) so the
        analysis can tell face-only rows from whole-image rows. ``None`` if the
        image cannot be read.
    """
    bgr = load_bgr(image_path)
    if bgr is None:
        return None

    gray = _gray(bgr)
    shape = (gray.shape[0], gray.shape[1])

    if roi_mask is not None:
        raw_mask, roi_source = roi_mask, "mask"
    elif roi_from_green:
        raw_mask = face_mask_from_green(bgr)
        roi_source = "green" if raw_mask is not None else "full"
    else:
        raw_mask, roi_source = None, "full"

    mask = _prepare_mask(raw_mask, shape)
    if mask is None:
        roi_source = "full"

    out: dict[str, float] = {
        "laplacian_var": laplacian_var(gray, mask),
        "tenengrad": tenengrad(gray, mask),
        "gradient_energy": gradient_energy(gray, mask),
        "noise_sigma_immerkaer": noise_sigma_immerkaer(gray, mask),
        "noise_sigma_mad": noise_sigma_mad(gray, mask),
        "snr": snr(gray, mask),
        "rms_contrast": rms_contrast(gray, mask),
        "michelson_contrast": michelson_contrast(gray, mask),
        "mean_luminance": mean_luminance(gray, mask),
        "shannon_entropy": shannon_entropy(gray, mask),
        "colorfulness": colorfulness(bgr, mask),
        "chromaticity_r": chromaticity_r(bgr, mask),
        "chromaticity_g": chromaticity_g(bgr, mask),
        "hue_mean_wb": hue_mean_wb(bgr, mask),
        "saturation_wb": saturation_wb(bgr, mask),
    }
    out["roi_pixels"] = float(gray.size if mask is None else int(mask.sum()))
    out["roi_source"] = roi_source  # type: ignore[assignment]

    if include_nriqa:
        score = brisque_score(bgr)
        if score is not None:
            out["brisque"] = score

    return out
