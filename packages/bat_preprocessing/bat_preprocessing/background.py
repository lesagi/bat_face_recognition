"""Background generation and replacement.

Combines functionality from ``app/background_generation/`` and
``app/background_replacement/``. Pure PIL / numpy / OpenCV — no framework
dependencies. Public function names retained for parity with the original
implementations.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Background generation utilities
# ---------------------------------------------------------------------------


def create_blur_image(height: int, width: int) -> np.ndarray:
    """Create a Gaussian-blurred random-pixel image."""
    random_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    kernel_size = random.choice([k for k in range(3, 30) if k % 2 == 1])
    return cv2.GaussianBlur(random_image, (kernel_size, kernel_size), 0)


def create_green_image(height: int, width: int) -> np.ndarray:
    """Create a uniform green BGR image (the legacy chroma-key colour)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (0, 255, 0)
    return img


def _generate_fallback_background(height: int, width: int) -> np.ndarray:
    """Fallback gradient when remote image fetch fails."""
    start_color = tuple(random.randint(50, 150) for _ in range(3))
    end_color = tuple(random.randint(150, 255) for _ in range(3))
    background = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        ratio = i / max(height, 1)
        background[i, :] = tuple(
            int(start_color[j] * (1 - ratio) + end_color[j] * ratio) for j in range(3)
        )
    return background


def get_random_cropped_image(
    height: int,
    width: int,
    max_retries: int = 3,
    timeout: float = 30,
    fallback_to_generated: bool = True,
) -> np.ndarray:
    """Fetch and centre-crop a random image from Picsum Photos.

    On failure (after ``max_retries`` attempts), either raise or fall back to
    a generated gradient depending on ``fallback_to_generated``.
    """
    try:
        import requests
        from requests.exceptions import ConnectionError, RequestException, Timeout
    except ImportError as exc:  # pragma: no cover
        if fallback_to_generated:
            return _generate_fallback_background(height, width)
        raise ImportError("requests is required to fetch Picsum images") from exc

    min_dim = max(height, width)
    url = f"https://picsum.photos/{min_dim}/{min_dim}"
    last_exc: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                img_w, img_h = image.size
                left = (img_w - width) // 2
                top = (img_h - height) // 2
                cropped = image.crop((left, top, left + width, top + height))
                return cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
            raise RuntimeError(f"HTTP {response.status_code}: failed to fetch image from Picsum")
        except (RequestException, ConnectionError, Timeout) as exc:
            last_exc = exc
        except Exception as exc:
            last_exc = exc
        if attempt < max_retries:
            time.sleep(2**attempt)

    if fallback_to_generated:
        return _generate_fallback_background(height, width)
    raise RuntimeError(
        f"Failed to fetch image from Picsum after {max_retries + 1} attempts; last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# BackgroundGenerator factory (parity with app/background_generation)
# ---------------------------------------------------------------------------


class BackgroundGenerator:
    """Factory class producing different kinds of background images."""

    @staticmethod
    def solid_color(
        height: int, width: int, color: tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        background = np.zeros((height, width, 3), dtype=np.uint8)
        background[:] = color
        return background

    @staticmethod
    def noise(height: int, width: int, intensity: int = 255) -> np.ndarray:
        return np.random.randint(0, intensity + 1, (height, width, 3), dtype=np.uint8)

    @staticmethod
    def blur(height: int, width: int) -> np.ndarray:
        return create_blur_image(height, width)

    @staticmethod
    def picsum(
        height: int,
        width: int,
        max_retries: int = 3,
        timeout: float = 30,
        fallback_to_generated: bool = True,
    ) -> np.ndarray:
        return get_random_cropped_image(height, width, max_retries, timeout, fallback_to_generated)

    @staticmethod
    def gradient(
        height: int,
        width: int,
        start_color: tuple[int, int, int] = (0, 0, 0),
        end_color: tuple[int, int, int] = (255, 255, 255),
        direction: str = "vertical",
    ) -> np.ndarray:
        background = np.zeros((height, width, 3), dtype=np.uint8)
        if direction == "vertical":
            for i in range(height):
                ratio = i / max(height, 1)
                background[i, :] = [
                    int(start_color[j] + (end_color[j] - start_color[j]) * ratio) for j in range(3)
                ]
        elif direction == "horizontal":
            for i in range(width):
                ratio = i / max(width, 1)
                background[:, i] = [
                    int(start_color[j] + (end_color[j] - start_color[j]) * ratio) for j in range(3)
                ]
        elif direction == "diagonal":
            for i in range(height):
                for j in range(width):
                    ratio = (i + j) / max(height + width, 1)
                    background[i, j] = [
                        int(start_color[k] + (end_color[k] - start_color[k]) * ratio)
                        for k in range(3)
                    ]
        else:
            raise ValueError(f"Unknown gradient direction: {direction}")
        return background


BACKGROUND_GENERATORS = {
    "blur": BackgroundGenerator.blur,
    "noise": BackgroundGenerator.noise,
    "gradient": BackgroundGenerator.gradient,
    "picsum": BackgroundGenerator.picsum,
    "solid_color": BackgroundGenerator.solid_color,
}


def get_background_generator(name: str) -> Callable[..., np.ndarray]:
    """Look up a background generator by name."""
    if name not in BACKGROUND_GENERATORS:
        raise KeyError(
            f"Unknown background generator: {name!r}. Available: {sorted(BACKGROUND_GENERATORS)}"
        )
    return BACKGROUND_GENERATORS[name]


# ---------------------------------------------------------------------------
# Background replacement
# ---------------------------------------------------------------------------


def _normalise_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Coerce ``mask`` to a uint8 0/1 array matching ``target_shape``."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    if mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)
    return mask


def _soften_mask(binary: np.ndarray, smooth: int, feather: int) -> np.ndarray:
    """0/1 uint8 mask → float32 alpha in [0, 1].

    ``smooth`` rounds the contour itself (morphological close, then Gaussian
    blur + re-threshold) so jagged seg-mask corners disappear; ``feather``
    then blurs the edge into a soft alpha ramp instead of a hard binary cut.
    Both are pixel radii.
    """
    alpha = binary.astype(np.float32)
    if smooth > 0:
        k = 2 * smooth + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        blurred = cv2.GaussianBlur(closed.astype(np.float32), (k, k), 0)
        alpha = (blurred >= 0.5).astype(np.float32)
    if feather > 0:
        k = 2 * feather + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return alpha


def replace_background(
    image: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray | Callable[..., np.ndarray],
    *,
    smooth: int = 0,
    feather: int = 0,
    **generator_kwargs,
) -> np.ndarray:
    """Replace the masked background of ``image``.

    Args:
        image: Foreground image, BGR ``(H, W, 3)``.
        mask: 2-D mask where the bat (foreground) is non-zero.
        background: Either a precomputed background image or a generator
            callable accepting ``(height, width, **kwargs)``.
        smooth: Contour-rounding radius in px (0 = keep the raw mask shape).
        feather: Edge-feathering radius in px (0 = hard binary cut).
        **generator_kwargs: Forwarded to the generator callable.

    Returns:
        The composited image.
    """
    if image is None or image.ndim != 3:
        raise ValueError("image must be a 3-channel array")

    height, width = image.shape[:2]
    binary_mask = _normalise_mask(mask, (height, width))

    if callable(background):
        bg = background(height, width, **generator_kwargs)
    else:
        bg = background
        if bg.shape[:2] != (height, width):
            bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_LINEAR)

    if bg.shape != image.shape:
        raise ValueError(f"background shape {bg.shape} does not match image shape {image.shape}")

    if smooth > 0 or feather > 0:
        alpha = _soften_mask(binary_mask, smooth, feather)[:, :, np.newaxis]
        blended = image.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)
        return np.clip(blended.round(), 0, 255).astype(np.uint8)

    mask_3d = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
    result = bg.copy()
    foreground = mask_3d > 0
    result[foreground] = image[foreground]
    return result


def replace_green_background(
    image: np.ndarray,
    background: np.ndarray | Callable[..., np.ndarray] | None = None,
) -> np.ndarray:
    """Replace a green-screen (chroma-key) background.

    Mirrors ``app/image_processor/image_utils.py:replace_green_background``.
    If ``background`` is omitted, a Picsum image is fetched.
    """
    if image is None or image.ndim != 3:
        raise ValueError("image must be a 3-channel array")
    height, width = image.shape[:2]

    if background is None:
        background_arr = get_random_cropped_image(height, width)
    elif callable(background):
        background_arr = background(height, width)
    else:
        background_arr = background
        if background_arr.shape[:2] != (height, width):
            background_arr = cv2.resize(
                background_arr, (width, height), interpolation=cv2.INTER_LINEAR
            )

    lower_green = np.array([0, 230, 0], dtype=np.uint8)
    upper_green = np.array([30, 255, 30], dtype=np.uint8)
    green_mask = cv2.inRange(image, lower_green, upper_green)
    green_mask = cv2.medianBlur(green_mask, 5)
    object_mask = cv2.bitwise_not(green_mask)

    object_foreground = cv2.bitwise_and(image, image, mask=object_mask)
    background_region = cv2.bitwise_and(background_arr, background_arr, mask=green_mask)
    return cv2.add(object_foreground, background_region)


def replace_background_with_alpha(
    image: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray | Callable[..., np.ndarray],
    **generator_kwargs,
) -> np.ndarray:
    """Background replacement that preserves an alpha channel.

    If ``image`` has an alpha channel (4 channels), the alpha values are kept
    intact in the output.
    """
    if image is None or image.ndim != 3:
        raise ValueError("image must be a 3-channel or 4-channel array")

    if image.shape[2] == 4:
        rgb = image[..., :3]
        alpha = image[..., 3:]
        composited = replace_background(rgb, mask, background, **generator_kwargs)
        return np.concatenate([composited, alpha], axis=2)

    return replace_background(image, mask, background, **generator_kwargs)
