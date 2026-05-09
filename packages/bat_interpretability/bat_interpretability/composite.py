"""5-identities-per-page composite layout.

The plan calls for the saliency report page to show 5 identities per page,
laid out as a grid of (5 rows) x (2 columns) — original on the left,
saliency overlay on the right. This module produces that composite as a
single :class:`PIL.Image.Image`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from bat_core.types import SaliencyImage
    from PIL.Image import Image as PILImage


def make_composite(
    images: Sequence[Path | str],
    saliencies: Sequence[np.ndarray],
    identities: Sequence[str],
    *,
    rows: int = 5,
    tile_size: int = 256,
    overlay_alpha: float = 0.55,
    cmap_name: str = "hot",
    title: str | None = None,
) -> PILImage:
    """Lay out per-identity (image, saliency) pairs into a single composite.

    Parameters
    ----------
    images:
        Paths to the original images.
    saliencies:
        2-D numpy arrays (already normalised to ``[0, 1]``) — one per image.
    identities:
        Identity labels — one per image. Used as the row title.
    rows:
        Number of identities per page (default ``5``, per the plan — non-
        negotiable for the standard report). ``len(images)`` must match.
    tile_size:
        Edge length in pixels for each tile (default ``256``).
    overlay_alpha:
        Blend factor for the saliency overlay on top of the image (default
        ``0.55``).
    cmap_name:
        Matplotlib colormap used to colourise the saliency map.
    title:
        Optional caption rendered as a header strip across the composite.

    Returns
    -------
    PIL.Image
        Single RGB image of size ``(2 * tile_size, rows * tile_size)``
        (plus a small header if ``title`` is given).
    """
    if not (len(images) == len(saliencies) == len(identities)):
        raise ValueError("images, saliencies, identities must all have the same length")
    if len(images) != rows:
        raise ValueError(
            f"composite expects exactly {rows} samples; got {len(images)}. "
            "Pass `rows=N` if you need a different layout."
        )

    from PIL import Image, ImageDraw, ImageFont

    cell = tile_size
    width = 2 * cell
    header_h = 28 if title else 0
    height = header_h + rows * cell
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))

    if title:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:  # pragma: no cover
            font = None
        draw.rectangle([(0, 0), (width, header_h)], fill=(32, 32, 32))
        draw.text((8, 6), title, fill=(255, 255, 255), font=font)

    for row_idx, (img_path, saliency, identity) in enumerate(
        zip(images, saliencies, identities, strict=True)
    ):
        original = _load_resized(img_path, cell)
        overlay = _make_overlay(original, saliency, alpha=overlay_alpha, cmap_name=cmap_name)
        y = header_h + row_idx * cell
        canvas.paste(original, (0, y))
        canvas.paste(overlay, (cell, y))
        # Tag the row with the identity in the top-left corner of the
        # original tile.
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([(0, y), (min(72, cell // 2), y + 16)], fill=(0, 0, 0))
        draw.text((4, y + 2), str(identity), fill=(255, 255, 255))

    return canvas


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #
def _load_resized(path: Path | str, size: int) -> PILImage:
    from PIL import Image

    img = Image.open(str(path)).convert("RGB").resize((size, size), Image.BILINEAR)
    return img


def _make_overlay(
    original: PILImage,
    saliency: np.ndarray,
    *,
    alpha: float,
    cmap_name: str,
) -> PILImage:
    """Colourise ``saliency`` and alpha-blend over ``original``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    sal = np.asarray(saliency, dtype=np.float32)
    if sal.ndim != 2:
        raise ValueError(f"saliency must be 2-D; got shape {sal.shape}")
    sal = _resize_2d(sal, original.size)
    cmap = plt.get_cmap(cmap_name)
    coloured = (cmap(sal)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(coloured)
    return Image.blend(original, heat, alpha)


def _resize_2d(arr: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Bilinear resize a 2-D array to ``(W, H)`` (PIL ordering)."""
    from PIL import Image

    w, h = size_wh
    if arr.shape == (h, w):
        return arr.astype(np.float32)
    arr_norm = arr.astype(np.float32)
    lo, hi = float(arr_norm.min()), float(arr_norm.max())
    if hi > lo:
        arr_norm = (arr_norm - lo) / (hi - lo)
    img = Image.fromarray((arr_norm * 255.0).astype(np.uint8))
    img = img.resize((w, h), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32)


def composite_from_saliency_images(
    saliency_images: Iterable[SaliencyImage],
    *,
    rows: int = 5,
    tile_size: int = 256,
    title: str | None = None,
) -> PILImage:
    """Convenience wrapper: take :class:`SaliencyImage` objects directly.

    Picks the first ``rows`` items (caller is responsible for sampling).
    """
    items = list(saliency_images)[:rows]
    if len(items) != rows:
        raise ValueError(f"need at least {rows} SaliencyImage entries; got {len(items)}")
    images = [item.image_path for item in items]
    saliencies = [np.asarray(item.saliency) for item in items]
    identities = [item.identity for item in items]
    return make_composite(
        images,
        saliencies,
        identities,
        rows=rows,
        tile_size=tile_size,
        title=title,
    )
