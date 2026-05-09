"""Saliency page: 5 identities x 2 cols (original | saliency).

Uses :func:`bat_interpretability.composite.composite_from_saliency_images` for
the layout, then writes the PIL image as a PNG and embeds it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bat_interpretability.composite import composite_from_saliency_images
from bat_reporting.sections.common import (
    SectionContext,
    add_heading,
    add_image,
    add_placeholder,
)
from bat_stats.naming import build_filename, build_title_suffix

if TYPE_CHECKING:  # pragma: no cover
    from bat_reporting.data import ReportData

DEFAULT_ROWS = 5


def render_section(story: list[Any], data: ReportData, ctx: SectionContext) -> None:
    from reportlab.platypus import PageBreak

    add_heading(story, "Saliency: 5 identities", ctx, level=1)

    items = list(data.saliency_images or [])
    # Filter projection placeholders (method == 'tsne' / 'umap') -- those land
    # on a different page.
    items = [item for item in items if item.method not in {"tsne", "umap"}]

    if len(items) < DEFAULT_ROWS:
        add_placeholder(
            story,
            (
                f"Need {DEFAULT_ROWS} saliency samples for the standard layout; "
                f"got {len(items)}. Skipping the saliency page."
            ),
            ctx,
        )
        story.append(PageBreak())
        return

    title = f"Saliency overlays\n{build_title_suffix(data.cfg)}"
    composite = composite_from_saliency_images(items, rows=DEFAULT_ROWS, title=title)

    png_path = ctx.output_dir / build_filename("saliency_composite", data.cfg, suffix=".png")
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    composite.save(str(png_path))
    # The 5x2 composite is portrait-tall; cap its height so it always fits
    # within an A4 frame after the page heading.
    add_image(story, png_path, ctx, max_height=620.0)
    story.append(PageBreak())


__all__ = ["render_section"]
