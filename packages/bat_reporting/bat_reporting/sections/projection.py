"""Embedding projection page: t-SNE + UMAP side-by-side."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bat_reporting.sections.common import SectionContext, add_heading, add_image, add_placeholder

if TYPE_CHECKING:  # pragma: no cover
    from bat_reporting.data import ReportData


def render_section(story: list[Any], data: ReportData, ctx: SectionContext) -> None:
    from reportlab.platypus import PageBreak, Spacer

    add_heading(story, "Embedding projection (t-SNE + UMAP)", ctx, level=1)

    proj = data.embedding_projection
    if proj is None or not proj.images:
        add_placeholder(
            story,
            "Embedding projections were not provided for this run.",
            ctx,
        )
        story.append(PageBreak())
        return

    half_width = ctx.page_width_pt / 2.0 - 6.0
    rendered_any = False
    if proj.tsne is not None:
        add_image(story, proj.tsne.image_path, ctx, width=half_width)
        rendered_any = True
        story.append(Spacer(1, 4))
    if proj.umap is not None:
        add_image(story, proj.umap.image_path, ctx, width=half_width)
        rendered_any = True

    if not rendered_any:
        # Render whatever projections we got even if neither tag was tsne/umap.
        for img in proj.images:
            add_image(story, img.image_path, ctx, width=half_width)
            story.append(Spacer(1, 4))

    story.append(PageBreak())


__all__ = ["render_section"]
