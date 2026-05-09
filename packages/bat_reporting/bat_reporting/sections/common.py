"""Shared context + helpers for PDF section renderers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    pass

# A4 page width minus margins, in points (used to scale embedded images).
DEFAULT_PAGE_USABLE_WIDTH_PT: float = 6.5 * 72.0  # 6.5" usable on letter / A4-ish.


@dataclass
class SectionContext:
    """State threaded through every section renderer."""

    output_dir: Path
    styles: Any  # reportlab.lib.styles.StyleSheet1 -- typed loosely for import-light callers
    page_width_pt: float = DEFAULT_PAGE_USABLE_WIDTH_PT


def add_heading(story: list[Any], text: str, ctx: SectionContext, level: int = 1) -> None:
    """Append a section heading flowable.

    ``level`` selects from ``Heading1`` / ``Heading2`` / ``Heading3``.
    """

    from reportlab.platypus import Paragraph, Spacer

    style_name = f"Heading{max(1, min(level, 3))}"
    style = ctx.styles[style_name]
    story.append(Paragraph(text, style))
    story.append(Spacer(1, 6))


def add_paragraph(story: list[Any], text: str, ctx: SectionContext) -> None:
    """Append a body paragraph (uses the ``BodyText`` style)."""

    from reportlab.platypus import Paragraph

    style = ctx.styles["BodyText"]
    story.append(Paragraph(text, style))


def add_image(
    story: list[Any],
    image_path: Path,
    ctx: SectionContext,
    *,
    width: float | None = None,
    max_height: float | None = None,
) -> None:
    """Append an image flowable, scaled to fit the usable page width."""

    from PIL import Image as PILImage
    from reportlab.platypus import Image, Spacer

    target_width = width if width is not None else ctx.page_width_pt
    try:
        with PILImage.open(str(image_path)) as im:
            iw, ih = im.size
    except Exception:
        iw, ih = (1, 1)
    scale = target_width / max(1, iw)
    target_height = ih * scale
    if max_height is not None and target_height > max_height:
        scale = max_height / max(1, ih)
        target_width = iw * scale
        target_height = max_height
    story.append(Image(str(image_path), width=target_width, height=target_height))
    story.append(Spacer(1, 8))


def add_placeholder(story: list[Any], message: str, ctx: SectionContext) -> None:
    """Append a styled "section unavailable" placeholder paragraph."""

    from reportlab.platypus import Paragraph, Spacer

    style = ctx.styles["Italic"] if "Italic" in ctx.styles.byName else ctx.styles["BodyText"]
    story.append(Paragraph(f"<i>{message}</i>", style))
    story.append(Spacer(1, 8))


def add_table(
    story: list[Any],
    rows: list[list[str]],
    ctx: SectionContext,
    *,
    col_widths: list[float] | None = None,
    header_row: bool = True,
) -> None:
    """Append a basic table with light grid styling."""

    from reportlab.lib import colors
    from reportlab.platypus import Spacer, Table, TableStyle

    table = Table(rows, colWidths=col_widths)
    style_cmds: list[Any] = [
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    if header_row and rows:
        style_cmds.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(style_cmds))
    story.append(table)
    story.append(Spacer(1, 8))


def cfg_to_mapping(cfg: Any) -> Mapping[str, Any]:
    """Best-effort coercion of a Hydra cfg to a plain dict for printing."""

    if isinstance(cfg, Mapping):
        return cfg
    # OmegaConf interop -- tolerate either DictConfig or a plain dataclass.
    try:
        from omegaconf import OmegaConf  # type: ignore[import-not-found]

        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[no-any-return]
    except Exception:
        pass
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {"value": str(cfg)}


__all__ = [
    "DEFAULT_PAGE_USABLE_WIDTH_PT",
    "SectionContext",
    "add_heading",
    "add_image",
    "add_paragraph",
    "add_placeholder",
    "add_table",
    "cfg_to_mapping",
]
