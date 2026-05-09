"""Identification table: CMC plot + TAR@FAR + top-1 / top-5 / mAP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bat_reporting.plots import plot_cmc
from bat_reporting.sections.common import (
    SectionContext,
    add_heading,
    add_image,
    add_placeholder,
    add_table,
)

if TYPE_CHECKING:  # pragma: no cover
    from bat_reporting.data import ReportData


def render_section(story: list[Any], data: ReportData, ctx: SectionContext) -> None:
    from reportlab.platypus import PageBreak

    add_heading(story, "Identification metrics", ctx, level=1)

    eval_report = data.eval_report
    if eval_report is None or eval_report.identification is None:
        add_placeholder(
            story,
            (
                "Identification metrics not available -- this is expected for "
                "pair-only (verification-only) runs."
            ),
            ctx,
        )
        story.append(PageBreak())
        return

    ident = eval_report.identification
    verif = eval_report.verification

    rows = [
        ["Metric", "Value"],
        ["top-1", f"{ident.top1:.4f}"],
        ["top-5", f"{ident.top5:.4f}"],
        ["mAP", f"{ident.map:.4f}"],
    ]
    if verif is not None:
        rows.extend(
            [
                ["TAR@FAR=1e-3", f"{verif.tar_at_far_1e3:.4f}"],
                ["TAR@FAR=1e-4", f"{verif.tar_at_far_1e4:.4f}"],
            ]
        )
    add_table(story, rows, ctx, col_widths=[180, 120])

    if ident.cmc:
        png = plot_cmc(list(ident.cmc), data.cfg, ctx.output_dir)
        add_image(story, png, ctx)
    else:
        add_placeholder(story, "CMC curve unavailable for this run.", ctx)

    story.append(PageBreak())


__all__ = ["render_section"]
