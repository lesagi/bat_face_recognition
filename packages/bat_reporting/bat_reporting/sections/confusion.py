"""Confusion-matrix section at the optimal threshold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from bat_evaluation.verification import confusion_at_threshold
from bat_reporting.plots import plot_confusion
from bat_reporting.sections.common import (
    SectionContext,
    add_heading,
    add_image,
    add_paragraph,
    add_placeholder,
)

if TYPE_CHECKING:  # pragma: no cover
    from bat_reporting.data import ReportData


def render_section(story: list[Any], data: ReportData, ctx: SectionContext) -> None:
    from reportlab.platypus import PageBreak

    add_heading(story, "Confusion matrix at optimal threshold", ctx, level=1)

    eval_report = data.eval_report
    if eval_report is None or eval_report.predictions is None or eval_report.verification is None:
        add_placeholder(
            story,
            "Verification predictions were not provided; cannot derive confusion matrix.",
            ctx,
        )
        story.append(PageBreak())
        return

    threshold = float(eval_report.verification.optimal_threshold)
    if not np.isfinite(threshold):
        add_placeholder(
            story,
            "Optimal threshold is non-finite (single-class predictions?); skipping.",
            ctx,
        )
        story.append(PageBreak())
        return

    cm = confusion_at_threshold(
        eval_report.predictions.y_true,
        eval_report.predictions.y_score,
        threshold,
    )

    png = plot_confusion(cm.tp, cm.fp, cm.tn, cm.fn, threshold, data.cfg, ctx.output_dir)
    add_image(story, png, ctx)
    add_paragraph(
        story,
        (
            f"Threshold = {threshold:.4f}; "
            f"precision = {cm.precision:.4f}, recall = {cm.recall:.4f}, "
            f"specificity = {cm.specificity:.4f}, F1 = {cm.f1:.4f}."
        ),
        ctx,
    )
    story.append(PageBreak())


__all__ = ["render_section"]
