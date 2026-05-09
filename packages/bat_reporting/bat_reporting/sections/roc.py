"""ROC curve + AUC section."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from bat_evaluation.verification import compute_roc
from bat_reporting.plots import plot_roc
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

    add_heading(story, "ROC curve and AUC", ctx, level=1)

    eval_report = data.eval_report
    if eval_report is None or eval_report.verification is None:
        add_placeholder(story, "Verification metrics were not provided.", ctx)
        story.append(PageBreak())
        return

    auc_value = float(eval_report.verification.roc_auc)
    if eval_report.predictions is None:
        # We can still cite the AUC even without the curve.
        add_paragraph(
            story,
            (
                f"ROC-AUC = <b>{auc_value:.4f}</b>. "
                "Per-pair predictions were not provided; ROC curve cannot be drawn."
            ),
            ctx,
        )
        story.append(PageBreak())
        return

    auc, fpr, tpr, _thr = compute_roc(
        np.asarray(eval_report.predictions.y_true, dtype=np.float64),
        np.asarray(eval_report.predictions.y_score, dtype=np.float64),
    )
    if np.isnan(auc):
        add_placeholder(
            story,
            "Predictions contained a single class; AUC is undefined for this run.",
            ctx,
        )
        story.append(PageBreak())
        return

    png = plot_roc(fpr.tolist(), tpr.tolist(), auc, data.cfg, ctx.output_dir)
    add_image(story, png, ctx)
    add_paragraph(
        story,
        (
            f"ROC-AUC = <b>{auc:.4f}</b>. "
            f"TAR@FAR=1e-3 = {eval_report.verification.tar_at_far_1e3:.4f}; "
            f"TAR@FAR=1e-4 = {eval_report.verification.tar_at_far_1e4:.4f}."
        ),
        ctx,
    )
    story.append(PageBreak())


__all__ = ["render_section"]
