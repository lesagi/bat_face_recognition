"""Training-curves section: loss + monitored val metrics over epochs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bat_reporting.plots import plot_training_curves
from bat_reporting.sections.common import SectionContext, add_heading, add_image, add_placeholder

if TYPE_CHECKING:  # pragma: no cover
    from bat_reporting.data import ReportData


def render_section(story: list[Any], data: ReportData, ctx: SectionContext) -> None:
    from reportlab.platypus import PageBreak

    add_heading(story, "Training curves", ctx, level=1)

    history = data.training_history
    if history is None or (not history.train and not history.val):
        add_placeholder(story, "Training history was not provided for this run.", ctx)
        story.append(PageBreak())
        return

    nested: dict[str, dict[str, list[float]]] = {}
    if history.train:
        nested["train"] = {k: list(v) for k, v in history.train.items()}
    if history.val:
        nested["val"] = {k: list(v) for k, v in history.val.items()}

    png = plot_training_curves(nested, data.cfg, ctx.output_dir)
    add_image(story, png, ctx)
    story.append(PageBreak())


__all__ = ["render_section"]
