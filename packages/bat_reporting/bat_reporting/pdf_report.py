"""Unified post-training PDF builder.

The plan dictates the section order; we just dispatch each section in order
and let it append flowables to the running Platypus story. The section
modules know how to handle missing optional fields (rendered as a placeholder
paragraph rather than raising), so a partial ``ReportData`` still produces a
valid PDF.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bat_reporting.data import ReportData
from bat_reporting.sections import (
    confusion,
    identification,
    permutation,
    projection,
    roc,
    saliency,
    title,
    training_curves,
)
from bat_reporting.sections.common import SectionContext

SectionRenderer = Callable[[list[Any], ReportData, SectionContext], None]

# Order is locked by the plan. Tests assert it.
DEFAULT_SECTION_ORDER: tuple[SectionRenderer, ...] = (
    title.render_section,
    training_curves.render_section,
    confusion.render_section,
    roc.render_section,
    identification.render_section,
    saliency.render_section,
    projection.render_section,
    permutation.render_section,
)


def build_unified_pdf(
    report_data: ReportData,
    output_path: Path,
    *,
    sections: Sequence[SectionRenderer] | None = None,
) -> Path:
    """Render the unified post-training PDF.

    Parameters
    ----------
    report_data:
        Aggregate of everything the report needs (cfg, eval, saliency, ...).
    output_path:
        Path the PDF will be written to. Parent directories are created.
        Plot PNGs land alongside the PDF (so MLflow can upload the whole
        directory in one shot).
    sections:
        Override the default section order (mainly a test seam).

    Returns
    -------
    pathlib.Path
        The same ``output_path`` (resolved to absolute), for chaining.
    """

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = out_path.parent

    styles = getSampleStyleSheet()
    ctx = SectionContext(output_dir=plot_dir, styles=styles)
    story: list[Any] = []

    chosen = list(sections) if sections is not None else list(DEFAULT_SECTION_ORDER)
    for renderer in chosen:
        renderer(story, report_data, ctx)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        title=f"bat-face-rec post-training report ({report_data.run_id})",
        author="bat_reporting",
    )
    doc.build(story)
    return out_path.resolve()


__all__ = ["DEFAULT_SECTION_ORDER", "SectionRenderer", "build_unified_pdf"]
