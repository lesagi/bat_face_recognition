"""Section renderers must render a placeholder rather than crash on missing
optional fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("reportlab")

from bat_reporting.data import ReportData  # noqa: E402
from bat_reporting.sections import (  # noqa: E402
    confusion,
    identification,
    permutation,
    projection,
    roc,
    saliency,
    title,
    training_curves,
)
from bat_reporting.sections.common import SectionContext  # noqa: E402


def _ctx(tmp_path: Path) -> SectionContext:
    from reportlab.lib.styles import getSampleStyleSheet

    return SectionContext(output_dir=tmp_path, styles=getSampleStyleSheet())


@pytest.mark.parametrize(
    "renderer",
    [
        title.render_section,
        training_curves.render_section,
        confusion.render_section,
        roc.render_section,
        identification.render_section,
        saliency.render_section,
        projection.render_section,
        permutation.render_section,
    ],
)
def test_section_handles_missing_data(renderer: Any, cfg: dict[str, Any], tmp_path: Path) -> None:
    """Empty ReportData -> every section renders a placeholder, not a crash."""

    story: list[Any] = []
    data = ReportData(cfg=cfg, run_id="empty-run")
    renderer(story, data, _ctx(tmp_path))
    assert story, f"{renderer.__name__} produced no flowables for an empty ReportData"


def test_pdf_default_section_order_matches_plan() -> None:
    """The 8-section order is locked by the plan -- guard against drift."""

    from bat_reporting.pdf_report import DEFAULT_SECTION_ORDER

    expected = (
        title.render_section,
        training_curves.render_section,
        confusion.render_section,
        roc.render_section,
        identification.render_section,
        saliency.render_section,
        projection.render_section,
        permutation.render_section,
    )
    assert tuple(DEFAULT_SECTION_ORDER) == expected
