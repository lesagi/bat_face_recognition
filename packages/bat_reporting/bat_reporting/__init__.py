"""bat_reporting -- unified post-training PDF + run-comparison HTML.

Public surface:

* :func:`build_unified_pdf` -- compose the 8-section post-training PDF.
* :func:`compare_runs` -- render the side-by-side HTML run-comparison report.
* :class:`ReportData` (and helpers) -- the aggregator dataclass passed to the
  PDF builder.
* :mod:`bat_reporting.plots` -- matplotlib helpers used by the PDF and the
  comparison HTML; every helper routes filenames through
  :func:`bat_stats.naming.build_filename` so artifacts are self-identifying.
"""

from __future__ import annotations

from bat_reporting.compare import RunComparisonInputs, compare_runs
from bat_reporting.data import (
    EmbeddingProjection,
    PermutationSummary,
    ReportData,
    TrainingHistory,
)
from bat_reporting.pdf_report import DEFAULT_SECTION_ORDER, build_unified_pdf

__all__ = [
    "DEFAULT_SECTION_ORDER",
    "EmbeddingProjection",
    "PermutationSummary",
    "ReportData",
    "RunComparisonInputs",
    "TrainingHistory",
    "build_unified_pdf",
    "compare_runs",
]
