"""Per-section renderers for the unified post-training PDF.

Each module exposes ``render_section(story, data, ctx) -> None`` where:

* ``story`` is the running ``list[reportlab.platypus.Flowable]`` the section
  should append to (we use Platypus, not raw canvas, so each page can grow
  flexibly);
* ``data`` is the :class:`bat_reporting.data.ReportData` aggregate;
* ``ctx`` is a :class:`bat_reporting.sections.common.SectionContext` carrying
  the ``output_dir`` for plot PNGs, the resolved styles, and any cached
  metadata.

The dispatcher in :mod:`bat_reporting.pdf_report` calls each section in the
order documented in the plan (title -> training curves -> confusion -> ROC ->
identification -> saliency -> projection -> permutation).
"""

from __future__ import annotations

from bat_reporting.sections import (
    common,
    confusion,
    identification,
    permutation,
    projection,
    roc,
    saliency,
    title,
    training_curves,
)

__all__ = [
    "common",
    "confusion",
    "identification",
    "permutation",
    "projection",
    "roc",
    "saliency",
    "title",
    "training_curves",
]
