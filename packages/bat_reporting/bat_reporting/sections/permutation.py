"""Permutation-test summary: table of metric-vs-null p-values + plots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bat_reporting.plots import plot_permutation_null
from bat_reporting.sections.common import (
    SectionContext,
    add_heading,
    add_image,
    add_paragraph,
    add_placeholder,
    add_table,
)

if TYPE_CHECKING:  # pragma: no cover
    from bat_reporting.data import ReportData


def render_section(story: list[Any], data: ReportData, ctx: SectionContext) -> None:
    from reportlab.platypus import PageBreak

    add_heading(story, "Permutation-test summary", ctx, level=1)

    perm = data.permutation
    if perm is None or perm.results is None:
        add_placeholder(story, "Permutation-test results were not provided.", ctx)
        story.append(PageBreak())
        return

    results = perm.results
    metrics_dict = _extract_metrics(results)

    if not metrics_dict:
        add_placeholder(story, "Permutation-test results contained no metrics.", ctx)
        story.append(PageBreak())
        return

    n_perm = getattr(results, "n_permutations", None)
    sig_level = getattr(results, "significance_level", None)
    if n_perm is not None and sig_level is not None:
        add_paragraph(
            story,
            f"n={n_perm} permutations; significance level = {sig_level}.",
            ctx,
        )

    rows: list[list[str]] = [
        ["Metric", "Observed", "Null mean", "Null std", "p-value", "Significant"]
    ]
    for metric, mr in metrics_dict.items():
        rows.append(
            [
                metric,
                f"{_get(mr, 'observed'):.4f}",
                f"{_get(mr, 'null_mean'):.4f}",
                f"{_get(mr, 'null_std'):.4f}",
                f"{_get(mr, 'p_value'):.4f}",
                "Yes" if _get(mr, "significant") else "No",
            ]
        )
    add_table(story, rows, ctx)

    # Inline pre-rendered visualizer plots if the caller provided any.
    for png in perm.plot_paths or []:
        add_image(story, png, ctx)

    # Otherwise emit one null-distribution plot per metric on the fly.
    if not perm.plot_paths:
        for metric, mr in metrics_dict.items():
            null_dist = _get(mr, "null_distribution") or []
            if not null_dist:
                continue
            png = plot_permutation_null(
                metric,
                float(_get(mr, "observed")),
                list(null_dist),
                data.cfg,
                ctx.output_dir,
            )
            add_image(story, png, ctx)

    story.append(PageBreak())


def _extract_metrics(results: Any) -> dict[str, Any]:
    """Return ``{metric_name: MetricResult-like}`` from the supplied results.

    Tolerates both the live ``PermutationTestResults`` dataclass and the
    plain dict produced by its ``.to_dict()`` method.
    """

    if hasattr(results, "metrics"):
        m = results.metrics
        return dict(m)
    if isinstance(results, dict) and "metrics" in results:
        return dict(results["metrics"])
    return {}


def _get(obj: Any, key: str) -> Any:
    """Read ``obj.key`` or ``obj[key]`` interchangeably."""

    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


__all__ = ["render_section"]
