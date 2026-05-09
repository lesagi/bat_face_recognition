"""Run-comparison HTML report.

Pulls metric data + tags for two MLflow runs (defaults to the live MLflow
client in this workspace, mockable via the ``mlflow_client`` parameter for
tests), renders a side-by-side metric table and optional ROC / CMC overlay
plots through a jinja2 template.

The two runs may be identified either by their MLflow run_id or by a
``RunComparison`` payload built upstream (the latter is the test seam).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bat_reporting.plots import plot_cmc, plot_roc

if TYPE_CHECKING:  # pragma: no cover
    from mlflow.tracking import MlflowClient

# Metrics we surface in the comparison table, in display order. Any extra
# metrics present on either run are appended at the end (alphabetically).
DEFAULT_METRIC_ORDER: tuple[str, ...] = (
    "test/roc_auc",
    "test/tar_at_far_1e3",
    "test/tar_at_far_1e4",
    "test/top1",
    "test/top5",
    "test/map",
    "val/f1",
    "val/roc_auc",
    "best_f1_value",
    "final_threshold",
)

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "compare.html"


@dataclass
class RunComparisonInputs:
    """Plain-data view over a single MLflow run, used as the comparison input."""

    run_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    experiment_name: str | None = None
    cfg: Any | None = None
    roc_curve: tuple[Sequence[float], Sequence[float]] | None = None  # (fpr, tpr)
    cmc_curve: Sequence[float] | None = None


def compare_runs(
    run_id_a: str,
    run_id_b: str,
    output_path: Path,
    *,
    mlflow_client: MlflowClient | None = None,
    inputs_a: RunComparisonInputs | None = None,
    inputs_b: RunComparisonInputs | None = None,
    plot_cfg: Any | None = None,
) -> Path:
    """Render a side-by-side run-comparison HTML report.

    Parameters
    ----------
    run_id_a, run_id_b:
        MLflow run ids. When ``inputs_a`` / ``inputs_b`` are omitted the
        function pulls metrics + tags from MLflow directly.
    output_path:
        Destination ``.html`` file. Parent directories are created.
    mlflow_client:
        Optional pre-built :class:`mlflow.tracking.MlflowClient`. Defaults to
        a freshly constructed client.
    inputs_a, inputs_b:
        Pre-built :class:`RunComparisonInputs` (test seam). When supplied,
        no MLflow call is made.
    plot_cfg:
        Optional Hydra cfg for filename / title routing of the overlay plots
        (so the PNGs sit alongside the HTML and follow the standard naming
        convention). When ``None``, falls back to ``inputs_a.cfg``.

    Returns
    -------
    Path to the written HTML report.
    """

    a = inputs_a if inputs_a is not None else _from_mlflow(run_id_a, mlflow_client)
    b = inputs_b if inputs_b is not None else _from_mlflow(run_id_b, mlflow_client)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_rows = _build_metric_rows(a.metrics, b.metrics)

    cfg_for_plots = plot_cfg if plot_cfg is not None else (a.cfg or b.cfg)
    roc_overlay = _maybe_render_roc(a, b, output_path.parent, cfg_for_plots)
    cmc_overlay = _maybe_render_cmc(a, b, output_path.parent, cfg_for_plots)

    html = _render_template(
        run_a=a,
        run_b=b,
        metric_rows=metric_rows,
        roc_overlay=roc_overlay.name if roc_overlay else None,
        cmc_overlay=cmc_overlay.name if cmc_overlay else None,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path.resolve()


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #
def _build_metric_rows(
    metrics_a: Mapping[str, float],
    metrics_b: Mapping[str, float],
) -> list[dict[str, Any]]:
    keys = list(DEFAULT_METRIC_ORDER)
    extras = sorted(set(metrics_a) | set(metrics_b) - set(keys))
    keys.extend(k for k in extras if k not in keys)

    rows: list[dict[str, Any]] = []
    for key in keys:
        a_val = metrics_a.get(key)
        b_val = metrics_b.get(key)
        if a_val is None and b_val is None:
            continue
        delta = float(b_val) - float(a_val) if (a_val is not None and b_val is not None) else None
        rows.append(
            {
                "metric": key,
                "a": _fmt(a_val),
                "b": _fmt(b_val),
                "delta": delta,
            }
        )
    return rows


def _fmt(v: Any) -> str | None:
    if v is None:
        return None
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _from_mlflow(run_id: str, client: MlflowClient | None) -> RunComparisonInputs:
    from mlflow.tracking import MlflowClient

    c = client if client is not None else MlflowClient()
    run = c.get_run(run_id)
    metrics = dict(run.data.metrics)
    tags = run.data.tags or {}
    return RunComparisonInputs(
        run_id=run_id,
        metrics={k: float(v) for k, v in metrics.items()},
        experiment_name=tags.get("experiment_name") or tags.get("mlflow.runName"),
    )


def _maybe_render_roc(
    a: RunComparisonInputs,
    b: RunComparisonInputs,
    out_dir: Path,
    cfg: Any | None,
) -> Path | None:
    if a.roc_curve is None or b.roc_curve is None or cfg is None:
        return None
    fpr_a, tpr_a = a.roc_curve
    fpr_b, tpr_b = b.roc_curve
    auc_a = float(a.metrics.get("test/roc_auc", float("nan")))
    return plot_roc(
        fpr_a,
        tpr_a,
        auc_a,
        cfg,
        out_dir,
        stem="roc_overlay",
        extra_curves=[(f"B ({b.run_id})", fpr_b, tpr_b)],
    )


def _maybe_render_cmc(
    a: RunComparisonInputs,
    b: RunComparisonInputs,
    out_dir: Path,
    cfg: Any | None,
) -> Path | None:
    if a.cmc_curve is None or b.cmc_curve is None or cfg is None:
        return None
    return plot_cmc(
        list(a.cmc_curve),
        cfg,
        out_dir,
        stem="cmc_overlay",
        extra_curves=[(f"B ({b.run_id})", list(b.cmc_curve))],
    )


def _render_template(**kwargs: Any) -> str:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(TEMPLATE_NAME)
    return template.render(**kwargs)


__all__ = [
    "DEFAULT_METRIC_ORDER",
    "RunComparisonInputs",
    "compare_runs",
]
