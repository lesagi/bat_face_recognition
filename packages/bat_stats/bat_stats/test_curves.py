"""Test-split evaluation curve figures.

Emitted as PNG artifacts to the MLflow ``figures/`` artifact directory so the
MLflow UI can render them inline.  All filenames and titles route through
:mod:`bat_stats.naming` per the load-bearing naming invariant.

Three figures (all optional depending on available data):

* **ROC curve** (``test_roc``): TPR vs FPR with the ROC-AUC annotated.
* **Recall vs FAR** (``test_recall_vs_far``): recall (TPR) vs FAR (FPR) on a
  log x-axis, with the standard biometric operating points (1e-4, 1e-3, 1e-2)
  marked as vertical lines.  The ``1e-2`` point is starred to highlight the
  headline metric adopted in commit ``7a3ca20``.
* **CMC curve** (``test_cmc``): cumulative match characteristic (rank-k
  accuracy vs rank) for embedding models.  Skipped when
  ``EvalReport.identification`` is ``None`` (pair/Siamese models).

All three are skipped entirely when ``EvalReport.predictions`` is absent or
contains a single class (AUC would be undefined / degenerate).  The helpers
never raise on degenerate input.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from bat_stats.naming import build_filename, build_title_suffix

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from bat_core import EvalReport, Predictions

_DPI = 150
_FIGSIZE: tuple[int, int] = (7, 5)

# Standard biometric operating points (FAR targets).
_FAR_TARGETS: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
_FAR_LABELS: dict[float, str] = {
    1e-4: "FAR=1e-4",
    1e-3: "FAR=1e-3",
    1e-2: "FAR=1e-2 ★",  # headline metric (recall@FAR=1e-2)
}
_FAR_PALETTE: dict[float, str] = {
    1e-4: "#d62728",
    1e-3: "#ff7f0e",
    1e-2: "#2ca02c",
}


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def _is_degenerate(y_true: Any) -> bool:
    """Return True if the label array contains fewer than two classes."""
    arr = np.asarray(list(y_true), dtype=np.float64)
    return len(arr) == 0 or len(np.unique(arr)) < 2


def _compute_roc(
    y_true: Any, y_score: Any
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``(auc, fpr, tpr)`` via sklearn's ``roc_curve``."""
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true_arr = np.asarray(list(y_true), dtype=np.float64)
    y_score_arr = np.asarray(list(y_score), dtype=np.float64)
    auc = float(roc_auc_score(y_true_arr, y_score_arr))
    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
    return auc, fpr, tpr


# ------------------------------------------------------------------
# public figure builders
# ------------------------------------------------------------------


def roc_curve_figure(predictions: "Predictions", cfg: Any) -> "Figure":
    """Build a ROC curve figure from *predictions*.

    Parameters
    ----------
    predictions:
        ``bat_core.Predictions`` with ``y_true`` and ``y_score`` tuples.
    cfg:
        Hydra cfg (or any mapping the naming helpers accept).
    """
    auc, fpr, tpr = _compute_roc(predictions.y_true, predictions.y_score)
    suffix = build_title_suffix(cfg)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve\n{suffix}")
    ax.legend(loc="lower right")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    return fig


def recall_vs_far_figure(predictions: "Predictions", cfg: Any) -> "Figure":
    """Build a Recall-vs-FAR figure on a log x-axis.

    Vertical lines mark FAR = 1e-4, 1e-3, 1e-2.  Each operating point that
    falls within the computed ROC curve is also marked with a dot at its
    actual recall.  The ``1e-2`` line is starred to surface the headline metric.

    Parameters
    ----------
    predictions:
        ``bat_core.Predictions`` with ``y_true`` and ``y_score`` tuples.
    cfg:
        Hydra cfg passed to :func:`bat_stats.naming.build_title_suffix`.
    """
    _, fpr, tpr = _compute_roc(predictions.y_true, predictions.y_score)
    suffix = build_title_suffix(cfg)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(fpr, tpr, color="steelblue", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("False Acceptance Rate (FAR, log scale)")
    ax.set_ylabel("Recall (TAR)")
    ax.set_title(f"Recall vs FAR\n{suffix}")

    for target in _FAR_TARGETS:
        color = _FAR_PALETTE[target]
        label = _FAR_LABELS[target]
        ax.axvline(x=target, linestyle="--", lw=1.2, color=color, label=label)
        # Mark the recall value on the curve at this FAR.
        mask = fpr <= target
        if mask.any():
            recall_at_far = float(tpr[mask].max())
            ax.scatter([target], [recall_at_far], color=color, zorder=5, s=60)

    ax.legend(loc="lower right")
    ax.set_ylim(0.0, 1.05)
    # Keep the x-axis from starting at zero (undefined on log scale).
    pos_fpr = fpr[fpr > 0]
    x_min = max(float(pos_fpr.min()) * 0.5 if len(pos_fpr) > 0 else 1e-5, 1e-6)
    ax.set_xlim(x_min, 1.0)
    fig.tight_layout()
    return fig


def cmc_curve_figure(cmc: tuple[float, ...], cfg: Any) -> "Figure":
    """Build a CMC (Cumulative Match Characteristic) curve figure.

    Parameters
    ----------
    cmc:
        Per-rank identification accuracy values
        (``bat_core.IdentificationMetrics.cmc``).
    cfg:
        Hydra cfg passed to :func:`bat_stats.naming.build_title_suffix`.
    """
    suffix = build_title_suffix(cfg)
    ranks = list(range(1, len(cmc) + 1))
    values = list(cmc)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(ranks, values, color="darkorange", lw=2, marker="o", markersize=5)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Identification Accuracy")
    ax.set_title(f"CMC Curve\n{suffix}")
    ax.set_xticks(ranks)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.5, len(ranks) + 0.5)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# orchestrator
# ------------------------------------------------------------------


def save_test_curves(
    eval_report: "EvalReport",
    cfg: Any,
    output_dir: Path | str,
) -> list[Path]:
    """Save test evaluation curve PNGs to *output_dir*.

    Builds the ROC, Recall-vs-FAR (and CMC if identification is present)
    figures and writes each as a PNG named via
    :func:`bat_stats.naming.build_filename`.

    Returns the list of :class:`Path` objects actually written.  An empty list
    signals that no figures were produced (degenerate / missing predictions),
    which the caller can silently skip.

    This function never raises — each figure is attempted independently;
    failures are silently skipped to avoid aborting a training run over a
    plotting error.

    Parameters
    ----------
    eval_report:
        ``bat_core.EvalReport`` from test evaluation.  ``predictions`` may be
        ``None`` (treated as degenerate — no figures produced).
    cfg:
        Hydra cfg used for naming.
    output_dir:
        Directory to write PNGs into (created if necessary).
    """
    preds = eval_report.predictions
    if preds is None or _is_degenerate(preds.y_true):
        return []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # ---- ROC ----
    try:
        fig = roc_curve_figure(preds, cfg)
        path = out / build_filename("test_roc", cfg, ".png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        _close_fig(fig)
        written.append(path)
    except Exception:  # noqa: BLE001
        pass

    # ---- Recall vs FAR ----
    try:
        fig = recall_vs_far_figure(preds, cfg)
        path = out / build_filename("test_recall_vs_far", cfg, ".png")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
        _close_fig(fig)
        written.append(path)
    except Exception:  # noqa: BLE001
        pass

    # ---- CMC (embedding models only) ----
    if (
        eval_report.identification is not None
        and eval_report.identification.cmc
    ):
        try:
            fig = cmc_curve_figure(eval_report.identification.cmc, cfg)
            path = out / build_filename("test_cmc", cfg, ".png")
            fig.savefig(path, dpi=_DPI, bbox_inches="tight")
            _close_fig(fig)
            written.append(path)
        except Exception:  # noqa: BLE001
            pass

    return written


def _close_fig(fig: Any) -> None:
    """Close *fig* without raising."""
    try:
        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "cmc_curve_figure",
    "recall_vs_far_figure",
    "roc_curve_figure",
    "save_test_curves",
]
