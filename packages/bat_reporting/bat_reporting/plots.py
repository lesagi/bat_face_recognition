"""Matplotlib helpers that emit PNGs for the unified PDF to embed.

Every helper in this module:

* takes a Hydra ``cfg`` and routes the output filename through
  :func:`bat_stats.naming.build_filename`, so a standalone plot file is
  self-identifying;
* decorates the title with :func:`bat_stats.naming.build_title_suffix`, so a
  PDF page or a thumbnail is unambiguous;
* uses the ``Agg`` backend so it works under headless test environments.

The PDF builder embeds the returned ``Path`` directly via
``reportlab.platypus.Image``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402 -- headless safety must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from bat_stats.naming import build_filename, build_title_suffix  # noqa: E402

DEFAULT_DPI = 150
DEFAULT_FIGSIZE = (8.0, 5.5)


def _save(fig: plt.Figure, out_dir: Path, fname: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_training_curves(
    history: Mapping[str, Sequence[float]] | Mapping[str, Mapping[str, Sequence[float]]],
    cfg: Any,
    output_dir: Path,
    *,
    stem: str = "training_curves",
) -> Path:
    """Plot training curves (loss + monitored metrics) over epochs.

    Accepts either:

    * ``{"train": {"loss": [...], "f1": [...]}, "val": {...}}`` — the natural
      shape produced by :class:`bat_reporting.data.TrainingHistory`; **or**
    * a flat ``{"loss": [...], "f1": [...]}`` mapping with no train/val split.
    """

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    nested = bool(history) and all(isinstance(v, Mapping) for v in history.values())

    if nested:
        # Two-level dict; plot each (split, metric) as its own line.
        colors = plt.get_cmap("tab10")
        idx = 0
        for split, metrics_map in history.items():
            for metric_name, series in metrics_map.items():
                if not series:
                    continue
                xs = np.arange(1, len(series) + 1)
                ax.plot(
                    xs,
                    list(series),
                    label=f"{split}/{metric_name}",
                    color=colors(idx % 10),
                    linewidth=1.5,
                )
                idx += 1
    else:
        colors = plt.get_cmap("tab10")
        for idx, (metric_name, series) in enumerate(history.items()):
            if not series:
                continue
            xs = np.arange(1, len(series) + 1)
            ax.plot(
                xs,
                list(series),
                label=str(metric_name),
                color=colors(idx % 10),
                linewidth=1.5,
            )

    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title(f"Training curves\n{build_title_suffix(cfg)}")
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, output_dir, build_filename(stem, cfg, suffix=".png"))


def plot_confusion(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    threshold: float,
    cfg: Any,
    output_dir: Path,
    *,
    stem: str = "confusion_matrix",
) -> Path:
    """Plot a 2x2 confusion matrix at the optimal threshold."""

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    matrix = np.asarray([[tp, fn], [fp, tn]], dtype=np.int64)
    im = ax.imshow(matrix, cmap="Blues", aspect="equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred pos", "pred neg"])
    ax.set_yticklabels(["true pos", "true neg"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color="black",
                fontsize=14,
            )
    ax.set_title(
        f"Confusion matrix (threshold={threshold:.4f})\n{build_title_suffix(cfg)}",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, output_dir, build_filename(stem, cfg, suffix=".png"))


def plot_roc(
    fpr: Sequence[float],
    tpr: Sequence[float],
    auc: float,
    cfg: Any,
    output_dir: Path,
    *,
    stem: str = "roc_curve",
    extra_curves: Sequence[tuple[str, Sequence[float], Sequence[float]]] | None = None,
) -> Path:
    """Plot a ROC curve. Optional ``extra_curves`` overlays additional series
    (used by :func:`compare_runs` for the ROC-overlay plot)."""

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.plot(list(fpr), list(tpr), color="C0", linewidth=2.0, label=f"AUC = {auc:.4f}")
    if extra_curves:
        cmap = plt.get_cmap("tab10")
        for idx, (label, x, y) in enumerate(extra_curves):
            ax.plot(list(x), list(y), color=cmap((idx + 1) % 10), linewidth=1.5, label=label)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curve\n{build_title_suffix(cfg)}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    return _save(fig, output_dir, build_filename(stem, cfg, suffix=".png"))


def plot_cmc(
    cmc: Sequence[float],
    cfg: Any,
    output_dir: Path,
    *,
    stem: str = "cmc_curve",
    extra_curves: Sequence[tuple[str, Sequence[float]]] | None = None,
) -> Path:
    """Plot the CMC curve (rank vs cumulative match)."""

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    cmc_arr = np.asarray(list(cmc), dtype=np.float64)
    ranks = np.arange(1, cmc_arr.size + 1)
    ax.plot(ranks, cmc_arr, marker="o", color="C0", linewidth=2.0, label="primary")
    if extra_curves:
        cmap = plt.get_cmap("tab10")
        for idx, (label, series) in enumerate(extra_curves):
            arr = np.asarray(list(series), dtype=np.float64)
            ax.plot(
                np.arange(1, arr.size + 1),
                arr,
                marker="o",
                color=cmap((idx + 1) % 10),
                linewidth=1.5,
                label=label,
            )
    ax.set_xlabel("rank k")
    ax.set_ylabel("cumulative match P(hit @ rank<=k)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"CMC curve\n{build_title_suffix(cfg)}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    return _save(fig, output_dir, build_filename(stem, cfg, suffix=".png"))


def plot_permutation_null(
    metric: str,
    observed: float,
    null_distribution: Sequence[float],
    cfg: Any,
    output_dir: Path,
    *,
    stem: str | None = None,
) -> Path:
    """Plot a single null-distribution histogram with the observed marker."""

    fname_stem = stem or f"null_dist_{metric}"
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    null_arr = np.asarray(list(null_distribution), dtype=np.float64)
    if null_arr.size:
        ax.hist(
            null_arr,
            bins=min(30, max(5, null_arr.size // 2)),
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
            label=f"null (n={null_arr.size})",
        )
    ax.axvline(
        observed, color="red", linestyle="--", linewidth=2.0, label=f"observed={observed:.4f}"
    )
    ax.set_xlabel(metric)
    ax.set_ylabel("density")
    ax.set_title(f"Null distribution: {metric}\n{build_title_suffix(cfg)}")
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, output_dir, build_filename(fname_stem, cfg, suffix=".png"))


__all__ = [
    "DEFAULT_DPI",
    "DEFAULT_FIGSIZE",
    "plot_cmc",
    "plot_confusion",
    "plot_permutation_null",
    "plot_roc",
    "plot_training_curves",
]
