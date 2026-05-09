"""Aggregate dataclass for everything the unified PDF needs.

The ``ReportData`` aggregator is a *pure sink*: every field is computed by
upstream packages (``bat_evaluation``, ``bat_stats``, ``bat_interpretability``,
``bat_training``); ``bat_reporting`` never recomputes metrics.

Every field is optional except ``cfg`` and ``run_id``: pair-only runs have no
identification metrics, training-from-scratch has no permutation results, etc.
The PDF builder renders a "not available" placeholder for missing sections
rather than raising.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bat_core.types import EvalReport, SaliencyImage


@dataclass
class TrainingHistory:
    """Per-epoch curves for the unified PDF's training-curves page.

    ``losses`` and ``metrics`` are dicts keyed by metric name (e.g. ``"loss"``,
    ``"f1"``, ``"roc_auc"``) where the value is a list of length ``epochs``.
    The PDF builder plots train and val series side-by-side when both exist.
    """

    epochs: int = 0
    train: dict[str, list[float]] = field(default_factory=dict)
    val: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class EmbeddingProjection:
    """Pre-rendered t-SNE / UMAP scatter PNGs.

    ``bat_interpretability.EmbeddingProjectionAdapter`` returns these as
    ``SaliencyImage`` objects with ``.method == "tsne"`` / ``"umap"``; we keep
    the original ``SaliencyImage`` list (so the PDF can read both the PNG path
    and the projected coordinates if needed).
    """

    images: list[SaliencyImage] = field(default_factory=list)

    @property
    def tsne(self) -> SaliencyImage | None:
        for img in self.images:
            if img.method == "tsne":
                return img
        return None

    @property
    def umap(self) -> SaliencyImage | None:
        for img in self.images:
            if img.method == "umap":
                return img
        return None


@dataclass
class PermutationSummary:
    """Slim view over ``bat_stats.PermutationTestResults``.

    We accept either the dataclass directly or the dict produced by
    :meth:`PermutationTestResults.to_dict` so callers can persist results to
    JSON between training and reporting. The reporter never needs the
    full retrain object; it only needs the per-metric numbers and (optionally)
    pre-rendered null-distribution PNGs from ``PermutationVisualizer``.
    """

    results: Any | None = None  # bat_stats.PermutationTestResults
    plot_paths: list[Path] = field(default_factory=list)


@dataclass
class ReportData:
    """All the inputs the unified PDF builder consumes.

    Parameters
    ----------
    cfg:
        The full Hydra cfg (mapping or DictConfig). Used both for the title
        page snapshot and -- via ``bat_stats.naming`` -- for filename / title
        decoration.
    manifest_hash:
        SHA-256 of the resolved manifest CSV (from ``bat_core.Manifest``).
    run_id:
        MLflow run id (string).
    timestamp:
        ISO-8601 timestamp string. The CLI passes
        ``datetime.now().isoformat()``; tests can pass a fixed sentinel.
    training_history:
        Per-epoch loss + metric curves.
    eval_report:
        ``bat_core.EvalReport`` from ``bat_evaluation.run_eval_protocol``.
        Contains both verification and (optional) identification metrics.
    saliency_images:
        ``bat_core.SaliencyImage`` instances from
        ``bat_interpretability.SiameseSaliencyAdapter`` /
        ``GradCAMAdapter``. The PDF picks the first 5 for the saliency page.
    embedding_projection:
        ``bat_interpretability.EmbeddingProjectionAdapter`` output wrapped in
        :class:`EmbeddingProjection`.
    permutation:
        ``bat_stats`` permutation-test results.
    extra:
        Free-form pairs that section renderers can opt into (kept open so we
        don't have to grow the dataclass for every minor PDF field).
    """

    cfg: Mapping[str, Any] | Any
    run_id: str
    manifest_hash: str | None = None
    timestamp: str | None = None
    training_history: TrainingHistory | None = None
    eval_report: EvalReport | None = None
    saliency_images: Sequence[SaliencyImage] = field(default_factory=tuple)
    embedding_projection: EmbeddingProjection | None = None
    permutation: PermutationSummary | None = None
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "EmbeddingProjection",
    "PermutationSummary",
    "ReportData",
    "TrainingHistory",
]
