"""Tests for :mod:`bat_stats.test_curves`.

Covers:
* Each figure builder returns a ``Figure`` whose title contains the
  experiment name produced by :func:`bat_stats.naming.experiment_name`.
* ``save_test_curves`` writes the expected PNG filenames (named via
  :func:`bat_stats.naming.build_filename`).
* Graceful edge-case handling: ``identification=None`` → no CMC file;
  single-class predictions → returns ``[]``, writes nothing, no exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — headless-safe before any plt import

import pytest

from bat_stats.naming import build_filename, build_title_suffix, experiment_name
from bat_stats.test_curves import (
    cmc_curve_figure,
    recall_vs_far_figure,
    roc_curve_figure,
    save_test_curves,
)


# ------------------------------------------------------------------
# Minimal stand-ins for bat_core types (no torch required in bat_stats)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _Predictions:
    y_true: tuple
    y_score: tuple


@dataclass(frozen=True)
class _VerificationMetrics:
    roc_auc: float = 0.8
    youden_j: float = 0.5
    optimal_threshold: float = 0.6
    tar_at_far_1e3: float = 0.4
    tar_at_far_1e4: float = 0.2
    threshold_at_far_1e2: float = 0.55
    recall_at_far_1e2: float = 0.7


@dataclass(frozen=True)
class _IdentificationMetrics:
    top1: float = 0.8
    top5: float = 0.95
    map: float = 0.75
    cmc: tuple = field(default_factory=lambda: tuple(0.8 + 0.02 * i for i in range(10)))


@dataclass(frozen=True)
class _EvalReport:
    verification: _VerificationMetrics = field(default_factory=_VerificationMetrics)
    identification: _IdentificationMetrics | None = field(
        default_factory=_IdentificationMetrics
    )
    predictions: _Predictions | None = None


def _binary_predictions(n: int = 80, seed: int = 0) -> _Predictions:
    """Build a balanced binary-class prediction set."""
    import numpy as np

    rng = np.random.default_rng(seed)
    y_true = tuple(int(v) for v in ([1] * (n // 2) + [0] * (n // 2)))
    # Scores: positives drawn from N(0.7, 0.1), negatives from N(0.3, 0.1).
    y_score_pos = rng.normal(0.7, 0.1, n // 2).tolist()
    y_score_neg = rng.normal(0.3, 0.1, n // 2).tolist()
    y_score = tuple(float(v) for v in y_score_pos + y_score_neg)
    return _Predictions(y_true=y_true, y_score=y_score)


def _single_class_predictions() -> _Predictions:
    return _Predictions(y_true=(1, 1, 1, 1), y_score=(0.9, 0.8, 0.7, 0.6))


# ------------------------------------------------------------------
# Tests: individual figure builders
# ------------------------------------------------------------------


def test_roc_curve_figure_title_contains_experiment_signature(cfg: Any) -> None:
    """ROC figure title must include the experiment suffix from naming helpers."""
    preds = _binary_predictions()
    fig = roc_curve_figure(preds, cfg)
    suffix = build_title_suffix(cfg)
    title = fig.axes[0].get_title()
    assert suffix in title, f"title suffix {suffix!r} not found in title {title!r}"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_recall_vs_far_figure_title_contains_experiment_signature(cfg: Any) -> None:
    """Recall-vs-FAR figure title must include the experiment suffix."""
    preds = _binary_predictions()
    fig = recall_vs_far_figure(preds, cfg)
    suffix = build_title_suffix(cfg)
    title = fig.axes[0].get_title()
    assert suffix in title, f"title suffix {suffix!r} not found in title {title!r}"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_recall_vs_far_figure_xscale_is_log(cfg: Any) -> None:
    """The x-axis must use a log scale."""
    preds = _binary_predictions()
    fig = recall_vs_far_figure(preds, cfg)
    assert fig.axes[0].get_xscale() == "log"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_cmc_curve_figure_title_contains_experiment_signature(cfg: Any) -> None:
    """CMC figure title must include the experiment suffix."""
    cmc = tuple(0.5 + 0.05 * i for i in range(10))
    fig = cmc_curve_figure(cmc, cfg)
    suffix = build_title_suffix(cfg)
    title = fig.axes[0].get_title()
    assert suffix in title, f"title suffix {suffix!r} not found in title {title!r}"
    import matplotlib.pyplot as plt
    plt.close(fig)


# ------------------------------------------------------------------
# Tests: save_test_curves orchestrator
# ------------------------------------------------------------------


def test_save_test_curves_writes_expected_filenames(
    tmp_path: Path, cfg: Any
) -> None:
    """Embedding report (with identification) → three PNGs with naming-helper names."""
    report = _EvalReport(
        predictions=_binary_predictions(),
        identification=_IdentificationMetrics(),
    )
    written = save_test_curves(report, cfg, tmp_path)

    assert len(written) == 3
    stems = {"test_roc", "test_recall_vs_far", "test_cmc"}
    for stem in stems:
        expected_name = build_filename(stem, cfg, ".png")
        assert any(p.name == expected_name for p in written), (
            f"expected {expected_name!r} in {[p.name for p in written]}"
        )
    # All files must actually exist on disk.
    for p in written:
        assert p.exists(), f"{p} was listed but not written"


def test_save_test_curves_skips_cmc_when_no_identification(
    tmp_path: Path, cfg: Any
) -> None:
    """Pair/Siamese report (no identification) → only ROC + recall-vs-FAR."""
    report = _EvalReport(
        predictions=_binary_predictions(),
        identification=None,
    )
    written = save_test_curves(report, cfg, tmp_path)

    assert len(written) == 2
    names = [p.name for p in written]
    assert any("test_roc" in n for n in names)
    assert any("test_recall_vs_far" in n for n in names)
    assert not any("test_cmc" in n for n in names)


def test_save_test_curves_returns_empty_on_single_class(
    tmp_path: Path, cfg: Any
) -> None:
    """Single-class predictions → degenerate, no files written, no exception."""
    report = _EvalReport(predictions=_single_class_predictions())
    written = save_test_curves(report, cfg, tmp_path)

    assert written == []
    # Directory was not even created (or is empty).
    assert not tmp_path.joinpath("figures").exists() or not list(
        tmp_path.joinpath("figures").glob("*.png")
    )


def test_save_test_curves_returns_empty_when_predictions_none(
    tmp_path: Path, cfg: Any
) -> None:
    """``predictions=None`` → no figures, no exception."""
    report = _EvalReport(predictions=None)
    written = save_test_curves(report, cfg, tmp_path)

    assert written == []


def test_save_test_curves_creates_output_dir(tmp_path: Path, cfg: Any) -> None:
    """The output directory is created if it doesn't exist."""
    nested = tmp_path / "deep" / "nested"
    report = _EvalReport(predictions=_binary_predictions(), identification=None)
    written = save_test_curves(report, cfg, nested)

    assert nested.exists()
    assert len(written) == 2
