"""Orchestration entry points for permutation tests.

Ported from ``app/statistical_tests/run_permutation_test.py``, with the
TF / argparse / mlflow CLI surface removed.  The CLI itself lives in
:mod:`bat_cli`; this module exposes the Python-level run helpers that the
CLI dispatches to.

Two modes:
- ``run_inference_test``: fast (numpy-only) -- delegates to
  :mod:`bat_stats.inference`.  The recommended default.
- ``run_retrain_test``: classical -- requires a ``trainer_factory`` that
  returns a :class:`bat_stats.trainer_hooks.TrainerProtocol`.  ``bat_stats``
  must not import ``bat_training`` directly (that would be an upward
  workspace dependency), so the default factory raises
  ``NotImplementedError``; callers inject a real factory via
  :mod:`bat_training.permutation_adapter`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from bat_core import Predictions
from bat_stats.inference import run_inference_permutation_test
from bat_stats.permutation_test import DEFAULT_METRICS, PermutationTest, PermutationTestResults
from bat_stats.trainer_hooks import TrainerProtocol, create_permutation_trainer
from bat_stats.visualizer import PermutationVisualizer


def run_inference_test(
    cfg: Any,
    predictions: Predictions | np.ndarray,
    labels: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
    n_permutations: int = 1000,
    significance_level: float = 0.05,
    metrics_to_test: Sequence[str] | None = None,
    degradation: bool = False,
    seed: int = 42,
    output_dir: str | Path | None = None,
    generate_report: bool = True,
    verbose: bool = True,
) -> PermutationTestResults:
    """Run the inference-based permutation test and (optionally) emit plots.

    Parameters
    ----------
    cfg
        Hydra cfg (or any object understood by :mod:`bat_stats.naming`).
        Drives every output filename and plot title.
    predictions, labels, threshold, n_permutations, significance_level,
    metrics_to_test, seed
        Forwarded to :func:`bat_stats.inference.run_inference_permutation_test`.
    degradation
        If ``True``, also compute the degradation curve (21 fractions in 5%
        increments).
    output_dir
        Where plots and JSON go.  If ``None``, no artifacts are written.
    generate_report
        If ``True`` and ``output_dir`` is set, emit the full visualizer report.
    """

    degradation_fractions: list[float] | None = None
    if degradation:
        degradation_fractions = [round(x * 0.05, 2) for x in range(21)]

    results, degradation_data = run_inference_permutation_test(
        predictions=predictions,
        labels=labels,
        threshold=threshold,
        n_permutations=n_permutations,
        significance_level=significance_level,
        metrics_to_test=metrics_to_test,
        degradation_fractions=degradation_fractions,
        seed=seed,
        verbose=verbose,
    )

    if output_dir is not None and generate_report:
        viz = PermutationVisualizer(results, cfg=cfg, output_dir=output_dir)
        viz.generate_full_report(
            show=False,
            export_json=True,
            degradation_data=degradation_data,
        )

    return results


def run_retrain_test(
    cfg: Any,
    observed_metrics: dict[str, float],
    *,
    trainer_factory: Callable[..., TrainerProtocol] | None = None,
    n_permutations: int = 100,
    permutation_epochs: int = 10,
    significance_level: float = 0.05,
    metrics_to_test: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    generate_report: bool = True,
    verbose: bool = True,
    **trainer_kwargs: Any,
) -> PermutationTestResults:
    """Run the classical retrain-from-scratch permutation test.

    ``bat_stats`` deliberately does not import ``bat_training`` (that would
    flip a dependency edge); the default ``trainer_factory`` is a stub that
    raises ``NotImplementedError``.  Production callers wire in the real
    PyTorch trainer via ``bat_training.permutation_adapter`` (or any other
    factory that returns a :class:`TrainerProtocol`).
    """

    factory = trainer_factory or create_permutation_trainer

    perm_test = PermutationTest(
        n_permutations=n_permutations,
        permutation_epochs=permutation_epochs,
        significance_level=significance_level,
        metrics_to_test=list(metrics_to_test) if metrics_to_test else list(DEFAULT_METRICS),
        save_null_distribution=True,
        verbose=verbose,
    )

    results = perm_test.run(
        observed_metrics=observed_metrics,
        trainer_factory=factory,
        **trainer_kwargs,
    )

    if output_dir is not None and generate_report:
        viz = PermutationVisualizer(results, cfg=cfg, output_dir=output_dir)
        viz.generate_full_report(
            show=False,
            export_json=True,
        )

    return results


__all__ = [
    "run_inference_test",
    "run_retrain_test",
]
