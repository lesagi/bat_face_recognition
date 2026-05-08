"""Permutation Test for Statistical Significance of Model Performance.

Ported from ``app/statistical_tests/permutation_test.py``.  Pure ``sklearn`` /
``numpy`` -- the framework-agnostic core of the test.  Trainer hooks live in
:mod:`bat_stats.trainer_hooks` and :mod:`bat_stats.runner`; this module only
needs a ``TrainerProtocol``-shaped object passed in.

The test:
1. Records the observed metrics from a trained model.
2. Trains N times with permuted labels (or skips retraining via the inference
   variant in :mod:`bat_stats.inference`).
3. Builds a null distribution from permuted results.
4. Computes p-values via ``(count(null >= observed) + 1) / (n + 1)``.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_METRICS: tuple[str, ...] = ("f1", "accuracy", "precision", "recall")


@dataclass
class MetricResult:
    """Per-metric outcome of a permutation test."""

    observed: float
    null_mean: float
    null_std: float
    p_value: float
    significant: bool
    null_distribution: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "p_value": self.p_value,
            "significant": self.significant,
            "null_distribution": self.null_distribution,
        }


@dataclass
class PermutationTestResults:
    """Aggregate results across all metrics."""

    n_permutations: int
    significance_level: float
    metrics: dict[str, MetricResult]
    total_time_seconds: float
    permutation_epochs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_permutations": self.n_permutations,
            "significance_level": self.significance_level,
            "permutation_epochs": self.permutation_epochs,
            "total_time_seconds": self.total_time_seconds,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> PermutationTestResults:
        with open(path) as f:
            data = json.load(f)
        metrics: dict[str, MetricResult] = {}
        for name, metric_data in data["metrics"].items():
            metrics[name] = MetricResult(
                observed=metric_data["observed"],
                null_mean=metric_data["null_mean"],
                null_std=metric_data["null_std"],
                p_value=metric_data["p_value"],
                significant=metric_data["significant"],
                null_distribution=metric_data.get("null_distribution", []),
            )
        return cls(
            n_permutations=data["n_permutations"],
            significance_level=data["significance_level"],
            permutation_epochs=data["permutation_epochs"],
            total_time_seconds=data["total_time_seconds"],
            metrics=metrics,
        )

    def print_summary(self) -> None:
        bar = "=" * 70
        print(f"\n{bar}", flush=True)
        print(f"Permutation Test Results (n={self.n_permutations})", flush=True)
        print(bar, flush=True)
        header = (
            f"{'Metric':<12} {'Observed':>10} {'Mean(Null)':>12} "
            f"{'Std(Null)':>10} {'P-value':>10} {'Significant':<12}"
        )
        print(header, flush=True)
        print(
            f"{'-' * 12} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 12}",
            flush=True,
        )
        for name, result in self.metrics.items():
            sig_str = f"Yes (p<{self.significance_level})" if result.significant else "No"
            print(
                f"{name:<12} {result.observed:>10.4f} {result.null_mean:>12.4f} "
                f"{result.null_std:>10.4f} {result.p_value:>10.4f} {sig_str:<12}",
                flush=True,
            )
        print(bar, flush=True)
        print(f"Total time: {self.total_time_seconds:.1f} seconds", flush=True)
        print(f"{bar}\n", flush=True)


class PermutationTest:
    """Run a classical retrain-from-scratch permutation test.

    The ``trainer_factory`` argument in :meth:`run` should be a callable
    returning an object satisfying :class:`bat_stats.trainer_hooks.TrainerProtocol`.
    The actual PyTorch implementation of that protocol is deferred to Phase 2
    (``bat_training``); see ``bat_stats/trainer_hooks.py``.
    """

    def __init__(
        self,
        n_permutations: int = 100,
        permutation_epochs: int = 10,
        significance_level: float = 0.05,
        metrics_to_test: list[str] | None = None,
        save_null_distribution: bool = True,
        verbose: bool = True,
    ) -> None:
        self.n_permutations = n_permutations
        self.permutation_epochs = permutation_epochs
        self.significance_level = significance_level
        self.metrics_to_test: list[str] = list(metrics_to_test or DEFAULT_METRICS)
        self.save_null_distribution = save_null_distribution
        self.verbose = verbose

        self.null_distributions: dict[str, list[float]] = {
            metric: [] for metric in self.metrics_to_test
        }

    def compute_p_value(self, observed: float, null_distribution: np.ndarray) -> float:
        """One-sided permutation p-value with continuity correction.

        ``p = (count(null >= observed) + 1) / (n + 1)``.
        """

        n = len(null_distribution)
        count_greater_equal = int(np.sum(null_distribution >= observed))
        return float((count_greater_equal + 1) / (n + 1))

    def run(
        self,
        observed_metrics: dict[str, float],
        trainer_factory: Callable[..., Any],
        **trainer_kwargs: Any,
    ) -> PermutationTestResults:
        """Run the full retrain-style permutation test.

        ``trainer_factory(**trainer_kwargs)`` must return a TrainerProtocol-like
        object exposing ``train_and_evaluate() -> dict[str, float]`` and
        ``reset_for_new_permutation()``.

        TODO(phase-2): wire to ``bat_training`` PyTorch trainers once they exist.
        """

        start_time = time.time()
        self.null_distributions = {metric: [] for metric in self.metrics_to_test}

        if self.verbose:
            self._print_header()

        trainer = trainer_factory(
            permute_labels=True,
            num_epochs=self.permutation_epochs,
            verbose=self.verbose,
            **trainer_kwargs,
        )

        permutation_times: list[float] = []
        for i in range(self.n_permutations):
            perm_start = time.time()
            if self.verbose:
                self._print_perm_header(i, start_time, permutation_times)

            if i > 0:
                trainer.reset_for_new_permutation()

            metrics = trainer.train_and_evaluate()

            for metric in self.metrics_to_test:
                if metric in metrics:
                    self.null_distributions[metric].append(float(metrics[metric]))

            perm_time = time.time() - perm_start
            permutation_times.append(perm_time)

            if self.verbose:
                metrics_str = ", ".join(
                    f"{k}: {v:.4f}" for k, v in metrics.items() if k in self.metrics_to_test
                )
                print(
                    f"   Permutation {i + 1} metrics: {metrics_str} " f"(took {perm_time:.1f}s)",
                    flush=True,
                )
            gc.collect()

        results = self.run_from_null_distributions(observed_metrics, self.null_distributions)
        # Patch the timing fields the inference path can't fill in.
        results = PermutationTestResults(
            n_permutations=self.n_permutations,
            significance_level=self.significance_level,
            permutation_epochs=self.permutation_epochs,
            metrics=results.metrics,
            total_time_seconds=time.time() - start_time,
        )

        if self.verbose:
            results.print_summary()
        return results

    def run_from_null_distributions(
        self,
        observed_metrics: dict[str, float],
        null_distributions: dict[str, list[float]],
    ) -> PermutationTestResults:
        """Calculate results from pre-computed null distributions."""

        results_metrics: dict[str, MetricResult] = {}
        for metric in self.metrics_to_test:
            if metric not in observed_metrics or metric not in null_distributions:
                continue
            null_dist = np.asarray(null_distributions[metric], dtype=np.float64)
            if null_dist.size == 0:
                if self.verbose:
                    print(
                        f"Warning: empty null distribution for '{metric}'",
                        flush=True,
                    )
                continue
            observed = float(observed_metrics[metric])
            p_value = self.compute_p_value(observed, null_dist)
            results_metrics[metric] = MetricResult(
                observed=observed,
                null_mean=float(np.mean(null_dist)),
                null_std=float(np.std(null_dist)),
                p_value=p_value,
                significant=p_value < self.significance_level,
                null_distribution=(
                    list(null_distributions[metric]) if self.save_null_distribution else []
                ),
            )

        n_perms = len(next(iter(null_distributions.values()))) if null_distributions else 0
        return PermutationTestResults(
            n_permutations=n_perms,
            significance_level=self.significance_level,
            permutation_epochs=self.permutation_epochs,
            metrics=results_metrics,
            total_time_seconds=0.0,
        )

    # ------------------------------------------------------------------
    # logging helpers
    # ------------------------------------------------------------------
    def _print_header(self) -> None:
        bar = "=" * 70
        print(f"\n{bar}", flush=True)
        print("Starting Permutation Test", flush=True)
        print(bar, flush=True)
        print(f"Number of permutations: {self.n_permutations}", flush=True)
        print(f"Epochs per permutation: {self.permutation_epochs}", flush=True)
        print(f"Significance level: {self.significance_level}", flush=True)
        print(f"Metrics to test: {self.metrics_to_test}", flush=True)
        print(f"{bar}\n", flush=True)

    def _print_perm_header(self, i: int, start_time: float, permutation_times: list[float]) -> None:
        if permutation_times:
            avg_time = sum(permutation_times) / len(permutation_times)
            remaining = avg_time * (self.n_permutations - i)
            remaining_min = remaining / 60
            eta_str = (
                f" | ETA: {remaining_min:.1f} min"
                if remaining_min < 60
                else f" | ETA: {remaining_min / 60:.1f} hrs"
            )
        else:
            eta_str = ""
        elapsed_min = (time.time() - start_time) / 60
        print(
            f"\n--- Permutation {i + 1}/{self.n_permutations} "
            f"(elapsed: {elapsed_min:.1f} min{eta_str}) ---",
            flush=True,
        )


__all__ = [
    "DEFAULT_METRICS",
    "MetricResult",
    "PermutationTest",
    "PermutationTestResults",
]
