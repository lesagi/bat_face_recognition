"""Inference-Based Permutation Test.

Ported from ``app/statistical_tests/inference_permutation_test.py``.

Instead of retraining N times (expensive), this module:
1. Takes a fixed set of model predictions on the test set.
2. Permutes the labels N times and recomputes metrics each time
   (pure numpy -- no training, no GPU).
3. Computes p-values with the same formula as the classical test.

The TF / file-loading wrapper from the legacy script is gone -- callers pass
in pre-computed ``predictions`` and ``labels`` numpy arrays (typically
extracted from a :class:`bat_core.Predictions` instance produced by
``bat_evaluation``).  The orchestration entry point in :mod:`bat_stats.runner`
stitches the two together.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from bat_core import Predictions
from bat_stats.permutation_test import (
    DEFAULT_METRICS,
    MetricResult,
    PermutationTest,
    PermutationTestResults,
)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute F1 / accuracy / precision / recall from binary predictions."""

    binary_preds = (predictions >= threshold).astype(np.float32)
    tp = float(np.sum((binary_preds == 1) & (labels == 1)))
    fp = float(np.sum((binary_preds == 1) & (labels == 0)))
    fn = float(np.sum((binary_preds == 0) & (labels == 1)))
    tn = float(np.sum((binary_preds == 0) & (labels == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "f1": float(f1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
    }


def partial_permute_labels(
    labels: np.ndarray, fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """Permute a *fraction* of labels while keeping the rest fixed.

    Preserves the marginal label distribution exactly.  Used both for the
    null distribution (fraction=1.0) and for the optional degradation curve.
    """

    if fraction <= 0.0:
        return labels.copy()
    n = len(labels)
    k = max(1, int(round(n * fraction)))
    result = labels.copy()
    indices = rng.choice(n, size=k, replace=False)
    subset = result[indices].copy()
    rng.shuffle(subset)
    result[indices] = subset
    return result


def predictions_from_dataclass(
    preds: Predictions,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a :class:`bat_core.Predictions` to numpy arrays.

    ``y_score`` is float64 because embedding-model cosine similarities
    routinely live within a few ULPs of 1.0 — the discriminating bits
    between match and non-match pairs are lost when cast to float32.
    ``bat_evaluation.verification`` already uses float64 for the same
    reason; aligning here keeps the permutation test consistent with
    the reported ROC-AUC and optimal threshold.
    """

    y_true = np.asarray(preds.y_true, dtype=np.float32)
    y_score = np.asarray(preds.y_score, dtype=np.float64)
    return y_score, y_true


def run_inference_permutation_test(
    predictions: np.ndarray | Predictions,
    labels: np.ndarray | None = None,
    *,
    threshold: float = 0.5,
    n_permutations: int = 1000,
    significance_level: float = 0.05,
    metrics_to_test: Sequence[str] | None = None,
    degradation_fractions: Sequence[float] | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[PermutationTestResults, dict[str, list[tuple[float, float]]] | None]:
    """Run an inference-based permutation test on cached predictions.

    Parameters
    ----------
    predictions
        Either a :class:`bat_core.Predictions` instance (in which case ``labels``
        is taken from ``y_true`` and ignored if also passed), or a 1-D numpy
        array of model scores.
    labels
        1-D array of ground-truth binary labels.  Required when ``predictions``
        is a numpy array.
    threshold
        Decision threshold.  Should typically come from the val-set Youden-J
        optimization in ``bat_evaluation``.
    n_permutations
        Number of label permutations for the null distribution.  Cheap (numpy
        only), so defaults to 1000.
    metrics_to_test
        Subset of {"f1", "accuracy", "precision", "recall"}.
    degradation_fractions
        If provided, also compute the degradation curve at each fraction.
    seed
        Seed for the RNG; same seed → reproducible null distribution.
    verbose
        Print progress.

    Returns
    -------
    (results, degradation_data)
        ``degradation_data`` is None if not requested.
    """

    if isinstance(predictions, Predictions):
        scores, ys = predictions_from_dataclass(predictions)
    else:
        scores = np.asarray(predictions, dtype=np.float64)
        if labels is None:
            raise ValueError("labels must be provided when predictions is an array")
        ys = np.asarray(labels, dtype=np.float32)

    if scores.shape != ys.shape:
        raise ValueError(f"predictions/labels shape mismatch: {scores.shape} vs {ys.shape}")

    metrics_list = list(metrics_to_test) if metrics_to_test else list(DEFAULT_METRICS)
    start = time.time()

    if verbose:
        bar = "=" * 70
        print(f"\n{bar}", flush=True)
        print("Inference-Based Permutation Test", flush=True)
        print(bar, flush=True)
        print(f"  Threshold:    {threshold:.4f}", flush=True)
        print(f"  Permutations: {n_permutations}", flush=True)
        print(f"  N samples:    {len(scores)}", flush=True)
        pos = int(np.sum(ys == 1))
        print(f"  Label dist.:  {pos} positive, {len(ys) - pos} negative", flush=True)
        print(f"{bar}\n", flush=True)

    observed_metrics = compute_metrics(scores, ys, threshold)
    if verbose:
        print("Observed metrics:", flush=True)
        for k, v in observed_metrics.items():
            print(f"  {k}: {v:.4f}", flush=True)

    rng = np.random.default_rng(seed=seed)
    null_distributions: dict[str, list[float]] = {m: [] for m in metrics_list}

    for i in range(n_permutations):
        permuted = partial_permute_labels(ys, fraction=1.0, rng=rng)
        perm_metrics = compute_metrics(scores, permuted, threshold)
        for metric in metrics_list:
            if metric in perm_metrics:
                null_distributions[metric].append(perm_metrics[metric])

        if verbose and (i + 1) % max(1, n_permutations // 10) == 0:
            print(f"  Permutation {i + 1}/{n_permutations}", flush=True)

    perm_test = PermutationTest(
        n_permutations=n_permutations,
        permutation_epochs=0,
        significance_level=significance_level,
        metrics_to_test=metrics_list,
        save_null_distribution=True,
        verbose=False,
    )

    results_metrics: dict[str, MetricResult] = {}
    for metric in metrics_list:
        if metric not in observed_metrics:
            continue
        observed = observed_metrics[metric]
        null_dist = np.array(null_distributions[metric], dtype=np.float64)
        p_value = perm_test.compute_p_value(observed, null_dist)
        results_metrics[metric] = MetricResult(
            observed=observed,
            null_mean=float(np.mean(null_dist)),
            null_std=float(np.std(null_dist)),
            p_value=p_value,
            significant=p_value < significance_level,
            null_distribution=null_distributions[metric],
        )

    total_time = time.time() - start
    results = PermutationTestResults(
        n_permutations=n_permutations,
        significance_level=significance_level,
        permutation_epochs=0,
        metrics=results_metrics,
        total_time_seconds=total_time,
    )

    degradation_data: dict[str, list[tuple[float, float]]] | None = None
    if degradation_fractions is not None:
        degradation_data = {m: [] for m in metrics_list}
        n_repeats = min(50, max(10, n_permutations // 20))
        if verbose:
            print(
                f"\nComputing degradation curve "
                f"({len(list(degradation_fractions))} fractions, "
                f"{n_repeats} repeats each)...",
                flush=True,
            )

        for frac in degradation_fractions:
            frac_metrics: dict[str, list[float]] = {m: [] for m in metrics_list}
            for _ in range(n_repeats):
                permuted = partial_permute_labels(ys, fraction=float(frac), rng=rng)
                m_vals = compute_metrics(scores, permuted, threshold)
                for metric in metrics_list:
                    if metric in m_vals:
                        frac_metrics[metric].append(m_vals[metric])
            for metric in metrics_list:
                mean_val = float(np.mean(frac_metrics[metric])) if frac_metrics[metric] else 0.0
                degradation_data[metric].append((float(frac), mean_val))
            if verbose:
                vals_str = ", ".join(
                    f"{m}: {np.mean(frac_metrics[m]):.4f}" for m in metrics_list if frac_metrics[m]
                )
                print(f"  fraction={frac:.2f}: {vals_str}", flush=True)

    if verbose:
        results.print_summary()

    return results, degradation_data


__all__ = [
    "compute_metrics",
    "partial_permute_labels",
    "predictions_from_dataclass",
    "run_inference_permutation_test",
]
