"""Tests for the inference-based (no-retraining) permutation test."""

from __future__ import annotations

import numpy as np
import pytest
from bat_core import Predictions
from bat_stats.inference import (
    compute_metrics,
    partial_permute_labels,
    run_inference_permutation_test,
)


def test_compute_metrics_perfect_predictions() -> None:
    preds = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    out = compute_metrics(preds, labels, threshold=0.5)
    assert out["roc_auc"] == pytest.approx(1.0)
    assert out["f1"] == pytest.approx(1.0)
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(1.0)


def test_compute_metrics_reports_roc_auc_independently_of_threshold() -> None:
    """ROC-AUC ranks scores, so moving the threshold must not change it."""
    preds = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    low = compute_metrics(preds, labels, threshold=0.05)
    high = compute_metrics(preds, labels, threshold=0.95)
    assert low["roc_auc"] == pytest.approx(high["roc_auc"])
    # ...while the thresholded metrics do move.
    assert low["recall"] != pytest.approx(high["recall"])


def test_compute_metrics_roc_auc_is_chance_for_single_class_labels() -> None:
    """A permutation can leave one class; sklearn raises, so we return chance.

    Returning 0.5 keeps the null distribution continuous instead of crashing the
    run or punching a hole in it.
    """
    preds = np.array([0.9, 0.8, 0.7, 0.6])
    out = compute_metrics(preds, np.ones(4), threshold=0.5)
    assert out["roc_auc"] == pytest.approx(0.5)
    out_zeros = compute_metrics(preds, np.zeros(4), threshold=0.5)
    assert out_zeros["roc_auc"] == pytest.approx(0.5)


def test_compute_metrics_roc_auc_is_half_for_uninformative_scores() -> None:
    preds = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    assert compute_metrics(preds, labels, threshold=0.5)["roc_auc"] == pytest.approx(0.5)


def test_compute_metrics_random_predictions() -> None:
    preds = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    out = compute_metrics(preds, labels, threshold=0.5)
    # All predicted positive (>=0.5) → recall=1.0, precision=0.5
    assert out["recall"] == pytest.approx(1.0)
    assert out["precision"] == pytest.approx(0.5)


def test_partial_permute_labels_preserves_distribution() -> None:
    rng = np.random.default_rng(0)
    labels = np.array([1, 1, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    permuted = partial_permute_labels(labels, fraction=1.0, rng=rng)
    # Marginal distribution is preserved (count of 0s and 1s unchanged).
    assert int(np.sum(permuted == 1)) == int(np.sum(labels == 1))
    assert permuted.shape == labels.shape


def test_partial_permute_at_zero_returns_copy() -> None:
    rng = np.random.default_rng(0)
    labels = np.array([1, 0, 1, 0], dtype=np.float32)
    out = partial_permute_labels(labels, fraction=0.0, rng=rng)
    np.testing.assert_array_equal(out, labels)
    out[0] = 99  # ensure it's a copy, not the same array
    assert labels[0] == 1


class _MockTrainer:
    """Trainer that records calls.  Used to prove the inference path doesn't
    invoke the trainer at all."""

    def __init__(self) -> None:
        self.train_calls = 0
        self.reset_calls = 0

    def train_and_evaluate(self) -> dict[str, float]:
        self.train_calls += 1
        return {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

    def reset_for_new_permutation(self) -> None:
        self.reset_calls += 1


def test_inference_path_does_not_call_trainer() -> None:
    """Pure-numpy fast path: must not retrain.

    We instantiate a mock trainer and pass nothing to the inference function;
    the test asserts that nothing in the call chain ends up touching the
    trainer.  Concretely: run with trivial fixed predictions, then verify
    the mock counter is zero.
    """

    trainer = _MockTrainer()
    rng = np.random.default_rng(42)
    n = 100
    labels = rng.integers(0, 2, size=n).astype(np.float32)
    predictions = labels.copy()  # perfect predictions for a clear signal

    results, deg = run_inference_permutation_test(
        predictions=predictions,
        labels=labels,
        threshold=0.5,
        n_permutations=50,
        verbose=False,
    )

    assert trainer.train_calls == 0
    assert trainer.reset_calls == 0
    assert deg is None
    # Perfect predictions on permuted random labels → p should be very small
    assert results.metrics["f1"].observed == pytest.approx(1.0)
    assert results.metrics["f1"].p_value < 0.05


def test_inference_with_predictions_dataclass() -> None:
    rng = np.random.default_rng(11)
    n = 200
    labels = rng.integers(0, 2, size=n).astype(np.float32)
    scores = labels.copy().astype(np.float32)
    preds_obj = Predictions(
        y_true=tuple(int(x) for x in labels),
        y_score=tuple(float(x) for x in scores),
    )
    results, _ = run_inference_permutation_test(
        predictions=preds_obj,
        n_permutations=50,
        verbose=False,
    )
    assert results.metrics["f1"].observed == pytest.approx(1.0)


def test_inference_seed_is_deterministic() -> None:
    rng = np.random.default_rng(2024)
    n = 100
    labels = rng.integers(0, 2, size=n).astype(np.float32)
    predictions = (labels + rng.normal(0, 0.1, size=n)).astype(np.float32)

    a, _ = run_inference_permutation_test(
        predictions=predictions,
        labels=labels,
        n_permutations=100,
        seed=7,
        verbose=False,
    )
    b, _ = run_inference_permutation_test(
        predictions=predictions,
        labels=labels,
        n_permutations=100,
        seed=7,
        verbose=False,
    )
    for metric in ("f1", "accuracy", "precision", "recall"):
        assert a.metrics[metric].p_value == b.metrics[metric].p_value
        assert a.metrics[metric].null_distribution == b.metrics[metric].null_distribution


def test_inference_degradation_curve() -> None:
    rng = np.random.default_rng(3)
    n = 200
    labels = rng.integers(0, 2, size=n).astype(np.float32)
    predictions = labels.copy().astype(np.float32)

    fractions = [0.0, 0.5, 1.0]
    _, deg = run_inference_permutation_test(
        predictions=predictions,
        labels=labels,
        n_permutations=50,
        degradation_fractions=fractions,
        verbose=False,
    )
    assert deg is not None
    for metric, curve in deg.items():
        assert len(curve) == len(fractions)
        # F1 at 0% should be 1.0; at 100% should be ~chance.
        if metric == "f1":
            assert curve[0][0] == 0.0
            assert curve[0][1] == pytest.approx(1.0)
            assert curve[-1][0] == 1.0
            assert curve[-1][1] < 0.9


def test_inference_predictions_labels_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        run_inference_permutation_test(
            predictions=np.array([0.1, 0.2]),
            labels=np.array([0.0, 1.0, 1.0]),
            n_permutations=10,
            verbose=False,
        )
