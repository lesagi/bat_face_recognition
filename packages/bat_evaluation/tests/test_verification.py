"""Verification-protocol tests: ROC/AUC, TAR@FAR, threshold optimisation."""
from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_curve

from bat_core import Predictions
from bat_evaluation import (
    compute_roc,
    confusion_at_threshold,
    evaluate_predictions,
    optimize_youden_j,
    tar_at_far,
)


def test_perfectly_separable_scores_give_auc_one():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.7, 1.0, size=200)
    neg = rng.uniform(0.0, 0.3, size=200)
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_score = np.concatenate([pos, neg])

    preds = Predictions(
        y_true=tuple(int(v) for v in y_true),
        y_score=tuple(float(v) for v in y_score),
    )
    metrics = evaluate_predictions(preds)

    assert metrics.roc_auc == pytest.approx(1.0)
    # Optimal threshold should fall in the gap between the two distributions.
    # sklearn reports the smallest positive sample as the optimum which can
    # sit a hair above the nominal upper bound of the negative range.
    assert max(neg) <= metrics.optimal_threshold <= min(pos) + 1e-9
    assert metrics.youden_j == pytest.approx(1.0)
    # Perfect separation -> TAR@FAR=0 is exactly 1.0.
    assert metrics.tar_at_far_1e3 == pytest.approx(1.0)
    assert metrics.tar_at_far_1e4 == pytest.approx(1.0)


def test_random_scores_give_auc_near_half():
    rng = np.random.default_rng(42)
    n = 5000
    y_true = rng.integers(0, 2, size=n).astype(np.float64)
    y_score = rng.uniform(0.0, 1.0, size=n)
    auc, _, _, _ = compute_roc(y_true, y_score)
    assert abs(auc - 0.5) < 0.05


def test_tar_at_far_matches_sklearn_directly():
    rng = np.random.default_rng(7)
    pos = rng.normal(0.7, 0.15, size=400)
    neg = rng.normal(0.3, 0.15, size=400)
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_score = np.concatenate([pos, neg])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    target_far = 1e-2
    expected = float(tpr[fpr <= target_far].max())
    auc, our_fpr, our_tpr, _ = compute_roc(y_true, y_score)
    assert tar_at_far(our_fpr, our_tpr, target_far) == pytest.approx(expected)


def test_tar_at_far_zero_floor_when_no_point_satisfies_budget():
    # FAR = 0 is rarely achievable; the convention is "0 if no operating
    # point satisfies the budget".
    fpr = np.asarray([0.5, 0.7, 1.0])
    tpr = np.asarray([0.5, 0.8, 1.0])
    assert tar_at_far(fpr, tpr, 0.1) == 0.0


def test_threshold_reproduces_sklearn_youden_j():
    rng = np.random.default_rng(123)
    n = 1000
    pos = rng.normal(0.65, 0.2, size=n // 2)
    neg = rng.normal(0.35, 0.2, size=n // 2)
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_score = np.concatenate([pos, neg])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    sk_idx = int(np.argmax(j))
    sk_threshold = float(thresholds[sk_idx])
    sk_max_j = float(j[sk_idx])

    our_threshold, our_j = optimize_youden_j(y_true, y_score)

    if math.isfinite(sk_threshold):
        assert our_threshold == pytest.approx(sk_threshold)
    else:
        # sklearn synthetic threshold; ours clamps to a finite value.
        assert math.isfinite(our_threshold)
    assert our_j == pytest.approx(sk_max_j)


def test_threshold_single_class_returns_default():
    y_true = np.zeros(10, dtype=np.float64)
    y_score = np.linspace(0, 1, 10)
    threshold, j = optimize_youden_j(y_true, y_score)
    assert threshold == 0.5
    assert j == 0.0


def test_confusion_at_threshold_matches_manual_count():
    y_true = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.float64)
    y_score = np.asarray([0.9, 0.4, 0.8, 0.7, 0.2, 0.1], dtype=np.float64)

    cm = confusion_at_threshold(y_true, y_score, threshold=0.5)
    # threshold 0.5: predicted positive when score >= 0.5
    # tp = pos & pred_pos: 0.9, 0.8 -> 2
    # fn = pos & pred_neg: 0.4 -> 1
    # fp = neg & pred_pos: 0.7 -> 1
    # tn = neg & pred_neg: 0.2, 0.1 -> 2
    assert cm.tp == 2
    assert cm.fn == 1
    assert cm.fp == 1
    assert cm.tn == 2
    assert cm.recall == pytest.approx(2 / 3)
    assert cm.specificity == pytest.approx(2 / 3)
    assert cm.precision == pytest.approx(2 / 3)
    assert cm.f1 == pytest.approx(2 / 3)


def test_evaluate_predictions_handles_single_class_gracefully():
    preds = Predictions(
        y_true=(1, 1, 1, 1),
        y_score=(0.1, 0.5, 0.7, 0.9),
    )
    m = evaluate_predictions(preds)
    assert math.isnan(m.roc_auc)
    assert m.youden_j == 0.0
    assert m.optimal_threshold == 0.5


def test_compute_roc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_roc(np.zeros(3), np.zeros(4))
