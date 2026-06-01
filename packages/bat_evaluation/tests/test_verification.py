"""Verification-protocol tests: ROC/AUC, TAR@FAR, threshold optimisation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from bat_core import Predictions
from bat_evaluation import (
    compute_roc,
    confusion_at_threshold,
    evaluate_predictions,
    optimize_youden_j,
    tar_at_far,
    threshold_at_far,
)
from bat_evaluation.threshold import threshold_at_min_recall
from sklearn.metrics import roc_curve


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


def test_threshold_at_far_matches_tar_at_far():
    # threshold_at_far's recall return must equal tar_at_far for the same
    # score distribution -- they describe the same operating point.
    rng = np.random.default_rng(11)
    pos = rng.normal(0.7, 0.15, size=400)
    neg = rng.normal(0.3, 0.15, size=400)
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_score = np.concatenate([pos, neg])

    _auc, fpr_arr, tpr_arr, _thr = compute_roc(y_true, y_score)
    for target in (1e-2, 0.05, 0.1):
        _thr, recall = threshold_at_far(y_true, y_score, target)
        assert recall == pytest.approx(tar_at_far(fpr_arr, tpr_arr, target))


def test_threshold_at_min_recall_realises_at_least_min_recall():
    # Contract: at the returned threshold the realised recall must satisfy
    # the requested floor; precision should equal what the function returned.
    rng = np.random.default_rng(101)
    pos = rng.normal(0.7, 0.15, size=300)
    neg = rng.normal(0.3, 0.15, size=600)
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_score = np.concatenate([pos, neg])

    for min_recall in (0.5, 0.75, 0.9):
        thr, precision_returned = threshold_at_min_recall(y_true, y_score, min_recall)
        cm = confusion_at_threshold(y_true, y_score, thr)
        assert cm.recall >= min_recall - 1e-9, (
            f"min_recall={min_recall}: realised recall {cm.recall} below floor"
        )
        # Precision_recall_curve and confusion_at_threshold may differ by an
        # epsilon when scores are tied at the threshold (`>=` vs `>` semantics
        # in the two libraries). Accept a small absolute tolerance.
        assert cm.precision == pytest.approx(precision_returned, abs=0.05)


def test_threshold_at_min_recall_picks_best_precision():
    # Among operating points satisfying the recall floor, the chosen threshold
    # must achieve the maximum precision sklearn's PR curve would report.
    from sklearn.metrics import precision_recall_curve

    rng = np.random.default_rng(202)
    pos = rng.normal(0.65, 0.15, size=250)
    neg = rng.normal(0.35, 0.15, size=500)
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_score = np.concatenate([pos, neg])

    min_recall = 0.75
    _thr, precision_returned = threshold_at_min_recall(y_true, y_score, min_recall)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    eligible = precision[:-1][recall[:-1] >= min_recall]
    expected_best = float(eligible.max())
    assert precision_returned == pytest.approx(expected_best)


def test_threshold_at_min_recall_single_class_returns_default():
    # Mirror of optimize_youden_j: when only one class is present we can't
    # define the operating point, so we return the legacy (0.5, 0.0).
    y_true = np.zeros(10)
    y_score = np.random.default_rng(0).uniform(size=10)
    thr, precision = threshold_at_min_recall(y_true, y_score, 0.75)
    assert thr == 0.5
    assert precision == 0.0


def test_threshold_at_far_picks_best_tpr_within_budget():
    # With overlapping pos/neg distributions and budget=0, the eligible
    # set is the FPR=0 column of the ROC curve. The function should pick
    # the highest TPR within that column, not the trivial (0,0) anchor.
    # Here: at threshold >= 0.85 the FPR is 0 and the second-highest
    # positive (0.8) is captured -> TPR = 0.5.
    y_true = np.asarray([1.0, 1.0, 0.0])
    y_score = np.asarray([0.9, 0.5, 0.7])
    thr, recall = threshold_at_far(y_true, y_score, 0.0)
    assert recall == pytest.approx(0.5)
    # Threshold should be between 0.7 (negative) and 0.9 (highest positive).
    assert 0.7 < thr <= 0.9


def test_threshold_at_far_classifies_as_expected_at_returned_threshold():
    # The contract: predicting `score >= threshold_at_far(...)` produces a
    # confusion matrix whose FPR is <= target_far. Anchor against confusion_at_threshold.
    rng = np.random.default_rng(42)
    pos = rng.normal(0.7, 0.15, size=200)
    neg = rng.normal(0.3, 0.15, size=200)
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_score = np.concatenate([pos, neg])
    target = 1e-2

    thr, recall_returned = threshold_at_far(y_true, y_score, target)
    cm = confusion_at_threshold(y_true, y_score, thr)
    realised_fpr = cm.fp / (cm.fp + cm.tn) if (cm.fp + cm.tn) > 0 else 0.0
    assert realised_fpr <= target + 1e-9
    assert cm.recall == pytest.approx(recall_returned)


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
