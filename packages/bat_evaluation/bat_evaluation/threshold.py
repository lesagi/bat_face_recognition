"""Threshold optimization via Youden's J statistic.

Ported from `app/siamese_training/threshold_optimizer.py`. The public surface
is intentionally minimal: a single function that returns the threshold that
maximizes Youden's J = TPR - FPR, and that J value.

The implementation uses the ROC curve thresholds produced by
``sklearn.metrics.roc_curve`` rather than a fixed-grid search. This is both
more accurate (the grid in the legacy code can miss the true optimum when the
score distribution is concentrated away from a uniform 0..1 range) and is
exactly what sklearn users would write -- so the test fixture can verify
against sklearn directly.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_curve, roc_curve


def optimize_youden_j(
    y_true: Iterable[float] | NDArray[np.floating],
    y_score: Iterable[float] | NDArray[np.floating],
) -> tuple[float, float]:
    """Return ``(optimal_threshold, max_youden_j)``.

    Youden's J = sensitivity + specificity - 1 = TPR - FPR. The optimal
    threshold is the one for which J is maximized along the ROC curve. When
    only a single class is present in ``y_true``, J is undefined and we
    return ``(0.5, 0.0)`` for backwards compatibility with the legacy
    behaviour.
    """
    y_true_arr = np.asarray(list(y_true), dtype=np.float64).ravel()
    y_score_arr = np.asarray(list(y_score), dtype=np.float64).ravel()
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(
            f"y_true and y_score have different shapes: "
            f"{y_true_arr.shape} vs {y_score_arr.shape}"
        )

    n_pos = int((y_true_arr == 1.0).sum())
    n_neg = int((y_true_arr == 0.0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    fpr, tpr, thresholds = roc_curve(y_true_arr, y_score_arr)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    best_threshold = float(thresholds[best_idx])
    best_j = float(j[best_idx])

    # sklearn prepends an artificial threshold equal to ``max(score)+1`` so the
    # ROC curve starts at (0, 0). That synthetic point has ``threshold = inf``
    # in some sklearn versions; clamp such artefacts to a finite value matching
    # the largest real score, which preserves the legacy "<= 1.0" expectation.
    if not np.isfinite(best_threshold):
        finite_thresholds = thresholds[np.isfinite(thresholds)]
        best_threshold = float(finite_thresholds.max()) if finite_thresholds.size else 1.0

    return best_threshold, best_j


def threshold_at_far(
    y_true: Iterable[float] | NDArray[np.floating],
    y_score: Iterable[float] | NDArray[np.floating],
    target_far: float,
) -> tuple[float, float]:
    """Return ``(threshold, recall)`` at the most lenient FPR <= target_far.

    Biometric verification convention: budget a maximum false-alarm rate
    and report the best true-acceptance rate that respects it. Returns
    ``(inf, 0.0)`` if no operating point satisfies the budget (and
    ``(0.5, 0.0)`` when ``y_true`` contains a single class, mirroring
    ``optimize_youden_j``).
    """
    if not 0.0 <= target_far <= 1.0:
        raise ValueError(f"target_far must be in [0, 1], got {target_far}")

    y_true_arr = np.asarray(list(y_true), dtype=np.float64).ravel()
    y_score_arr = np.asarray(list(y_score), dtype=np.float64).ravel()
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(
            f"y_true and y_score have different shapes: "
            f"{y_true_arr.shape} vs {y_score_arr.shape}"
        )

    n_pos = int((y_true_arr == 1.0).sum())
    n_neg = int((y_true_arr == 0.0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    fpr, tpr, thresholds = roc_curve(y_true_arr, y_score_arr)
    mask = fpr <= target_far
    if not mask.any():
        return float("inf"), 0.0

    # Among points respecting the budget, pick the highest TPR. Ties: pick
    # the lowest threshold (most lenient — matches tar_at_far convention).
    candidates = np.where(mask)[0]
    best_idx = int(candidates[np.argmax(tpr[candidates])])
    threshold = float(thresholds[best_idx])
    recall = float(tpr[best_idx])

    # sklearn synthesises a leading threshold = max(score) + 1 / np.inf to
    # anchor the ROC curve at (0, 0). If we ended up there, fall back to the
    # largest real threshold so the value is usable downstream.
    if not np.isfinite(threshold):
        finite = thresholds[np.isfinite(thresholds)]
        threshold = float(finite.max()) if finite.size else 1.0

    return threshold, recall


def threshold_at_min_recall(
    y_true: Iterable[float] | NDArray[np.floating],
    y_score: Iterable[float] | NDArray[np.floating],
    min_recall: float,
) -> tuple[float, float]:
    """Return ``(threshold, precision)`` at the strictest threshold with recall >= min_recall.

    Biometric "TPR-floor" framing -- the mirror of :func:`threshold_at_far`.
    Caller declares a minimum recall they will accept; we pick the operating
    point that respects the floor and maximises precision. Ties on precision
    are broken by picking the lower threshold (i.e., the more lenient point
    that still hits the recall floor, matching ``threshold_at_far``'s
    "most lenient" convention).

    Returns ``(0.5, 0.0)`` when ``y_true`` is single-class (consistent with
    ``optimize_youden_j``). Returns ``(-inf, 0.0)`` when no operating point
    achieves the recall floor (e.g., ``min_recall = 1.0`` on a degenerate
    classifier).
    """
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError(f"min_recall must be in [0, 1], got {min_recall}")

    y_true_arr = np.asarray(list(y_true), dtype=np.float64).ravel()
    y_score_arr = np.asarray(list(y_score), dtype=np.float64).ravel()
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(
            f"y_true and y_score have different shapes: "
            f"{y_true_arr.shape} vs {y_score_arr.shape}"
        )

    n_pos = int((y_true_arr == 1.0).sum())
    n_neg = int((y_true_arr == 0.0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    # ``precision_recall_curve`` returns (precision, recall, thresholds) where
    # precision/recall have length N+1 and thresholds has length N -- the last
    # (precision, recall) is the "predict everything negative" point at
    # (1.0, 0.0), which has no associated threshold. Discard it so all three
    # arrays align.
    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_score_arr)
    precision = precision[:-1]
    recall = recall[:-1]

    mask = recall >= min_recall
    if not mask.any():
        return float("-inf"), 0.0

    # ``precision_recall_curve`` returns thresholds sorted ascending. Among
    # eligible points we pick the one with max precision; ``np.argmax`` returns
    # the first occurrence -> lowest threshold among precision ties, matching
    # ``threshold_at_far``'s "most lenient" convention.
    candidates = np.where(mask)[0]
    best_in_candidates = int(np.argmax(precision[candidates]))
    best_idx = int(candidates[best_in_candidates])
    threshold = float(thresholds[best_idx])
    realised_precision = float(precision[best_idx])

    return threshold, realised_precision


__all__ = ["optimize_youden_j", "threshold_at_far", "threshold_at_min_recall"]
