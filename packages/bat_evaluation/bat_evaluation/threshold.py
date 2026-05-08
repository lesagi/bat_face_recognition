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

from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_curve


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


__all__ = ["optimize_youden_j"]
