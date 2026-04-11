"""
Post-training threshold optimization for Siamese network predictions.

Instead of using a fixed 0.5 threshold for binary decisions, this module
finds the optimal threshold that maximizes Youden's J statistic
(sensitivity + specificity - 1) on the validation/test set.
"""

import numpy as np
from typing import Dict, Tuple


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    num_thresholds: int = 200,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the decision threshold that maximizes Youden's J statistic.

    J = sensitivity + specificity - 1
      = TPR - FPR
      = TPR + TNR - 1

    This balances true-positive rate and true-negative rate, producing
    the threshold where the ROC curve is farthest from the diagonal.

    Args:
        y_true: Ground-truth binary labels (1.0 = same pair, 0.0 = different pair).
        y_pred_proba: Model output probabilities (continuous, 0-1).
        num_thresholds: Number of candidate thresholds to evaluate.

    Returns:
        Tuple of (optimal_threshold, metrics_dict) where metrics_dict contains
        sensitivity, specificity, precision, f1, and youden_j at the optimal point.
    """
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float32).ravel()

    thresholds = np.linspace(0.0, 1.0, num_thresholds + 2)[1:-1]

    best_j = -1.0
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {}

    positives = y_true == 1.0
    negatives = y_true == 0.0
    num_pos = positives.sum()
    num_neg = negatives.sum()

    if num_pos == 0 or num_neg == 0:
        return 0.5, {
            "sensitivity": 0.0, "specificity": 0.0,
            "precision": 0.0, "f1": 0.0, "youden_j": 0.0,
        }

    for t in thresholds:
        pred_pos = y_pred_proba >= t

        tp = (pred_pos & positives).sum()
        fp = (pred_pos & negatives).sum()
        fn = (~pred_pos & positives).sum()

        sensitivity = tp / num_pos
        specificity = 1.0 - (fp / num_neg)

        j = sensitivity + specificity - 1.0

        if j > best_j:
            best_j = j
            best_threshold = float(t)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
            best_metrics = {
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "f1": float(f1),
                "youden_j": float(j),
            }

    return best_threshold, best_metrics
