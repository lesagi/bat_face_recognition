"""Verification metrics for pair-wise (same / different) face recognition.

Direct lift of `app/siamese_training/auc_evaluator.py` (sklearn-only) plus
threshold optimization. No CSV I/O and no plotting -- inputs are
`bat_core.Predictions` (or anything reducible to `(y_true, y_score)`) and the
output is a `bat_core.VerificationMetrics` dataclass.

Behaviour preserved from the legacy module:
- ROC-AUC via `sklearn.metrics.roc_auc_score`.
- Optimal threshold via Youden's J (delegated to `bat_evaluation.threshold`).
- TAR@FAR computed at FAR = 1e-3 and 1e-4 from the full ROC curve.
- F1 / precision / recall / specificity at the optimal threshold.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from bat_core import Predictions, VerificationMetrics
from bat_evaluation.threshold import optimize_youden_j, threshold_at_far
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class ConfusionAtThreshold:
    """Confusion-matrix-derived statistics at a given decision threshold."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float  # a.k.a. sensitivity / TPR
    specificity: float  # 1 - FPR
    f1: float


def _to_arrays(
    y_true: Iterable[float] | NDArray[np.floating],
    y_score: Iterable[float] | NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y_true_arr = np.asarray(list(y_true), dtype=np.float64).ravel()
    y_score_arr = np.asarray(list(y_score), dtype=np.float64).ravel()
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError(
            f"y_true and y_score have different shapes: "
            f"{y_true_arr.shape} vs {y_score_arr.shape}"
        )
    return y_true_arr, y_score_arr


def compute_roc(
    y_true: Iterable[float] | NDArray[np.floating],
    y_score: Iterable[float] | NDArray[np.floating],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute ROC curve and ROC-AUC.

    Returns
    -------
    (roc_auc, fpr, tpr, thresholds): ROC curve from sklearn. ``roc_auc`` is
    NaN when the labels contain only one class (AUC is undefined).
    """
    y_true_arr, y_score_arr = _to_arrays(y_true, y_score)

    n_pos = int((y_true_arr == 1.0).sum())
    n_neg = int((y_true_arr == 0.0).sum())
    if n_pos == 0 or n_neg == 0:
        return (
            float("nan"),
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([np.inf, -np.inf]),
        )

    auc = float(roc_auc_score(y_true_arr, y_score_arr))
    fpr, tpr, thr = roc_curve(y_true_arr, y_score_arr)
    return auc, fpr, tpr, thr


def tar_at_far(
    fpr: NDArray[np.floating],
    tpr: NDArray[np.floating],
    target_far: float,
) -> float:
    """True acceptance rate at the largest FPR not exceeding ``target_far``.

    Standard biometrics convention: pick the most lenient operating point
    that still satisfies the FAR budget. Returns 0.0 if the ROC curve has no
    point with FPR <= target_far.
    """
    if target_far < 0.0 or target_far > 1.0:
        raise ValueError(f"target_far must be in [0, 1], got {target_far}")
    fpr_arr = np.asarray(fpr, dtype=np.float64)
    tpr_arr = np.asarray(tpr, dtype=np.float64)
    mask = fpr_arr <= target_far
    if not mask.any():
        return 0.0
    return float(tpr_arr[mask].max())


def confusion_at_threshold(
    y_true: Iterable[float] | NDArray[np.floating],
    y_score: Iterable[float] | NDArray[np.floating],
    threshold: float,
) -> ConfusionAtThreshold:
    """Compute the F1 confusion matrix at the given decision threshold."""
    y_true_arr, y_score_arr = _to_arrays(y_true, y_score)
    pred_pos = y_score_arr >= threshold
    pos = y_true_arr == 1.0
    neg = y_true_arr == 0.0

    tp = int((pred_pos & pos).sum())
    fp = int((pred_pos & neg).sum())
    fn = int((~pred_pos & pos).sum())
    tn = int((~pred_pos & neg).sum())

    n_pos = tp + fn
    n_neg = tn + fp

    recall = tp / n_pos if n_pos > 0 else 0.0
    specificity = tn / n_neg if n_neg > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ConfusionAtThreshold(
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
        f1=float(f1),
    )


def evaluate_predictions(predictions: Predictions) -> VerificationMetrics:
    """Compute verification metrics from a :class:`bat_core.Predictions`.

    The output is a :class:`bat_core.VerificationMetrics` dataclass containing:
    ROC-AUC, optimal Youden-J threshold, max Youden-J value, TAR@FAR(1e-3),
    TAR@FAR(1e-4).
    """
    y_true = np.asarray(predictions.y_true, dtype=np.float64)
    y_score = np.asarray(predictions.y_score, dtype=np.float64)

    auc, fpr, tpr, _thr = compute_roc(y_true, y_score)

    optimal_threshold, max_j = optimize_youden_j(y_true, y_score)
    tar_1e3 = tar_at_far(fpr, tpr, 1e-3)
    tar_1e4 = tar_at_far(fpr, tpr, 1e-4)
    thr_1e2, recall_1e2 = threshold_at_far(y_true, y_score, 1e-2)

    return VerificationMetrics(
        roc_auc=float(auc) if not np.isnan(auc) else float("nan"),
        youden_j=float(max_j),
        optimal_threshold=float(optimal_threshold),
        tar_at_far_1e3=float(tar_1e3),
        tar_at_far_1e4=float(tar_1e4),
        threshold_at_far_1e2=float(thr_1e2),
        recall_at_far_1e2=float(recall_1e2),
    )


def predictions_from_embedding(
    embedding: object,
    pair_indices: Iterable[tuple[int, int]] | None = None,
) -> Predictions:
    """Derive verification predictions from an :class:`bat_core.Embedding`.

    Computes pair-wise cosine similarity between the embedding rows. If
    ``pair_indices`` is ``None``, every unordered pair (i, j) with i < j is
    used, producing N*(N-1)/2 predictions. Labels are 1 if the two rows share
    the same identity, else 0.
    """
    import torch
    from bat_core import Embedding

    if not isinstance(embedding, Embedding):
        raise TypeError(f"expected bat_core.Embedding, got {type(embedding).__name__}")
    tensor = embedding.tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Embedding.tensor must be a torch.Tensor")

    feats = tensor.detach().to(dtype=torch.float32, device="cpu")
    feats = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = (feats @ feats.T).numpy()
    identities = list(embedding.identities)

    n = len(identities)
    if pair_indices is None:
        idx_i, idx_j = np.triu_indices(n, k=1)
        pairs_i: NDArray[np.int_] = idx_i
        pairs_j: NDArray[np.int_] = idx_j
    else:
        pairs = list(pair_indices)
        if not pairs:
            return Predictions(y_true=(), y_score=())
        pairs_arr = np.asarray(pairs, dtype=np.int64)
        if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
            raise ValueError("pair_indices must yield (i, j) tuples")
        pairs_i = pairs_arr[:, 0]
        pairs_j = pairs_arr[:, 1]

    scores = sim[pairs_i, pairs_j]
    ids_arr = np.asarray(identities, dtype=object)
    labels = (ids_arr[pairs_i] == ids_arr[pairs_j]).astype(np.int64)

    return Predictions(
        y_true=tuple(int(v) for v in labels),
        y_score=tuple(float(v) for v in scores),
    )


__all__ = [
    "ConfusionAtThreshold",
    "compute_roc",
    "confusion_at_threshold",
    "evaluate_predictions",
    "predictions_from_embedding",
    "tar_at_far",
]
