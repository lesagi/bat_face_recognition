"""Monotone score transformations for cosine-similarity verification scores.

Embedding-model verification scores are cosine similarities. On a
poorly-separated model they cluster within a few ULPs of 1.0 — the
discriminating bits between matching and non-matching pairs live at the
seventh decimal place or finer. Float64 preserves them (and downstream
metric code already uses float64), but the **values themselves** are still
visually crowded: an ROC curve plotted against raw cosine similarity is a
vertical wall at x=1.0.

This module provides *monotone* transformations that **spread** scores out
without changing the underlying ranking. Two important properties hold for
any monotone transformation ``f``:

1. ROC-AUC is invariant: ``roc_auc(y, f(s)) == roc_auc(y, s)``.
2. Thresholds map 1-to-1: any threshold ``t`` in raw space corresponds to a
   single ``f(t)`` in transformed space that produces the **same** binary
   predictions.

So these transforms do not change *what* the model has learned. They change
*how readable* the resulting curves are, and — incidentally — they make
float32 sufficient for downstream consumers that don't want to upgrade to
float64.

This module is intentionally **presentation-only**. It does not modify
``bat_core.Predictions`` or ``VerificationMetrics`` semantics. The
precision-of-record across the eval / permutation / metric pipeline remains
float64, set in ``bat_stats.inference.predictions_from_dataclass``. Treat
the helpers here as plotting and storage utilities.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

# A floor on ``1 - s`` for ``neg_log_1_minus``. Values of ``s`` within this
# distance of 1.0 are treated as ``s == 1.0 - _EPS``. 1e-12 is well inside
# float64 precision (~1e-16) and gives ``neg_log_1_minus(1.0) ≈ 27.6`` —
# large but finite, so plots and downstream calls don't trip on ``inf``.
_EPS = 1e-12


def neg_log_1_minus(
    s: Iterable[float] | NDArray[np.floating] | float,
    eps: float = _EPS,
) -> NDArray[np.float64]:
    """Spread cosine sims clustered near 1 into a wide log-scale range.

    Maps ``s -> -log(max(1 - s, eps))``. ``s = 1.0`` clamps to a finite max
    (``-log(eps)``); ``s = 0.0`` -> 0; ``s = -1.0`` -> ``-log(2) ≈ -0.69``.
    Strictly monotone increasing on ``s < 1``, so ROC-AUC and threshold
    semantics are preserved.

    Useful when plotting ROC curves for embedding-similarity scores: the
    raw curve sits as a vertical wall at x=1, whereas
    ``-log(1 - s)`` puts the operating points into a readable range.
    """
    arr = np.asarray(list(s) if not isinstance(s, np.ndarray) else s, dtype=np.float64)
    return -np.log(np.maximum(1.0 - arr, eps))


def inv_neg_log_1_minus(
    t: Iterable[float] | NDArray[np.floating] | float,
) -> NDArray[np.float64]:
    """Inverse of :func:`neg_log_1_minus`. ``t -> 1 - exp(-t)``.

    Round-trips up to the ``eps`` floor applied in the forward direction;
    values originating at exactly ``s = 1.0`` round-trip to ``1 - eps``.
    """
    arr = np.asarray(list(t) if not isinstance(t, np.ndarray) else t, dtype=np.float64)
    return 1.0 - np.exp(-arr)


def arccos_scale(
    s: Iterable[float] | NDArray[np.floating] | float,
) -> NDArray[np.float64]:
    """Angular distance between unit vectors. ``s -> arccos(clip(s, -1, 1))``.

    Strictly monotone *decreasing* — high cosine similarity (close to 1)
    maps to small angle (close to 0). This is the natural distance for
    L2-normalised embeddings: ``s = 1`` -> 0 rad; ``s = 0`` -> pi/2;
    ``s = -1`` -> pi.

    Because the transformation is *decreasing*, ROC-AUC computed on
    ``arccos_scale(s)`` against unchanged labels equals
    ``1 - roc_auc(y, s)``. Callers using this scale for ranking should
    negate the labels or flip the score direction accordingly.
    """
    arr = np.asarray(list(s) if not isinstance(s, np.ndarray) else s, dtype=np.float64)
    return np.arccos(np.clip(arr, -1.0, 1.0))


__all__ = [
    "arccos_scale",
    "inv_neg_log_1_minus",
    "neg_log_1_minus",
]
