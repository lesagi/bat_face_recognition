"""Binary cross-entropy loss for pair-mode training with sample weights."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class BCELoss(nn.Module):
    """Binary cross-entropy with optional per-sample weighting.

    Wraps :class:`torch.nn.functional.binary_cross_entropy` so callers can pass
    ``sample_weights`` (broadcastable to the per-sample loss tensor).

    Args:
        from_logits: If ``True``, applies sigmoid first (uses BCE-with-logits
            for numerical stability). Default ``False``.
        reduction: One of ``"none" | "mean" | "sum"``. Default ``"mean"``.
    """

    family: Literal["pair", "embedding"] = "pair"

    def __init__(
        self,
        from_logits: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be none|mean|sum, got {reduction!r}")
        self.from_logits = bool(from_logits)
        self.reduction = reduction

    def forward(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y_pred = model_output
        if y_pred.ndim > 1 and y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(-1)
        y_true = labels.to(dtype=y_pred.dtype)

        if self.from_logits:
            per_sample = nn.functional.binary_cross_entropy_with_logits(
                y_pred, y_true, reduction="none"
            )
        else:
            per_sample = nn.functional.binary_cross_entropy(
                y_pred, y_true, reduction="none"
            )

        if sample_weights is not None:
            per_sample = per_sample * sample_weights.to(dtype=per_sample.dtype)

        if self.reduction == "mean":
            return per_sample.mean()
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample
