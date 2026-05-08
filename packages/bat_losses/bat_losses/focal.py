"""Binary Focal Loss for pair-mode (Siamese) training.

Port of ``app/siamese_training/focal_loss.py`` (Lin et al., ICCV 2017) to
PyTorch. With ``gamma=0`` and ``alpha=0.5`` this reduces to standard binary
cross-entropy.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class FocalLoss(nn.Module):
    """Binary focal loss for imbalanced pair classification.

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    Args:
        alpha: Weight for the positive class. Default ``0.75``.
        gamma: Focusing parameter; default ``2.0`` (TF parity).
        from_logits: If ``True``, applies sigmoid to ``model_output`` first.
        reduction: One of ``"none" | "mean" | "sum"``. Default ``"mean"``.
    """

    family: Literal["pair", "embedding"] = "pair"

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        from_logits: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be none|mean|sum, got {reduction!r}")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.from_logits = bool(from_logits)
        self.reduction = reduction

    def forward(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y_pred = model_output
        # Squeeze a trailing singleton if present (matches TF impl).
        if y_pred.ndim > 1 and y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(-1)
        y_true = labels.to(dtype=y_pred.dtype)

        p = torch.sigmoid(y_pred) if self.from_logits else y_pred

        # Match Keras epsilon exactly (1e-7).
        eps = 1e-7
        p = torch.clamp(p, min=eps, max=1.0 - eps)

        positive = y_true >= 0.5
        p_t = torch.where(positive, p, 1.0 - p)
        alpha_t = torch.where(
            positive,
            torch.full_like(p_t, self.alpha),
            torch.full_like(p_t, 1.0 - self.alpha),
        )

        focal_weight = torch.pow(1.0 - p_t, self.gamma)
        ce = -torch.log(p_t)
        loss = alpha_t * focal_weight * ce

        if sample_weights is not None:
            loss = loss * sample_weights.to(dtype=loss.dtype)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
