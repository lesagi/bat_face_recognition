"""Internal helper for embedding-family margin losses.

ArcFace / AdaFace / CosFace / SubCenter all share the same loss-side shape:
the *head* applies the margin transform and emits logits; the *loss* is
``CrossEntropyLoss`` over those logits. We centralize the implementation here
so each public class can carry its own reference (Deng/Kim/Wang/Deng-2020) and
record its margin/scale defaults in the docstring while sharing one body.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class _MarginCrossEntropy(nn.Module):
    """Cross-entropy over margin-head logits with optional sample weights."""

    family: Literal["pair", "embedding"] = "embedding"

    def __init__(
        self,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be none|mean|sum, got {reduction!r}")
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if labels.dtype != torch.long:
            labels = labels.long()

        per_sample = nn.functional.cross_entropy(
            model_output,
            labels,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        if sample_weights is not None:
            per_sample = per_sample * sample_weights.to(dtype=per_sample.dtype)

        if self.reduction == "mean":
            return per_sample.mean()
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample
