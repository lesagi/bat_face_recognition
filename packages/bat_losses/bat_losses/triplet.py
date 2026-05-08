"""Margin triplet loss for pair-mode training.

Triplet loss accepts a tensor of pre-computed pair similarities or distances:
``model_output`` is expected to have shape ``(N, 2)`` where column 0 is the
anchor-positive distance and column 1 is the anchor-negative distance, OR
shape ``(N, 3)`` carrying the three embedding vectors stacked. We support the
canonical form: ``model_output`` is a tuple-like 3-tensor stack
``(anchor, positive, negative)`` of shape ``(3, N, D)`` OR a pre-distance pair
``(d_ap, d_an)`` of shape ``(N, 2)``. ``labels`` is ignored (kept for protocol
compliance).

Default margin = ``0.2`` (Schroff et al., FaceNet 2015).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class TripletLoss(nn.Module):
    """Margin triplet loss.

    Args:
        margin: Margin enforced between anchor-positive and anchor-negative
            distances. Default ``0.2``.
        p: Norm degree for distance computation when receiving raw embeddings.
            Default ``2`` (Euclidean).
        reduction: One of ``"none" | "mean" | "sum"``. Default ``"mean"``.
    """

    family: Literal["pair", "embedding"] = "pair"

    def __init__(
        self,
        margin: float = 0.2,
        p: int = 2,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be none|mean|sum, got {reduction!r}")
        self.margin = float(margin)
        self.p = int(p)
        self.reduction = reduction

    def forward(
        self,
        model_output: torch.Tensor,
        labels: torch.Tensor,  # noqa: ARG002 - kept for protocol compliance
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if model_output.ndim == 2 and model_output.shape[-1] == 2:
            d_ap = model_output[:, 0]
            d_an = model_output[:, 1]
        elif model_output.ndim == 3 and model_output.shape[0] == 3:
            anchor, positive, negative = model_output[0], model_output[1], model_output[2]
            d_ap = torch.norm(anchor - positive, p=self.p, dim=-1)
            d_an = torch.norm(anchor - negative, p=self.p, dim=-1)
        else:
            raise ValueError(
                "TripletLoss expects shape (N,2) of (d_ap,d_an) or (3,N,D) "
                f"of (anchor,positive,negative); got {tuple(model_output.shape)}"
            )

        per_sample = torch.clamp(d_ap - d_an + self.margin, min=0.0)

        if sample_weights is not None:
            per_sample = per_sample * sample_weights.to(dtype=per_sample.dtype)

        if self.reduction == "mean":
            return per_sample.mean()
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample
