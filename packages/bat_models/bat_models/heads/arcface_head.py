"""ArcFace head — Deng et al., CVPR 2019.

Implements additive angular margin loss as a logit transformation:

    cos(theta_y) -> cos(theta_y + m)            for the ground-truth class
    cos(theta_j)  unchanged                      for j != y
    final logit = scale * cos(...)

When called without ``labels`` (or in eval mode) the head returns plain
cosine logits (no margin), which is what verification / retrieval needs.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional


class ArcFaceHead(nn.Module):
    """Additive angular margin head.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of the embedding produced by the backbone.
    num_classes:
        Number of identity classes seen during training.
    margin:
        Angular margin ``m`` in radians (default ``0.5`` per Deng 2019).
    scale:
        Scaling factor ``s`` applied to the cosine logits (default ``64.0``).
    easy_margin:
        If ``True``, applies the "easy margin" variant that avoids the
        ``cos(theta + m) > cos(pi - m)`` clamp. Default ``False``.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = float(margin)
        self.scale = float(scale)
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)

        # Precomputed constants for the margin transform.
        self._cos_m = math.cos(self.margin)
        self._sin_m = math.sin(self.margin)
        # Threshold beyond which cos(theta + m) is non-monotonic.
        self._th = math.cos(math.pi - self.margin)
        self._mm = math.sin(math.pi - self.margin) * self.margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits.

        - If ``labels`` is provided and the module is in training mode, the
          ground-truth class logit is replaced with ``cos(theta + m)`` and
          the result is multiplied by ``scale``.
        - Otherwise the plain cosine logits ``s * cos(theta)`` are returned.
        """
        # Cosine similarity between L2-normalised embeddings and class weights.
        emb_norm = functional.normalize(embeddings, p=2, dim=1)
        w_norm = functional.normalize(self.weight, p=2, dim=1)
        cosine = functional.linear(emb_norm, w_norm)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None or not self.training or self.margin == 0.0:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0))
        phi = cosine * self._cos_m - sine * self._sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self._th, phi, cosine - self._mm)

        one_hot = functional.one_hot(labels.long(), num_classes=self.num_classes).to(cosine.dtype)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale
