"""AdaFace head — Kim et al., CVPR 2022.

AdaFace adapts the angular margin to feature-norm-derived "image quality".
With ``g`` the standardised, clipped feature norm in ``[-1, 1]`` (Eq. 19 of
Kim 2022):

    g_angle = -m * g
    g_add   =  m * g + m          (note the constant ``+ m`` term)
    cos'(theta_y) = cos(theta_y + g_angle) - g_add
    final logit   = scale * cos'(...)

The constant ``+ m`` in ``g_add`` is essential: at ``g = 0`` (average-quality
sample) AdaFace must reduce to CosFace with additive margin ``m``, not to a
plain-softmax logit. The limit cases are: ``g = -1`` -> ArcFace
(``cos(theta + m)``), ``g = 0`` -> CosFace (``cos(theta) - m``), ``g = 1`` ->
negative angular margin with a shift.

The norm is standardised by exponential-moving-average ``batch_mean`` /
``batch_std`` updated per-batch and scaled/clipped by ``h``. Reference
implementation: https://github.com/mk-minchul/AdaFace

When called without labels (or in eval mode) the head returns plain cosine
logits scaled by ``scale``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional


class AdaFaceHead(nn.Module):
    """Adaptive-margin head with per-sample feature-norm modulation.

    Parameters
    ----------
    embedding_dim:
        Dim of the backbone embedding.
    num_classes:
        Number of training identities.
    margin:
        Base margin ``m`` (default ``0.4`` per Kim 2022).
    h:
        Half-width of the norm window (default ``0.333``).
    scale:
        Logit scale ``s`` (default ``64.0``).
    t_alpha:
        EMA momentum on the batch-mean of feature norms (default ``0.01``).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.4,
        h: float = 0.333,
        scale: float = 64.0,
        t_alpha: float = 0.01,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = float(margin)
        self.h = float(h)
        self.scale = float(scale)
        self.t_alpha = float(t_alpha)
        self.eps = float(eps)

        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)

        # Running stats for batch feature norm; persistent buffers so they
        # survive checkpoint round-trips.
        self.register_buffer("batch_mean", torch.tensor(20.0))
        self.register_buffer("batch_std", torch.tensor(100.0))

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return AdaFace logits, or plain cosine logits in eval / no-label mode."""
        # Per-sample feature norm and L2-normalised direction.
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp(min=self.eps)
        emb_norm = embeddings / norms
        w_norm = functional.normalize(self.weight, p=2, dim=1)

        cosine = functional.linear(emb_norm, w_norm).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None or not self.training or self.margin == 0.0:
            return cosine * self.scale

        # Update running mean/std of feature norms (no-grad).
        with torch.no_grad():
            mean_now = norms.mean().detach()
            std_now = norms.std().detach()
            self.batch_mean.mul_(1.0 - self.t_alpha).add_(self.t_alpha * mean_now)
            self.batch_std.mul_(1.0 - self.t_alpha).add_(self.t_alpha * std_now.clamp(min=self.eps))

        # Standardised feature norm in [-1, 1] window of width ``h``.
        margin_scaler = (norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = (margin_scaler * self.h).clamp(-1.0, 1.0)

        # g_angle / g_add are broadcast over classes (Kim 2022, Eq. 19).
        # g_add carries the constant ``+ m`` so that g=0 -> CosFace (margin m),
        # not plain softmax. Omitting it silently disables the average-quality
        # margin and makes AdaFace collapse toward softmax.
        g_angle = -self.margin * margin_scaler  # shape (B, 1)
        g_add = self.margin * (1.0 + margin_scaler)  # = m + m * margin_scaler

        # Apply margin only on the ground-truth class.
        one_hot = functional.one_hot(labels.long(), num_classes=self.num_classes).to(cosine.dtype)

        # cos(theta + g_angle) for the GT class, then subtract g_add.
        theta = torch.acos(cosine)
        theta_y = theta + g_angle  # broadcasts (B, 1) over class dim
        # Keep theta_y in valid range [0, pi].
        theta_y = theta_y.clamp(min=1e-7, max=math.pi - 1e-7)
        cos_theta_y = torch.cos(theta_y)

        modified = cos_theta_y - g_add  # subtract additive margin
        logits = one_hot * modified + (1.0 - one_hot) * cosine
        return logits * self.scale
