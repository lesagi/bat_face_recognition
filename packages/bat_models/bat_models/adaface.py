"""AdaFace face-recognition model: backbone + projection + AdaFaceHead.

AdaFace consumes the *unnormalised* embedding so the head can use the
feature norm as a quality proxy. The exported inference module returns the
L2-normalised embedding (as is standard for retrieval).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from bat_models.heads.adaface_head import AdaFaceHead


class AdaFaceModel(nn.Module):
    """Embedding-family model composing a backbone with an AdaFace head.

    Parameters mirror ``ArcFaceModel``; ``margin`` defaults to ``0.4``,
    ``h`` controls AdaFace's norm window.
    """

    family: Literal["pair", "embedding"] = "embedding"

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int = 512,
        num_classes: int = 1,
        margin: float = 0.4,
        h: float = 0.333,
        scale: float = 64.0,
        backbone_out_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        out_dim = backbone_out_dim
        if out_dim is None:
            out_dim = getattr(backbone, "output_dim", None)
        if out_dim is None:
            raise ValueError(
                "AdaFaceModel needs to know the backbone's output dim; "
                "either pass backbone_out_dim=... or expose "
                "backbone.output_dim."
            )

        self.projection = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.Dropout(p=0.0),
            nn.Linear(out_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        self.head = AdaFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            h=h,
            scale=scale,
        )

    # ------------------------------------------------------------------ #
    # FaceModel protocol                                                 #
    # ------------------------------------------------------------------ #
    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the L2-normalised embedding of shape ``(B, embedding_dim)``."""
        feat = self.backbone(x)
        emb = self.projection(feat)
        return torch.nn.functional.normalize(emb, p=2, dim=1)

    def forward_train(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Forward producing margin-modified logits of shape ``(B, num_classes)``."""
        feat = self.backbone(x)
        # AdaFace uses unnormalised embeddings: the head pulls quality from
        # the feature norm.
        emb = self.projection(feat)
        return self.head(emb, labels)

    def export_for_inference(self) -> nn.Module:
        """Return an ``nn.Module`` that maps image -> normalised embedding."""
        return _EmbeddingExport(self.backbone, self.projection)

    # ------------------------------------------------------------------ #
    # nn.Module                                                          #
    # ------------------------------------------------------------------ #
    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        if labels is None:
            return self.forward_embedding(x)
        return self.forward_train(x, labels)


class _EmbeddingExport(nn.Module):
    """Inference-only module: image -> L2-normalised embedding."""

    def __init__(self, backbone: nn.Module, projection: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.projection(feat)
        return torch.nn.functional.normalize(emb, p=2, dim=1)
