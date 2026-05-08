"""ArcFace face-recognition model: backbone + projection + ArcFaceHead."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from bat_models.heads.arcface_head import ArcFaceHead


class ArcFaceModel(nn.Module):
    """Embedding-family model composing a backbone with an ArcFace head.

    Parameters
    ----------
    backbone:
        ``nn.Module`` whose ``forward`` returns a ``(B, backbone_out_dim)``
        feature vector. ``backbone_out_dim`` is read from the attribute
        ``backbone.output_dim`` if present, else inferred lazily on the first
        forward (None default = lazy).
    embedding_dim:
        Dim of the embedding consumed by the head (default ``512``).
    num_classes:
        Number of identity classes used during margin training.
    margin / scale:
        Forwarded to ``ArcFaceHead``.
    backbone_out_dim:
        Optional explicit output dim of the backbone; if ``None`` we read
        ``backbone.output_dim`` (the convention used by ``resnet50_backbone``).
    """

    family: Literal["pair", "embedding"] = "embedding"

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int = 512,
        num_classes: int = 1,
        margin: float = 0.5,
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
                "ArcFaceModel needs to know the backbone's output dim; "
                "either pass backbone_out_dim=... or expose "
                "backbone.output_dim."
            )

        # BN -> Linear -> BN projection block, standard for ArcFace pipelines.
        self.projection = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.Dropout(p=0.0),
            nn.Linear(out_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
        self.head = ArcFaceHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
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
