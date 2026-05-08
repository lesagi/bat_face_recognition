"""ResNet50 backbone (FC head removed) for embedding-family models."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class _ResNet50Backbone(nn.Module):
    """ResNet50 with the classifier head replaced by an Identity layer.

    Output shape: ``(B, 2048)``.
    """

    output_dim: int = 2048

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        net.fc = nn.Identity()  # type: ignore[assignment]
        self._net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._net(x)
        # nn.Identity() returns shape (B, 2048).
        return out


def resnet50_backbone(pretrained: bool = True) -> nn.Module:
    """Build a ResNet50 backbone whose ``forward`` outputs a 2048-d feature.

    Parameters
    ----------
    pretrained:
        If ``True`` (default), use torchvision's ImageNet1K_V2 weights.
    """
    return _ResNet50Backbone(pretrained=pretrained)
