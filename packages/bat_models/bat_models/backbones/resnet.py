"""ResNet50 backbone (FC head removed) for embedding-family models.

SOTA deviation (documented intentionally)
------------------------------------------
The canonical ArcFace/AdaFace papers use a modified "IR"/"IR-SE" ResNet: a
3x3 stride-1 stem (no aggressive 7x7 stride-2 + maxpool downsampling) and an
"Option-E" embedding head (``BN -> Dropout(0.4) -> Flatten(C*7*7) ->
Linear(512) -> BN``) operating on the spatial feature map, with 112x112 inputs.

This project instead uses the stock torchvision ImageNet-pretrained ResNet-50
(7x7 stride-2 + maxpool stem, global average pool to 2048) followed by a
``BN -> Dropout(0.0) -> Linear(2048, 512) -> BN`` projection (see
``bat_models.arcface.ArcFaceModel``). This is a deliberate transfer-learning
choice for the small-data regime here, but it is a deviation from the cited
SOTA and interacts with input resolution: the torchvision stem downsamples
112x112 inputs aggressively (112 -> 56 -> 28 after stem+maxpool) before the
residual stages. A faithful IR/IR-SE backbone + Option-E head remains a
possible future option; until then this deviation should be stated explicitly
in the paper's Methods (the global-average-pool head is *not* the Option-E
flattened head used by Deng 2019 / Kim 2022).
"""

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
