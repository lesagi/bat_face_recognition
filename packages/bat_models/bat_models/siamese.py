"""PyTorch port of ``app/siamese_core/network.py``.

The TF/Keras baseline is preserved verbatim for parity:

- Input: 105 x 105 RGB image (3 channels). The TF model used 3 channels
  (``shape=(105, 105, 3)``) and we mirror that here so Phase-4 parity holds.
- Conv block sequence (filters / kernel / pool):
    1. Conv 64 @ 10x10, ReLU -> MaxPool 2x2 / stride 2
    2. Conv 128 @ 7x7,  ReLU -> MaxPool 2x2 / stride 2
    3. Conv 128 @ 4x4,  ReLU -> MaxPool 2x2 / stride 2
    4. Conv 256 @ 4x4,  ReLU -> Flatten -> Dense 4096 sigmoid
- Pair head: L1 distance between the two embeddings -> Dense 1 sigmoid.

L2 regularisation in the TF version (``l2(1e-4)``) is implemented in PyTorch
by attaching a ``weight_decay``-equivalent attribute that the trainer applies
through the optimiser. We do not bake L2 into ``forward``; the trainer is the
right place to set ``weight_decay=1e-4`` for parity.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

SIAMESE_INPUT_EDGE_LENGTH: int = 105
SIAMESE_INPUT_CHANNELS: int = 3
SIAMESE_EMBEDDING_DIM: int = 4096
SIAMESE_L2_WEIGHT_DECAY: float = 1e-4


class _SiameseEmbedding(nn.Module):
    """The 4-conv + dense embedding tower.

    Channel ordering is PyTorch-native (B, C, H, W). The TF model used
    "valid" padding everywhere; we replicate that with ``padding=0``.
    Spatial sizes step:
        105 --conv10--> 96 --pool2--> 48
         48 --conv7--> 42 --pool2--> 21
         21 --conv4--> 18 --pool2-->  9
          9 --conv4-->  6
        flatten 256 * 6 * 6 = 9216 -> Dense(4096, sigmoid).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=SIAMESE_INPUT_CHANNELS,
            out_channels=64,
            kernel_size=10,
            stride=1,
            padding=0,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0)
        self.flatten = nn.Flatten()
        # Spatial map after conv4 is 6x6 with 256 channels.
        self.dense = nn.Linear(256 * 6 * 6, SIAMESE_EMBEDDING_DIM)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.sigmoid(self.dense(x))
        return x


class SiameseModel(nn.Module):
    """Pair-family model that ports the TF Siamese network to PyTorch.

    Satisfies ``bat_core.FaceModel`` (``family = "pair"``).

    - ``forward_embedding(x)`` returns a ``(B, 4096)`` embedding for a single
      batch of images.
    - ``forward_train(x_pair, labels)`` accepts ``x_pair`` of shape
      ``(B, 2, C, H, W)`` (or a tuple of two ``(B, C, H, W)`` tensors) and
      returns a similarity score in ``[0, 1]`` of shape ``(B, 1)``.
    - ``forward(...)`` is an alias for ``forward_train`` so the module is
      usable inside standard PyTorch training loops.
    """

    family: Literal["pair", "embedding"] = "pair"

    def __init__(self) -> None:
        super().__init__()
        self.embedding = _SiameseEmbedding()
        # The TF model's Dense(1, sigmoid) classifier on top of |a-b|.
        self.classifier = nn.Linear(SIAMESE_EMBEDDING_DIM, 1)
        # Recommended weight-decay value for parity with TF L2(1e-4).
        self.recommended_weight_decay: float = SIAMESE_L2_WEIGHT_DECAY

    # ------------------------------------------------------------------ #
    # FaceModel protocol                                                 #
    # ------------------------------------------------------------------ #
    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Single-image embedding. Input ``(B, 3, 105, 105)`` -> ``(B, 4096)``."""
        return self.embedding(x)

    def forward_train(
        self, x_pair: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute pair similarity score in ``[0, 1]``.

        ``labels`` is accepted to satisfy the FaceModel protocol but is not
        used by Siamese (the loss consumes labels directly).
        """
        del labels  # unused
        a, b = self._split_pair(x_pair)
        emb_a = self.embedding(a)
        emb_b = self.embedding(b)
        l1 = torch.abs(emb_a - emb_b)
        logits = self.classifier(l1)
        return torch.sigmoid(logits)

    def export_for_inference(self) -> nn.Module:
        """Return the embedding tower as the inference module.

        Verification at inference time uses pairwise distances on embeddings,
        not the trained sigmoid classifier head, so we expose only the
        embedding network.
        """
        self.embedding.eval()
        return self.embedding

    # ------------------------------------------------------------------ #
    # nn.Module                                                          #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x_pair: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_train(x_pair, labels)

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_pair(
        x_pair: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x_pair, tuple):
            a, b = x_pair
            return a, b
        if x_pair.ndim == 5 and x_pair.shape[1] == 2:
            return x_pair[:, 0], x_pair[:, 1]
        raise ValueError(
            "SiameseModel expected pair tensor of shape (B, 2, C, H, W) "
            f"or a 2-tuple of (B, C, H, W); got shape {tuple(x_pair.shape)}."
        )
