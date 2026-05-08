"""Tests for ``AdaFaceModel`` and ``AdaFaceHead``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from bat_models.adaface import AdaFaceModel  # noqa: E402
from bat_models.heads.adaface_head import AdaFaceHead  # noqa: E402


class _DummyBackbone(torch.nn.Module):
    output_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(3, 64),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


def test_adaface_model_forward_embedding_shape() -> None:
    model = AdaFaceModel(
        backbone=_DummyBackbone(), embedding_dim=512, num_classes=10
    ).eval()
    x = torch.randn(4, 3, 112, 112)
    with torch.no_grad():
        emb = model.forward_embedding(x)
    assert emb.shape == (4, 512)


def test_adaface_head_h_zero_collapses_margin_to_zero() -> None:
    """``h = 0`` zeros the norm-modulated margin -> baseline cosine logits."""
    torch.manual_seed(1)
    head_no_h = AdaFaceHead(
        embedding_dim=32, num_classes=5, margin=0.4, h=0.0, scale=10.0
    )
    head_no_h.train()
    head_baseline = AdaFaceHead(
        embedding_dim=32, num_classes=5, margin=0.0, h=0.0, scale=10.0
    )
    head_baseline.train()
    # Force the two heads to share weights so the cosine logits agree.
    head_baseline.weight.data.copy_(head_no_h.weight.data)

    emb = torch.randn(6, 32) * 5.0
    labels = torch.randint(0, 5, (6,))
    out_no_h = head_no_h(emb, labels)
    out_baseline = head_baseline(emb, None)  # plain scaled cosine
    assert torch.allclose(out_no_h, out_baseline, atol=1e-4)


def test_adaface_head_larger_h_changes_logits() -> None:
    """Increasing ``h`` from 0 to 0.5 should change the GT-class logits."""
    torch.manual_seed(0)
    weight = torch.randn(5, 16)

    def _build(h: float) -> AdaFaceHead:
        torch.manual_seed(0)
        head = AdaFaceHead(
            embedding_dim=16, num_classes=5, margin=0.4, h=h, scale=10.0
        )
        head.weight.data.copy_(weight)
        head.train()
        return head

    head_zero = _build(0.0)
    head_pos = _build(0.5)

    emb = torch.randn(8, 16) * 5.0
    labels = torch.randint(0, 5, (8,))

    out_zero = head_zero(emb, labels)
    out_pos = head_pos(emb, labels)

    # Some entries must differ — the GT class margin scales with h.
    diff = (out_pos - out_zero).abs().max().item()
    assert diff > 1e-3


def test_adaface_head_eval_returns_cosine_logits() -> None:
    head = AdaFaceHead(embedding_dim=8, num_classes=3, margin=0.4, h=0.333)
    head.eval()
    emb = torch.randn(2, 8)
    labels = torch.tensor([0, 1])
    out_with_labels = head(emb, labels)
    out_no_labels = head(emb, None)
    assert torch.allclose(out_with_labels, out_no_labels, atol=1e-6)
