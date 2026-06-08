"""Tests for ``AdaFaceModel`` and ``AdaFaceHead``."""

from __future__ import annotations

import math

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_adaface_model_forward_embedding_shape() -> None:
    model = AdaFaceModel(backbone=_DummyBackbone(), embedding_dim=512, num_classes=10).eval()
    x = torch.randn(4, 3, 112, 112)
    with torch.no_grad():
        emb = model.forward_embedding(x)
    assert emb.shape == (4, 512)


def test_adaface_head_h_zero_collapses_to_cosface() -> None:
    """``h = 0`` forces the standardised norm ``g`` to 0, so AdaFace reduces to
    CosFace: the ground-truth logit is ``s * (cos(theta) - m)`` (NOT plain
    cosine). This pins the constant ``+ m`` term in ``g_add`` (Kim 2022 Eq. 19);
    omitting it would make ``h = 0`` collapse to plain softmax instead.
    """
    torch.manual_seed(1)
    margin, scale = 0.4, 10.0
    head = AdaFaceHead(embedding_dim=32, num_classes=5, margin=margin, h=0.0, scale=scale)
    head.train()

    emb = torch.randn(6, 32) * 5.0
    labels = torch.randint(0, 5, (6,))

    # Reference cosine using the same weights/normalisation as the head.
    w_norm = torch.nn.functional.normalize(head.weight, p=2, dim=1)
    cos = (emb / emb.norm(dim=1, keepdim=True)) @ w_norm.t()
    cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    out = head(emb, labels)

    expected = cos * scale
    gt = torch.nn.functional.one_hot(labels, num_classes=5).bool()
    expected = expected.clone()
    expected[gt] = (cos[gt] - margin) * scale
    assert torch.allclose(out, expected, atol=1e-3)


def test_adaface_head_arcface_limit_at_low_norm() -> None:
    """When the standardised norm ``g`` clips to -1 (feature norm far below the
    running mean), AdaFace reduces to ArcFace: the GT logit is
    ``s * cos(theta + m)`` (with AdaFace's ``[eps, pi-eps]`` angle clamp).
    """
    torch.manual_seed(0)
    margin, scale = 0.4, 10.0
    head = AdaFaceHead(embedding_dim=16, num_classes=5, margin=margin, h=0.333, scale=scale)
    head.train()
    # Small std + norm far below mean drives ``g`` to the -1 clip.
    head.batch_mean.fill_(20.0)
    head.batch_std.fill_(1.0)

    directions = torch.randn(8, 16)
    directions = directions / directions.norm(dim=1, keepdim=True)
    emb = directions * 10.0
    labels = torch.randint(0, 5, (8,))

    w_norm = torch.nn.functional.normalize(head.weight, p=2, dim=1)
    cos = (emb / emb.norm(dim=1, keepdim=True)) @ w_norm.t()
    cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    out = head(emb, labels)

    theta = torch.acos(cos)
    theta_y = (theta + margin).clamp(min=1e-7, max=math.pi - 1e-7)
    arc = torch.cos(theta_y)
    expected = (cos * scale).clone()
    gt = torch.nn.functional.one_hot(labels, num_classes=5).bool()
    expected[gt] = arc[gt] * scale
    assert torch.allclose(out, expected, atol=1e-3)


def test_adaface_head_larger_h_changes_logits() -> None:
    """Increasing ``h`` from 0 to 0.5 should change the GT-class logits."""
    torch.manual_seed(0)
    weight = torch.randn(5, 16)

    def _build(h: float) -> AdaFaceHead:
        torch.manual_seed(0)
        head = AdaFaceHead(embedding_dim=16, num_classes=5, margin=0.4, h=h, scale=10.0)
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
