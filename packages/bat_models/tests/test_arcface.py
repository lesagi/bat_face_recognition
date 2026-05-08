"""Tests for ``ArcFaceModel`` and ``ArcFaceHead``."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from bat_models.arcface import ArcFaceModel  # noqa: E402
from bat_models.heads.arcface_head import ArcFaceHead  # noqa: E402


class _DummyBackbone(torch.nn.Module):
    """Tiny stand-in for ResNet50 so tests stay fast and offline."""

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


def test_arcface_model_forward_embedding_shape() -> None:
    model = ArcFaceModel(
        backbone=_DummyBackbone(),
        embedding_dim=512,
        num_classes=10,
    ).eval()
    x = torch.randn(4, 3, 112, 112)
    with torch.no_grad():
        emb = model.forward_embedding(x)
    assert emb.shape == (4, 512)
    norms = torch.norm(emb, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_arcface_model_forward_train_logits_shape() -> None:
    num_classes = 7
    model = ArcFaceModel(
        backbone=_DummyBackbone(), embedding_dim=128, num_classes=num_classes
    ).train()
    x = torch.randn(5, 3, 112, 112)
    labels = torch.randint(0, num_classes, (5,))
    logits = model.forward_train(x, labels)
    assert logits.shape == (5, num_classes)


def test_arcface_head_zero_margin_equals_cosine_logits() -> None:
    """With m = 0 the head is identical to plain scaled cosine logits."""
    head = ArcFaceHead(embedding_dim=32, num_classes=5, margin=0.0, scale=64.0)
    head.train()
    embeddings = torch.randn(8, 32)
    labels = torch.randint(0, 5, (8,))
    out_with_labels = head(embeddings, labels)
    out_no_labels = head(embeddings, None)
    assert torch.allclose(out_with_labels, out_no_labels, atol=1e-6)


def test_arcface_head_eval_mode_drops_margin() -> None:
    """In eval mode, even with m > 0 we expect plain cosine logits."""
    head = ArcFaceHead(embedding_dim=16, num_classes=4, margin=0.5, scale=30.0)
    head.eval()
    embeddings = torch.randn(3, 16)
    labels = torch.tensor([0, 1, 2])
    out = head(embeddings, labels)
    out_no_labels = head(embeddings, None)
    assert torch.allclose(out, out_no_labels, atol=1e-6)


def test_arcface_head_margin_changes_gt_logit() -> None:
    """With m > 0, the GT-class logit should differ from the cosine baseline."""
    torch.manual_seed(0)
    head = ArcFaceHead(embedding_dim=8, num_classes=3, margin=0.4, scale=10.0)
    head.train()
    emb = torch.randn(2, 8)
    labels = torch.tensor([0, 1])
    margin_logits = head(emb, labels)
    baseline = head(emb, None)  # plain cosine logits
    # GT logits should differ; non-GT should match.
    for i, lbl in enumerate(labels.tolist()):
        assert not math.isclose(margin_logits[i, lbl].item(), baseline[i, lbl].item(), abs_tol=1e-4)
        for j in range(3):
            if j == lbl:
                continue
            assert math.isclose(margin_logits[i, j].item(), baseline[i, j].item(), abs_tol=1e-4)
