"""Embedding-family losses: cross-entropy semantics, gradients, weights."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bat_losses import AdaFaceLoss, ArcFaceLoss, CosFaceLoss, SubCenterArcFaceLoss  # noqa: E402

EMBEDDING_LOSS_CLASSES = [
    ArcFaceLoss,
    AdaFaceLoss,
    CosFaceLoss,
    SubCenterArcFaceLoss,
]


@pytest.mark.parametrize("cls", EMBEDDING_LOSS_CLASSES)
def test_matches_cross_entropy_on_plain_logits(cls: type) -> None:
    """With m=0 (already absorbed by the head), the loss IS cross-entropy."""
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    labels = torch.randint(0, 5, (8,))
    ours = cls()(logits, labels)
    ref = torch.nn.CrossEntropyLoss()(logits, labels)
    assert torch.allclose(ours, ref, atol=1e-6)


@pytest.mark.parametrize("cls", EMBEDDING_LOSS_CLASSES)
def test_gradient_flows(cls: type) -> None:
    logits = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0])
    loss = cls()(logits, labels)
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("cls", EMBEDDING_LOSS_CLASSES)
def test_sample_weights_apply(cls: type) -> None:
    torch.manual_seed(1)
    logits = torch.randn(6, 4)
    labels = torch.randint(0, 4, (6,))
    fn = cls(reduction="sum")
    base = fn(logits, labels)
    weighted = fn(logits, labels, sample_weights=torch.full((6,), 2.0))
    assert torch.allclose(weighted, 2.0 * base, atol=1e-5)


@pytest.mark.parametrize("cls", EMBEDDING_LOSS_CLASSES)
def test_accepts_long_or_int_labels(cls: type) -> None:
    logits = torch.randn(3, 4)
    fn = cls()
    a = fn(logits, torch.tensor([0, 1, 2], dtype=torch.long))
    b = fn(logits, torch.tensor([0, 1, 2], dtype=torch.int32))
    assert torch.allclose(a, b)
