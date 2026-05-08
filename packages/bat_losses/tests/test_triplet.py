"""TripletLoss: distance/embedding inputs, gradient flow, margin behavior."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bat_losses import TripletLoss  # noqa: E402


def test_triplet_zero_when_negative_far_enough() -> None:
    # d_ap=0.1, d_an=1.0, margin=0.2 -> 0.1 - 1.0 + 0.2 = -0.7 -> 0
    pairs = torch.tensor([[0.1, 1.0]])
    loss = TripletLoss(margin=0.2)(pairs, torch.zeros(1))
    assert loss.item() == 0.0


def test_triplet_positive_when_violation() -> None:
    # d_ap=0.9, d_an=1.0, margin=0.2 -> 0.1 violation
    pairs = torch.tensor([[0.9, 1.0]])
    loss = TripletLoss(margin=0.2)(pairs, torch.zeros(1))
    assert abs(loss.item() - 0.1) < 1e-6


def test_triplet_from_embeddings_gradient_flows() -> None:
    anchor = torch.randn(4, 8, requires_grad=True)
    positive = torch.randn(4, 8, requires_grad=True)
    negative = torch.randn(4, 8, requires_grad=True)
    triplet = torch.stack([anchor, positive, negative], dim=0)  # (3, N, D)
    loss = TripletLoss(margin=0.5)(triplet, torch.zeros(4))
    loss.backward()
    assert anchor.grad is not None and anchor.grad.abs().sum().item() > 0.0


def test_triplet_sample_weights_apply() -> None:
    pairs = torch.tensor([[0.9, 1.0], [0.5, 0.6]])
    fn = TripletLoss(margin=0.2, reduction="sum")
    base = fn(pairs, torch.zeros(2))
    weighted = fn(pairs, torch.zeros(2), sample_weights=torch.tensor([3.0, 3.0]))
    assert torch.allclose(weighted, 3.0 * base, atol=1e-6)


def test_triplet_rejects_bad_shape() -> None:
    import pytest

    bad = torch.zeros(4, 5)
    with pytest.raises(ValueError):
        TripletLoss()(bad, torch.zeros(4))
