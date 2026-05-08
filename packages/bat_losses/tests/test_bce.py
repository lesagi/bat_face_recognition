"""BCELoss: gradient flow, perfect-prediction sanity, sample weights."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bat_losses import BCELoss  # noqa: E402


def test_bce_perfect_predictions_near_zero() -> None:
    preds = torch.tensor([0.999, 0.001, 0.999, 0.001])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = BCELoss()(preds, labels)
    assert loss.item() < 1e-2


def test_bce_gradient_flows() -> None:
    preds = torch.tensor([0.7, 0.3, 0.6], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0])
    loss = BCELoss()(preds, labels)
    loss.backward()
    assert preds.grad is not None
    assert preds.grad.abs().sum().item() > 0.0


def test_bce_sample_weights_change_loss() -> None:
    preds = torch.tensor([0.4, 0.6])
    labels = torch.tensor([1.0, 0.0])
    fn = BCELoss(reduction="sum")
    base = fn(preds, labels)
    weighted = fn(preds, labels, sample_weights=torch.tensor([2.0, 2.0]))
    assert torch.allclose(weighted, 2.0 * base)


def test_bce_from_logits_matches_sigmoid_then_bce() -> None:
    logits = torch.tensor([1.5, -0.5, 0.0])
    labels = torch.tensor([1.0, 0.0, 1.0])
    a = BCELoss(from_logits=True)(logits, labels)
    b = BCELoss(from_logits=False)(torch.sigmoid(logits), labels)
    assert torch.allclose(a, b, atol=1e-6)
