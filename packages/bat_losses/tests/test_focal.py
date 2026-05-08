"""FocalLoss: gamma=0,alpha=0.5 -> BCE; gradient flow; sample weights."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bat_losses import BCELoss, FocalLoss  # noqa: E402


def test_focal_gamma0_alpha_half_reduces_to_bce() -> None:
    preds = torch.tensor([0.2, 0.8, 0.5, 0.6])
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
    focal = FocalLoss(alpha=0.5, gamma=0.0)(preds, labels)
    bce = BCELoss()(preds, labels)
    # FocalLoss with alpha=0.5 multiplies BCE by 0.5; rescale before comparing.
    assert torch.allclose(focal * 2.0, bce, atol=1e-5)


def test_focal_gradient_flows() -> None:
    preds = torch.tensor([0.7, 0.3, 0.6], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0])
    loss = FocalLoss()(preds, labels)
    loss.backward()
    assert preds.grad is not None
    assert preds.grad.abs().sum().item() > 0.0


def test_focal_downweights_easy_examples() -> None:
    """A confident-correct example should contribute less under gamma>0."""
    easy = torch.tensor([0.99])
    hard = torch.tensor([0.51])
    label = torch.tensor([1.0])
    fn = FocalLoss(alpha=0.5, gamma=2.0, reduction="none")
    assert fn(easy, label).item() < fn(hard, label).item()


def test_focal_sample_weights_apply() -> None:
    preds = torch.tensor([0.3, 0.7])
    labels = torch.tensor([1.0, 0.0])
    fn = FocalLoss(reduction="sum")
    base = fn(preds, labels)
    weighted = fn(preds, labels, sample_weights=torch.tensor([2.0, 2.0]))
    assert torch.allclose(weighted, 2.0 * base, atol=1e-6)


def test_focal_squeezes_trailing_singleton() -> None:
    preds = torch.tensor([[0.7], [0.3]])
    labels = torch.tensor([1.0, 0.0])
    loss = FocalLoss()(preds, labels)
    assert loss.ndim == 0  # mean reduction => scalar
