"""Tests for ``bat_models.SiameseModel``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from bat_models.siamese import (  # noqa: E402
    SIAMESE_EMBEDDING_DIM,
    SIAMESE_INPUT_CHANNELS,
    SIAMESE_INPUT_EDGE_LENGTH,
    SiameseModel,
)


def _make_input(batch: int = 2) -> torch.Tensor:
    return torch.randn(
        batch,
        SIAMESE_INPUT_CHANNELS,
        SIAMESE_INPUT_EDGE_LENGTH,
        SIAMESE_INPUT_EDGE_LENGTH,
    )


def test_forward_embedding_shape() -> None:
    model = SiameseModel().eval()
    x = _make_input(batch=4)
    with torch.no_grad():
        emb = model.forward_embedding(x)
    assert emb.shape == (4, SIAMESE_EMBEDDING_DIM)


def test_forward_train_returns_sigmoid() -> None:
    model = SiameseModel().eval()
    a = _make_input(batch=3)
    b = _make_input(batch=3)
    with torch.no_grad():
        sim = model.forward_train((a, b), labels=None)
    assert sim.shape == (3, 1)
    assert torch.all(sim >= 0.0)
    assert torch.all(sim <= 1.0)


def test_forward_train_pair_tensor() -> None:
    """Stacked (B, 2, C, H, W) input should also work."""
    model = SiameseModel().eval()
    a = _make_input(batch=2)
    b = _make_input(batch=2)
    pair = torch.stack([a, b], dim=1)
    assert pair.shape == (2, 2, 3, 105, 105)
    with torch.no_grad():
        sim = model.forward_train(pair, labels=None)
    assert sim.shape == (2, 1)
    assert torch.all((sim >= 0.0) & (sim <= 1.0))


def test_export_for_inference_returns_embedding() -> None:
    model = SiameseModel().eval()
    inf = model.export_for_inference()
    x = _make_input(batch=1)
    with torch.no_grad():
        emb = inf(x)
    assert emb.shape == (1, SIAMESE_EMBEDDING_DIM)


def test_family_attribute() -> None:
    model = SiameseModel()
    assert model.family == "pair"
