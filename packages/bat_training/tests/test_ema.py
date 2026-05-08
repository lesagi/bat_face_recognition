"""Tests for ExponentialMovingAverage."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bat_training.ema import ExponentialMovingAverage


def _tiny_model() -> torch.nn.Module:
    m = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.zeros_like(m.weight))
    return m


def test_ema_update_rule() -> None:
    """``shadow = decay * shadow + (1 - decay) * param`` (one step)."""
    m = _tiny_model()
    # Initial weights = 0, EMA shadow snapshot = 0.
    ema = ExponentialMovingAverage(m, decay=0.9)
    # Bump the live weights to all-ones and update the EMA.
    with torch.no_grad():
        m.weight.copy_(torch.ones_like(m.weight))
    ema.update(m)
    # Expected: 0.9 * 0 + 0.1 * 1 = 0.1
    shadow = ema.shadow["weight"]
    assert torch.allclose(shadow, torch.full_like(shadow, 0.1))


def test_ema_apply_and_restore_round_trip() -> None:
    m = _tiny_model()
    ema = ExponentialMovingAverage(m, decay=0.5)
    # Diverge: live = 1, shadow stays at 0
    with torch.no_grad():
        m.weight.copy_(torch.ones_like(m.weight))
    # apply_to should put zeros (the shadow) into the live model.
    ema.apply_to(m)
    assert torch.allclose(m.weight, torch.zeros_like(m.weight))
    # restore should put the (saved) ones back.
    ema.restore(m)
    assert torch.allclose(m.weight, torch.ones_like(m.weight))


def test_ema_state_dict_round_trip() -> None:
    m = _tiny_model()
    ema = ExponentialMovingAverage(m, decay=0.99)
    with torch.no_grad():
        m.weight.copy_(torch.full_like(m.weight, 5.0))
    ema.update(m)
    state = ema.state_dict()

    # Build fresh EMA and load.
    m2 = _tiny_model()
    ema2 = ExponentialMovingAverage(m2, decay=0.99)
    ema2.load_state_dict(state)
    assert torch.allclose(ema2.shadow["weight"], ema.shadow["weight"])


def test_ema_invalid_decay_raises() -> None:
    m = _tiny_model()
    with pytest.raises(ValueError):
        ExponentialMovingAverage(m, decay=-0.1)
    with pytest.raises(ValueError):
        ExponentialMovingAverage(m, decay=1.5)
