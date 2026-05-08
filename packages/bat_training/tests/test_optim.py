"""Tests for optimizer + warmup-cosine LR + gradient accumulation."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from bat_training.optim import (
    GradientAccumulator,
    build_optimizer,
    build_scheduler,
    warmup_cosine_lr,
)


def test_warmup_cosine_lr_at_known_steps() -> None:
    # warmup_steps=4, total_steps=10. At step 0 the first warmup tick
    # gives 1/4; at step 3 we're at 4/4 == 1.0 (end of warmup); at step 9
    # we're at the last cosine point; at step 10 (=total) we're at min.
    assert warmup_cosine_lr(0, warmup_steps=4, total_steps=10) == pytest.approx(0.25)
    assert warmup_cosine_lr(1, warmup_steps=4, total_steps=10) == pytest.approx(0.5)
    assert warmup_cosine_lr(3, warmup_steps=4, total_steps=10) == pytest.approx(1.0)
    # Cosine half: at step warmup_steps + (total - warmup) / 2 = 4 + 3 = 7
    expected_mid = 0.5 * (1.0 + math.cos(math.pi * 0.5))  # ~0.5
    assert warmup_cosine_lr(7, warmup_steps=4, total_steps=10) == pytest.approx(
        expected_mid, abs=1e-6
    )
    # At total_steps: clamp to min_lr_ratio
    assert warmup_cosine_lr(10, warmup_steps=4, total_steps=10, min_lr_ratio=0.1) == pytest.approx(
        0.1
    )
    assert warmup_cosine_lr(99, warmup_steps=4, total_steps=10, min_lr_ratio=0.0) == 0.0


def test_warmup_cosine_lr_zero_warmup() -> None:
    # Without warmup: step 0 -> 1.0, decays cosine over total_steps.
    assert warmup_cosine_lr(0, warmup_steps=0, total_steps=10) == pytest.approx(1.0)
    assert warmup_cosine_lr(5, warmup_steps=0, total_steps=10) == pytest.approx(0.5, abs=1e-6)


def test_build_optimizer_dispatches() -> None:
    p = [torch.nn.Parameter(torch.randn(2, 2))]
    a = build_optimizer(p, name="adam", lr=1e-3, weight_decay=1e-4)
    assert isinstance(a, torch.optim.Adam)
    aw = build_optimizer(p, name="adamw", lr=1e-3)
    assert isinstance(aw, torch.optim.AdamW)
    s = build_optimizer(p, name="sgd", lr=1e-2, momentum=0.9)
    assert isinstance(s, torch.optim.SGD)
    with pytest.raises(ValueError):
        build_optimizer(p, name="foo")  # type: ignore[arg-type]


def test_build_scheduler_lambda_lr_calls_warmup_cosine() -> None:
    import warnings

    model = torch.nn.Linear(4, 4)
    opt = build_optimizer(model.parameters(), name="adam", lr=1.0)
    sched = build_scheduler(opt, warmup_steps=2, total_steps=4)
    # initial scale (before any step) = lr_lambda(0) = 1/2 = 0.5
    initial_lr = opt.param_groups[0]["lr"]
    assert initial_lr == pytest.approx(0.5)
    # Suppress the "scheduler step before optimizer step" UserWarning -- this
    # is the expected behaviour when probing schedule values directly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sched.step()  # advances to step=1: 2/2 = 1.0
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0)


def test_gradient_accumulator_step_cadence() -> None:
    acc = GradientAccumulator(steps=4)
    expected = [False, False, False, True, False, False, False, True]
    for want in expected:
        acc.tick()
        assert acc.should_step is want


def test_gradient_accumulator_invalid_steps_raises() -> None:
    with pytest.raises(ValueError):
        GradientAccumulator(steps=0)
