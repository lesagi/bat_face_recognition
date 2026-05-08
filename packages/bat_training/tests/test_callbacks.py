"""Tests for EarlyStopping and deterministic-mode helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bat_training.callbacks import EarlyStopping, set_deterministic_mode, worker_init_fn


def test_early_stopping_max_mode_triggers_after_patience() -> None:
    es = EarlyStopping(monitor="f1", mode="max", patience=2, min_delta=1e-3)
    # epoch 1: improvement -- counter=0
    assert es.step(0.5, epoch=1) is False
    # epoch 2: no improvement -- counter=1
    assert es.step(0.4, epoch=2) is False
    # epoch 3: still no improvement -- counter=2 -> trigger
    assert es.step(0.4, epoch=3) is True
    assert es.state.triggered is True


def test_early_stopping_min_mode() -> None:
    es = EarlyStopping(monitor="loss", mode="min", patience=1, min_delta=1e-3)
    assert es.step(1.0, epoch=1) is False
    # No improvement -> trigger after patience=1.
    assert es.step(1.5, epoch=2) is True


def test_early_stopping_min_delta_resets_only_for_real_improvements() -> None:
    es = EarlyStopping(monitor="f1", mode="max", patience=1, min_delta=0.05)
    es.step(0.5, epoch=1)
    # 0.51 is < 0.5 + 0.05; counts as no improvement -> patience tick = 1
    assert es.step(0.51, epoch=2) is True


def test_set_deterministic_mode_idempotent() -> None:
    # warn_only=True so even ops without deterministic kernels don't crash.
    set_deterministic_mode(seed=123, warn_only=True)
    set_deterministic_mode(seed=123, warn_only=True)


def test_worker_init_fn_runs() -> None:
    worker_init_fn(0)
    worker_init_fn(7)
