"""Tests for permutation_adapter.create_pair_trainer_factory.

The factory must return a callable that produces a TrainerProtocol-shaped
object consumable by ``bat_stats.runner.run_retrain_test``.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

from bat_losses import BCELoss
from bat_training import create_pair_trainer_factory
from bat_training._common import TrainerConfig


def test_factory_produces_trainer_protocol_shape(
    tmp_path: Any, tiny_pair_model_cls: Any, make_pair_loader: Any
) -> None:
    cfg = TrainerConfig(epochs=1, lr=1e-2, output_dir=tmp_path / "run")
    factory = create_pair_trainer_factory(
        model_factory=tiny_pair_model_cls,
        loss_factory=BCELoss,
        train_loader_factory=lambda permute: make_pair_loader(),
        val_loader_factory=None,
        cfg=cfg,
    )
    adapter = factory(permute_labels=True, num_epochs=1, verbose=False)

    assert hasattr(adapter, "train_and_evaluate")
    assert callable(adapter.train_and_evaluate)
    assert hasattr(adapter, "reset_for_new_permutation")
    assert callable(adapter.reset_for_new_permutation)


def test_factory_runs_one_permutation(
    tmp_path: Any, tiny_pair_model_cls: Any, make_pair_loader: Any
) -> None:
    cfg = TrainerConfig(epochs=1, lr=1e-2, output_dir=tmp_path / "run")
    factory = create_pair_trainer_factory(
        model_factory=tiny_pair_model_cls,
        loss_factory=BCELoss,
        train_loader_factory=lambda permute: make_pair_loader(),
        val_loader_factory=None,
        cfg=cfg,
    )
    adapter = factory(permute_labels=True, num_epochs=1, verbose=False)
    metrics = adapter.train_and_evaluate()
    for key in ("f1", "accuracy", "precision", "recall"):
        assert key in metrics, f"missing required metric {key!r}"


def test_factory_consumed_by_bat_stats_runner_signature(
    tmp_path: Any, tiny_pair_model_cls: Any, make_pair_loader: Any
) -> None:
    """The factory must accept the kwargs ``bat_stats.PermutationTest`` passes."""
    cfg = TrainerConfig(epochs=1, lr=1e-2, output_dir=tmp_path / "run")
    factory = create_pair_trainer_factory(
        model_factory=tiny_pair_model_cls,
        loss_factory=BCELoss,
        train_loader_factory=lambda permute: make_pair_loader(),
        val_loader_factory=None,
        cfg=cfg,
    )
    adapter = factory(
        permute_labels=True, num_epochs=1, verbose=False, ignored_extra="ok"
    )
    assert hasattr(adapter, "train_and_evaluate")

    adapter.reset_for_new_permutation()
