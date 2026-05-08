"""Tests for `make_trainer` family dispatch."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from bat_core.exceptions import InterfaceViolationError
from bat_training import EmbeddingTrainer, PairTrainer, make_trainer
from bat_training._common import TrainerConfig


class _PairLossStub:
    family = "pair"

    def __call__(self, model_output: Any, labels: Any, sample_weights: Any = None) -> Any:
        return torch.tensor(0.0)


class _EmbedLossStub:
    family = "embedding"

    def __call__(self, model_output: Any, labels: Any, sample_weights: Any = None) -> Any:
        return torch.tensor(0.0)


class _PairModel(torch.nn.Module):
    family = "pair"

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)


class _EmbedModel(torch.nn.Module):
    family = "embedding"

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)


def test_make_trainer_dispatches_pair() -> None:
    t = make_trainer(TrainerConfig(), _PairModel(), _PairLossStub())
    assert isinstance(t, PairTrainer)


def test_make_trainer_dispatches_embedding() -> None:
    t = make_trainer(TrainerConfig(), _EmbedModel(), _EmbedLossStub())
    assert isinstance(t, EmbeddingTrainer)


def test_make_trainer_family_mismatch_raises() -> None:
    with pytest.raises(InterfaceViolationError):
        make_trainer(TrainerConfig(), _PairModel(), _EmbedLossStub())
    with pytest.raises(InterfaceViolationError):
        make_trainer(TrainerConfig(), _EmbedModel(), _PairLossStub())


def test_make_trainer_unknown_family_raises() -> None:
    class _Loss:
        family = "weird"

        def __call__(self, *_: Any, **__: Any) -> Any:  # pragma: no cover
            return torch.tensor(0.0)

    with pytest.raises(InterfaceViolationError):
        make_trainer(TrainerConfig(), _PairModel(), _Loss())
