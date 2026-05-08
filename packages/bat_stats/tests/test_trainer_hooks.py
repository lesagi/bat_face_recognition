"""Tests for the trainer-hooks Protocol and Phase-2 stub."""

from __future__ import annotations

import pytest
from bat_stats.trainer_hooks import TrainerProtocol, create_permutation_trainer


def test_create_permutation_trainer_raises_until_phase_2() -> None:
    """The default factory is a stub that raises NotImplementedError."""

    with pytest.raises(NotImplementedError, match="phase-2|bat_training|Phase-2"):
        create_permutation_trainer()


def test_concrete_trainer_satisfies_protocol() -> None:
    class _T:
        def train_and_evaluate(self) -> dict[str, float]:
            return {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

        def reset_for_new_permutation(self) -> None:
            pass

    assert isinstance(_T(), TrainerProtocol)


def test_partial_implementation_does_not_satisfy_protocol() -> None:
    class _T:
        # Missing reset_for_new_permutation
        def train_and_evaluate(self) -> dict[str, float]:
            return {}

    assert not isinstance(_T(), TrainerProtocol)
