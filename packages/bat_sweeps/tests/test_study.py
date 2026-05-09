"""make_study tests."""

from __future__ import annotations

import pytest

pytest.importorskip("optuna")

import optuna  # noqa: E402

from bat_sweeps.study import DEFAULT_STORAGE, make_study  # noqa: E402


def test_default_storage_url_lives_next_to_mlruns() -> None:
    """The default storage URL is sqlite alongside mlruns/."""
    assert DEFAULT_STORAGE.startswith("sqlite:///")
    assert DEFAULT_STORAGE.endswith("optuna_studies.db")


def test_make_study_default_pruner_is_median() -> None:
    study = make_study(
        name="t_median",
        storage=None,  # in-memory
    )
    assert isinstance(study.pruner, optuna.pruners.MedianPruner)
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE


def test_make_study_supports_minimize_direction() -> None:
    study = make_study(name="t_min", storage=None, direction="minimize")
    assert study.direction == optuna.study.StudyDirection.MINIMIZE


def test_make_study_pruner_none_is_nop() -> None:
    study = make_study(name="t_nop", storage=None, pruner="none")
    assert isinstance(study.pruner, optuna.pruners.NopPruner)


def test_make_study_unknown_pruner_raises() -> None:
    with pytest.raises(ValueError, match="unknown pruner"):
        make_study(name="t", storage=None, pruner="hyperband")
