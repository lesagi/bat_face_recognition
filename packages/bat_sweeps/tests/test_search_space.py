"""parse_search_space + apply_overrides tests."""

from __future__ import annotations

import pytest

pytest.importorskip("optuna")

import optuna  # noqa: E402
from bat_sweeps.search_space import apply_overrides, parse_search_space  # noqa: E402


def _run_with_sampler(spec: dict) -> dict:
    """Drive the sampler through one Optuna trial and return overrides."""
    sampler = parse_search_space(spec)
    captured: dict = {}

    def _objective(trial: optuna.trial.Trial) -> float:
        captured.update(sampler(trial))
        return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=1)
    return captured


def test_parse_float_param_in_range() -> None:
    overrides = _run_with_sampler({"loss.margin": {"type": "float", "low": 0.3, "high": 0.6}})
    assert "loss.margin" in overrides
    assert 0.3 <= overrides["loss.margin"] <= 0.6


def test_parse_int_param_in_range() -> None:
    overrides = _run_with_sampler({"trainer.batch_size": {"type": "int", "low": 8, "high": 64}})
    val = overrides["trainer.batch_size"]
    assert isinstance(val, int)
    assert 8 <= val <= 64


def test_parse_categorical_param_in_choices() -> None:
    overrides = _run_with_sampler(
        {"trainer.optimizer": {"type": "categorical", "choices": ["adam", "sgd"]}}
    )
    assert overrides["trainer.optimizer"] in {"adam", "sgd"}


def test_parse_loguniform_param_in_range() -> None:
    overrides = _run_with_sampler({"trainer.lr": {"type": "loguniform", "low": 1e-4, "high": 1e-1}})
    val = overrides["trainer.lr"]
    assert 1e-4 <= val <= 1e-1


def test_parse_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        parse_search_space({"x": {"type": "exponential", "low": 0, "high": 1}})


def test_parse_missing_low_raises() -> None:
    with pytest.raises(ValueError, match="missing required 'low'"):
        parse_search_space({"x": {"type": "float", "high": 1.0}})


def test_parse_categorical_requires_choices() -> None:
    with pytest.raises(ValueError, match="missing required 'choices'"):
        parse_search_space({"x": {"type": "categorical"}})


def test_parse_categorical_rejects_empty_choices() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_search_space({"x": {"type": "categorical", "choices": []}})


def test_parse_top_level_must_be_dict() -> None:
    with pytest.raises(TypeError):
        parse_search_space("loss.margin")  # type: ignore[arg-type]


def test_apply_overrides_creates_intermediate_dicts() -> None:
    base: dict = {"loss": {"name": "arcface"}}
    merged = apply_overrides(base, {"loss.margin": 0.4, "trainer.lr": 1e-3})
    assert merged["loss"]["margin"] == 0.4
    assert merged["loss"]["name"] == "arcface"  # untouched
    assert merged["trainer"]["lr"] == 1e-3
    # base must not be mutated
    assert "margin" not in base["loss"]
    assert "trainer" not in base


def test_apply_overrides_rejects_dotted_into_non_dict() -> None:
    base: dict = {"loss": "arcface"}  # ancestor is a string, not a dict
    with pytest.raises(TypeError):
        apply_overrides(base, {"loss.margin": 0.4})
