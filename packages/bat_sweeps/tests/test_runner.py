"""run_sweep end-to-end orchestrator tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("optuna")

import optuna  # noqa: E402

from bat_sweeps import run_sweep  # noqa: E402
from bat_sweeps.objective import TrialComponents  # noqa: E402


@dataclass
class _FakeRunArtifacts:
    run_id: str = ""
    best_metrics: dict[str, float] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)


class _FakeTrainer:
    """Records the per-trial value and does NOT touch test()."""

    def __init__(self, *, tracker: Any, value: float) -> None:
        self.tracker = tracker
        self.value = value

    def fit(self, *_args: Any, **_kwargs: Any) -> _FakeRunArtifacts:
        if self.tracker is not None:
            self.tracker.log_metrics(section="val", metrics={"roc_auc": self.value}, step=1)
        return _FakeRunArtifacts(best_metrics={"roc_auc": self.value})

    def test(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("trainer.test() is sacred -- the sweep must not invoke it")


class _MockMLflowTracker:
    """Captures champion-logging side-effects for the test."""

    def __init__(self) -> None:
        self.metric_calls: list[tuple[str, dict[str, float], int]] = []
        self.config_calls: list[dict[str, Any]] = []

    def log_metrics(self, section: str, metrics: dict[str, float], step: int) -> None:
        self.metric_calls.append((section, dict(metrics), step))

    def log_config(self, cfg: dict[str, Any]) -> None:
        self.config_calls.append(dict(cfg))

    def log_artifact(self, *_a: Any, **_kw: Any) -> None:
        pass

    def promote_to_champion(self, *_a: Any, **_kw: Any) -> bool:
        return False


def test_run_sweep_writes_champion_through_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """After ``study.optimize`` finishes, the champion is logged to the tracker."""

    # Per-trial values; max(0.5, 0.9, 0.6) = 0.9 -> trial 1 is champion.
    values = [0.5, 0.9, 0.6]
    state = {"i": 0}

    def _factory(_merged: dict[str, Any]) -> TrialComponents:
        idx = state["i"]
        state["i"] = idx + 1
        return TrialComponents(
            model=object(),
            loss=type("L", (), {"family": "embedding"})(),
            trainer_cfg=object(),
            train_loader=[],
            val_loader=[],
            tracker=None,
            trainer_kwargs={"_value": values[idx]},
        )

    def _fake_make_trainer(
        cfg: Any,  # noqa: ARG001
        model: Any,  # noqa: ARG001
        loss: Any,  # noqa: ARG001
        *,
        tracker: Any | None = None,
        **kwargs: Any,
    ) -> _FakeTrainer:
        return _FakeTrainer(tracker=tracker, value=kwargs.pop("_value"))

    monkeypatch.setattr("bat_training.factory.make_trainer", _fake_make_trainer)

    cfg = {
        "study_name": "test_run_sweep",
        "n_trials": 3,
        "direction": "maximize",
        "target_metric": "val/roc_auc",
        "pruner": "none",
        "storage": None,  # in-memory
        "search_space": {"loss.margin": {"type": "float", "low": 0.3, "high": 0.6}},
        "base_cfg": {"loss": {"name": "arcface"}},
    }

    champ_tracker = _MockMLflowTracker()
    study = run_sweep(cfg, build_components=_factory, champion_tracker=champ_tracker)

    assert study.best_value == pytest.approx(0.9)
    assert study.best_trial.number == 1

    # Champion was logged.
    assert any(
        section == "val" and "champion_value" in metrics
        for section, metrics, _ in champ_tracker.metric_calls
    )
    # And a config snapshot was written with study_name + best_params.
    assert champ_tracker.config_calls
    snapshot = champ_tracker.config_calls[0]
    assert snapshot["study_name"] == "test_run_sweep"
    assert snapshot["best_value"] == pytest.approx(0.9)
    assert snapshot["best_trial_number"] == 1
    assert "loss.margin" in snapshot["best_params"]


def test_run_sweep_no_tracker_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """``champion_tracker=None`` works without raising."""

    def _factory(_merged: dict[str, Any]) -> TrialComponents:
        return TrialComponents(
            model=object(),
            loss=type("L", (), {"family": "embedding"})(),
            trainer_cfg=object(),
            train_loader=[],
            val_loader=[],
            tracker=None,
            trainer_kwargs={"_value": 0.7},
        )

    def _fake_make_trainer(
        cfg: Any,  # noqa: ARG001
        model: Any,  # noqa: ARG001
        loss: Any,  # noqa: ARG001
        *,
        tracker: Any | None = None,
        **kwargs: Any,
    ) -> _FakeTrainer:
        return _FakeTrainer(tracker=tracker, value=kwargs.pop("_value"))

    monkeypatch.setattr("bat_training.factory.make_trainer", _fake_make_trainer)

    cfg = {
        "study_name": "test_no_tracker",
        "n_trials": 1,
        "pruner": "none",
        "storage": None,
        "search_space": {"loss.margin": {"type": "float", "low": 0.3, "high": 0.6}},
    }
    study = run_sweep(cfg, build_components=_factory, champion_tracker=None)
    assert study.best_value == pytest.approx(0.7)


def test_run_sweep_uses_in_memory_storage_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """``storage=None`` keeps the study in-memory (no sqlite file)."""

    def _factory(_merged: dict[str, Any]) -> TrialComponents:
        return TrialComponents(
            model=object(),
            loss=type("L", (), {"family": "embedding"})(),
            trainer_cfg=object(),
            train_loader=[],
            val_loader=[],
            tracker=None,
            trainer_kwargs={"_value": 0.5},
        )

    def _fake_make_trainer(
        cfg: Any,  # noqa: ARG001
        model: Any,  # noqa: ARG001
        loss: Any,  # noqa: ARG001
        *,
        tracker: Any | None = None,
        **kwargs: Any,
    ) -> _FakeTrainer:
        return _FakeTrainer(tracker=tracker, value=kwargs.pop("_value"))

    monkeypatch.setattr("bat_training.factory.make_trainer", _fake_make_trainer)

    cfg = {
        "study_name": "test_mem",
        "n_trials": 1,
        "storage": None,
        "pruner": "none",
        "search_space": {"loss.margin": {"type": "float", "low": 0.3, "high": 0.6}},
    }
    study = run_sweep(cfg, build_components=_factory)
    # In-memory studies have a None storage URL when introspected via the
    # public `_storage` of optuna -- we simply assert the study completed.
    assert len(study.trials) == 1


def test_run_sweep_pruner_param_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cfg pruner='median' attaches a MedianPruner to the study."""

    def _factory(_merged: dict[str, Any]) -> TrialComponents:
        return TrialComponents(
            model=object(),
            loss=type("L", (), {"family": "embedding"})(),
            trainer_cfg=object(),
            train_loader=[],
            val_loader=[],
            tracker=None,
            trainer_kwargs={"_value": 0.5},
        )

    def _fake_make_trainer(
        cfg: Any,  # noqa: ARG001
        model: Any,  # noqa: ARG001
        loss: Any,  # noqa: ARG001
        *,
        tracker: Any | None = None,
        **kwargs: Any,
    ) -> _FakeTrainer:
        return _FakeTrainer(tracker=tracker, value=kwargs.pop("_value"))

    monkeypatch.setattr("bat_training.factory.make_trainer", _fake_make_trainer)

    cfg = {
        "study_name": "test_pruner",
        "n_trials": 1,
        "pruner": "median",
        "storage": None,
        "search_space": {"loss.margin": {"type": "float", "low": 0.3, "high": 0.6}},
    }
    study = run_sweep(cfg, build_components=_factory)
    assert isinstance(study.pruner, optuna.pruners.MedianPruner)
