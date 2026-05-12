"""End-to-end objective tests with a stubbed make_trainer.

These tests deliberately replace :func:`bat_training.factory.make_trainer`
with a tiny in-process trainer so they exercise the objective wiring
without dragging :mod:`accelerate`, :mod:`torch.optim`, etc. The real
trainer integration is covered by Worker H's tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("optuna")

import optuna  # noqa: E402
from bat_sweeps.objective import TrialComponents, build_objective  # noqa: E402

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeRunArtifacts:
    run_id: str = ""
    best_metrics: dict[str, float] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)


class _FakeTrainer:
    """Tiny stand-in for PairTrainer/EmbeddingTrainer.

    On ``fit`` it walks the per-epoch metric schedule the test passed in,
    forwards each epoch's metrics through the wrapped tracker (so the
    pruning callback fires), and returns a :class:`_FakeRunArtifacts`
    carrying the last value as ``best_metrics``.

    Records whether ``test()`` was ever called -- the sacred-test check
    asserts on this.
    """

    test_called: bool = False  # class-level flag; reset per-test

    def __init__(
        self,
        *,
        tracker: Any,
        epoch_metrics: list[dict[str, float]],
        target_metric: str,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.tracker = tracker
        self.epoch_metrics = epoch_metrics
        self.target_metric = target_metric
        self.callbacks = list(callbacks or [])

    def fit(self, train_loader: Any, val_loader: Any) -> _FakeRunArtifacts:  # noqa: ARG002
        last = 0.0
        bare = self.target_metric.removeprefix("val/")
        for epoch, m in enumerate(self.epoch_metrics, start=1):
            if self.tracker is not None:
                # Trainers always log train then val per epoch; mimic that.
                self.tracker.log_metrics(
                    section="train", metrics={"loss": m.get("loss", 0.5)}, step=epoch
                )
                self.tracker.log_metrics(section="val", metrics=m, step=epoch)
            # The real trainers fire callbacks AFTER the val log call. Mirror.
            for cb in self.callbacks:
                cb.on_validation_end(epoch=epoch, metrics=m)
            if bare in m:
                last = float(m[bare])
        return _FakeRunArtifacts(best_metrics={bare: last})

    def test(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover
        type(self).test_called = True
        raise AssertionError("trainer.test() must NEVER be called by the sweep")


@dataclass
class _FakeLoss:
    family: str = "embedding"

    def __call__(self, *_a: Any, **_kw: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def _build_components_factory(
    schedules: dict[int, list[dict[str, float]]],
    target_metric: str,
):
    """Return a build_components closure that picks a schedule per trial.

    ``schedules`` maps trial ``number`` -> list-of-metric-dicts (one per
    epoch). The closure also stores the latest fake trainer so the test
    can assert on it.
    """

    state = {"trainer": None, "trials_seen": 0}

    def _factory(_merged_cfg: dict[str, Any]) -> TrialComponents:
        # Trial number is implicit; we use a counter.
        idx = state["trials_seen"]
        state["trials_seen"] = idx + 1
        schedule = schedules.get(idx, [{}])

        def _make_trainer_factory(*_args: Any, **kwargs: Any) -> _FakeTrainer:
            ft = _FakeTrainer(
                tracker=kwargs.get("tracker"),
                epoch_metrics=schedule,
                target_metric=target_metric,
            )
            state["trainer"] = ft
            return ft

        # Smuggle the per-trial factory through to objective._objective via
        # a sentinel attribute on the loss; the test patches make_trainer
        # at module level instead, so this is unused. Returning components.
        return TrialComponents(
            model=object(),
            loss=_FakeLoss(family="embedding"),
            trainer_cfg=object(),
            train_loader=[],
            val_loader=[],
            tracker=None,
            trainer_kwargs={"_schedule": schedule, "_target_metric": target_metric},
        )

    return _factory, state


def _patch_make_trainer(monkeypatch: pytest.MonkeyPatch, state: dict) -> None:
    """Replace bat_training.factory.make_trainer with a fake-aware version."""

    def _fake_make_trainer(
        cfg: Any,  # noqa: ARG001
        model: Any,  # noqa: ARG001
        loss: Any,  # noqa: ARG001
        *,
        tracker: Any | None = None,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> _FakeTrainer:
        schedule = kwargs.pop("_schedule", [{}])
        target = kwargs.pop("_target_metric", "val/roc_auc")
        ft = _FakeTrainer(
            tracker=tracker,
            epoch_metrics=schedule,
            target_metric=target,
            callbacks=callbacks,
        )
        state["trainer"] = ft
        return ft

    monkeypatch.setattr("bat_training.factory.make_trainer", _fake_make_trainer)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_objective_runs_two_trials_two_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both trials complete and report at least one value."""

    schedules = {
        0: [{"roc_auc": 0.6, "loss": 0.5}, {"roc_auc": 0.7, "loss": 0.4}],
        1: [{"roc_auc": 0.5, "loss": 0.6}, {"roc_auc": 0.8, "loss": 0.3}],
    }
    factory, state = _build_components_factory(schedules, target_metric="val/roc_auc")
    _patch_make_trainer(monkeypatch, state)

    objective = build_objective(
        base_cfg={"loss": {"name": "arcface"}},
        search_space={"loss.margin": {"type": "float", "low": 0.3, "high": 0.6}},
        build_components=factory,
        target_metric="val/roc_auc",
    )

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
    study.optimize(objective, n_trials=2)

    assert len(study.trials) == 2
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) == 2
    # Best value is the higher final roc_auc.
    assert study.best_value == pytest.approx(0.8)


def test_objective_never_calls_trainer_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sacred-test invariant: trainer.test() is never invoked from the sweep."""
    schedules = {0: [{"roc_auc": 0.7, "loss": 0.4}]}
    factory, state = _build_components_factory(schedules, target_metric="val/roc_auc")
    _patch_make_trainer(monkeypatch, state)

    _FakeTrainer.test_called = False
    objective = build_objective(
        base_cfg={},
        search_space={"loss.margin": {"type": "float", "low": 0.3, "high": 0.6}},
        build_components=factory,
        target_metric="val/roc_auc",
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    assert _FakeTrainer.test_called is False


def test_objective_pruning_works_with_median_pruner(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trial reporting much-worse intermediate values gets pruned."""

    # Trial 0: high values seed the pruner.
    # Trial 1: high values reinforce.
    # Trial 2: very low first-epoch value -> should be pruned.
    schedules = {
        0: [{"roc_auc": 0.95}, {"roc_auc": 0.96}, {"roc_auc": 0.97}, {"roc_auc": 0.98}],
        1: [{"roc_auc": 0.94}, {"roc_auc": 0.95}, {"roc_auc": 0.96}, {"roc_auc": 0.97}],
        2: [{"roc_auc": 0.10}, {"roc_auc": 0.10}, {"roc_auc": 0.10}, {"roc_auc": 0.10}],
    }
    factory, state = _build_components_factory(schedules, target_metric="val/roc_auc")
    _patch_make_trainer(monkeypatch, state)

    objective = build_objective(
        base_cfg={},
        search_space={"loss.margin": {"type": "float", "low": 0.3, "high": 0.6}},
        build_components=factory,
        target_metric="val/roc_auc",
    )
    # n_startup_trials=0 so the third trial is pruner-eligible immediately.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=3)

    states = [t.state for t in study.trials]
    assert optuna.trial.TrialState.PRUNED in states


def test_objective_rejects_test_target_metric() -> None:
    """``target_metric='test/...'`` is forbidden -- test split is sacred."""
    factory, _ = _build_components_factory({}, target_metric="val/roc_auc")
    with pytest.raises(ValueError, match="sacred"):
        build_objective(
            base_cfg={},
            search_space={"loss.margin": {"type": "float", "low": 0.3, "high": 0.6}},
            build_components=factory,
            target_metric="test/roc_auc",
        )


def test_objective_accepts_bare_metric_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing ``'roc_auc'`` (no ``val/`` prefix) is normalised to val/."""
    schedules = {0: [{"roc_auc": 0.5}, {"roc_auc": 0.7}]}
    factory, state = _build_components_factory(schedules, target_metric="val/roc_auc")
    _patch_make_trainer(monkeypatch, state)

    objective = build_objective(
        base_cfg={},
        search_space={"loss.margin": {"type": "float", "low": 0.3, "high": 0.6}},
        build_components=factory,
        target_metric="roc_auc",  # bare
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    assert study.best_value == pytest.approx(0.7)


def test_objective_overrides_reach_build_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """The merged cfg handed to build_components contains sampled overrides."""
    schedules = {0: [{"roc_auc": 0.5}]}
    state = {"trainer": None, "trials_seen": 0}
    seen_cfgs: list[dict[str, Any]] = []

    def _factory(merged: dict[str, Any]) -> TrialComponents:
        seen_cfgs.append(merged)
        idx = state["trials_seen"]
        state["trials_seen"] = idx + 1
        return TrialComponents(
            model=object(),
            loss=_FakeLoss("embedding"),
            trainer_cfg=object(),
            train_loader=[],
            val_loader=[],
            tracker=None,
            trainer_kwargs={
                "_schedule": schedules[idx],
                "_target_metric": "val/roc_auc",
            },
        )

    _patch_make_trainer(monkeypatch, state)

    objective = build_objective(
        base_cfg={"loss": {"name": "arcface", "scale": 64.0}},
        search_space={"loss.margin": {"type": "float", "low": 0.3, "high": 0.6}},
        build_components=_factory,
        target_metric="val/roc_auc",
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    assert len(seen_cfgs) == 1
    merged = seen_cfgs[0]
    # The base value survived.
    assert merged["loss"]["scale"] == 64.0
    assert merged["loss"]["name"] == "arcface"
    # The override landed.
    assert 0.3 <= merged["loss"]["margin"] <= 0.6


def test_objective_wires_pruning_through_callback_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The objective plugs OptunaPruningCallback into trainer.callbacks=[...]
    (not into a wrapper around the tracker)."""
    from bat_sweeps.pruning_callback import OptunaPruningCallback

    schedules = {0: [{"roc_auc": 0.5}, {"roc_auc": 0.7}]}
    factory, state = _build_components_factory(schedules, target_metric="val/roc_auc")
    _patch_make_trainer(monkeypatch, state)

    objective = build_objective(
        base_cfg={},
        search_space={"loss.margin": {"type": "float", "low": 0.3, "high": 0.6}},
        build_components=factory,
        target_metric="val/roc_auc",
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    ft = state["trainer"]
    assert ft is not None
    # Exactly one OptunaPruningCallback should be in the trainer's callback
    # list; the legacy tracker-wrap path is no longer used.
    pruning_cbs = [cb for cb in ft.callbacks if isinstance(cb, OptunaPruningCallback)]
    assert len(pruning_cbs) == 1
    # Pruning callback must have observed the last reported value.
    assert pruning_cbs[0].last_value == pytest.approx(0.7)


def test_objective_preserves_user_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """User-supplied callbacks from build_components survive the objective."""
    from bat_sweeps.pruning_callback import OptunaPruningCallback

    class _UserCb:
        def __init__(self) -> None:
            self.seen: list[int] = []

        def on_validation_end(self, epoch: int, metrics: dict[str, float]) -> None:
            self.seen.append(epoch)

    user_cb = _UserCb()
    schedules = {0: [{"roc_auc": 0.5}, {"roc_auc": 0.7}]}
    target_metric = "val/roc_auc"
    state = {"trainer": None, "trials_seen": 0}

    def _factory(_merged: dict[str, Any]) -> TrialComponents:
        idx = state["trials_seen"]
        state["trials_seen"] = idx + 1
        return TrialComponents(
            model=object(),
            loss=_FakeLoss("embedding"),
            trainer_cfg=object(),
            train_loader=[],
            val_loader=[],
            tracker=None,
            trainer_kwargs={
                "callbacks": [user_cb],
                "_schedule": schedules[idx],
                "_target_metric": target_metric,
            },
        )

    _patch_make_trainer(monkeypatch, state)
    objective = build_objective(
        base_cfg={},
        search_space={"loss.margin": {"type": "float", "low": 0.3, "high": 0.6}},
        build_components=_factory,
        target_metric=target_metric,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    ft = state["trainer"]
    assert ft is not None
    # Both callbacks present: the user's plus the pruning hook (appended).
    assert user_cb in ft.callbacks
    assert any(isinstance(cb, OptunaPruningCallback) for cb in ft.callbacks)
    # The user callback observed every val epoch.
    assert user_cb.seen == [1, 2]
