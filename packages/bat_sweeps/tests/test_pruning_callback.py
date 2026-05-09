"""OptunaPruningCallback + OptunaPruningTracker behaviour."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("optuna")

import optuna  # noqa: E402

from bat_sweeps.pruning_callback import OptunaPruningCallback, OptunaPruningTracker  # noqa: E402


class _CapturingTracker:
    """Captures every (section, metrics, step) the wrapper forwards."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, float], int]] = []
        self.artifacts: list[tuple[Any, str | None]] = []
        self.configs: list[dict[str, Any]] = []
        self.promotions: list[tuple[str, str]] = []

    def log_metrics(self, section: str, metrics: dict[str, float], step: int) -> None:
        self.calls.append((section, dict(metrics), step))

    def log_artifact(self, path: Any, dest_dir: str | None = None) -> None:
        self.artifacts.append((path, dest_dir))

    def log_config(self, cfg: dict[str, Any]) -> None:
        self.configs.append(dict(cfg))

    def promote_to_champion(self, run_id: str, criterion: str) -> bool:
        self.promotions.append((run_id, criterion))
        return True


def test_callback_strips_val_prefix_and_reports() -> None:
    """``target_metric='val/roc_auc'`` -> bare ``roc_auc`` lookup."""

    captured: list[tuple[float, int]] = []

    class _FakeTrial:
        number = 0

        def report(self, value: float, step: int) -> None:
            captured.append((float(value), int(step)))

        def should_prune(self) -> bool:
            return False

    cb = OptunaPruningCallback(trial=_FakeTrial(), target_metric="val/roc_auc")  # type: ignore[arg-type]
    cb.on_validation_end(epoch=1, metrics={"roc_auc": 0.9, "loss": 0.1})
    assert captured == [(0.9, 1)]
    assert cb.last_value == 0.9


def test_callback_rejects_test_target() -> None:
    class _FakeTrial:
        number = 0

    with pytest.raises(ValueError, match="sacred"):
        OptunaPruningCallback(trial=_FakeTrial(), target_metric="test/roc_auc")  # type: ignore[arg-type]


def test_callback_skips_when_metric_absent() -> None:
    class _FakeTrial:
        number = 0
        reported: list[tuple[float, int]] = []

        def report(self, value: float, step: int) -> None:  # pragma: no cover
            type(self).reported.append((value, step))

        def should_prune(self) -> bool:  # pragma: no cover
            return False

    cb = OptunaPruningCallback(trial=_FakeTrial(), target_metric="val/roc_auc")  # type: ignore[arg-type]
    cb.on_validation_end(epoch=0, metrics={"loss": 1.0})  # no roc_auc -> no-op
    assert cb.last_value is None


def test_callback_raises_pruned_when_should_prune() -> None:
    class _FakeTrial:
        number = 7

        def report(self, value: float, step: int) -> None:
            pass

        def should_prune(self) -> bool:
            return True

    cb = OptunaPruningCallback(trial=_FakeTrial(), target_metric="val/roc_auc")  # type: ignore[arg-type]
    with pytest.raises(optuna.TrialPruned):
        cb.on_validation_end(epoch=2, metrics={"roc_auc": 0.5})


def test_tracker_wrapper_forwards_all_sections_and_triggers_only_on_val() -> None:
    """The tracker wrapper passes every section through but only val triggers report."""
    inner = _CapturingTracker()
    reports: list[tuple[float, int]] = []

    class _FakeTrial:
        number = 0

        def report(self, value: float, step: int) -> None:
            reports.append((float(value), int(step)))

        def should_prune(self) -> bool:
            return False

    cb = OptunaPruningCallback(trial=_FakeTrial(), target_metric="val/roc_auc")  # type: ignore[arg-type]
    wrapped = OptunaPruningTracker(inner=inner, callback=cb)

    wrapped.log_metrics(section="train", metrics={"loss": 0.5}, step=1)
    wrapped.log_metrics(section="val", metrics={"roc_auc": 0.8, "loss": 0.4}, step=1)

    # Inner saw both sections.
    assert {c[0] for c in inner.calls} == {"train", "val"}
    # Pruner only got the val payload.
    assert reports == [(0.8, 1)]


def test_tracker_wrapper_handles_none_inner() -> None:
    class _FakeTrial:
        number = 0

        def report(self, value: float, step: int) -> None:
            pass

        def should_prune(self) -> bool:
            return False

    cb = OptunaPruningCallback(trial=_FakeTrial(), target_metric="val/roc_auc")  # type: ignore[arg-type]
    wrapped = OptunaPruningTracker(inner=None, callback=cb)
    # Must not raise.
    wrapped.log_metrics(section="val", metrics={"roc_auc": 0.5}, step=1)
    wrapped.log_artifact("x", "y")
    wrapped.log_config({"a": 1})
    assert wrapped.promote_to_champion("r", "c") is False
