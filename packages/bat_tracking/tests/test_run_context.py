"""Tests for :func:`bat_tracking.start_run` context manager."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bat_tracking import MLflowTracker, start_run


def _fake_experiment(experiment_id: str = "1") -> SimpleNamespace:
    return SimpleNamespace(experiment_id=experiment_id, name="exp")


def test_start_run_yields_tracker_and_binds_run_id() -> None:
    fake_client = MagicMock()
    fake_client.get_experiment_by_name.return_value = _fake_experiment("1")

    fake_run = SimpleNamespace(info=SimpleNamespace(run_id="ctx-run-77"))
    cm_enter = MagicMock(return_value=fake_run)
    cm_exit = MagicMock(return_value=False)
    fake_run_cm = MagicMock()
    fake_run_cm.__enter__ = cm_enter
    fake_run_cm.__exit__ = cm_exit

    with (
        patch(
            "bat_tracking.mlflow_tracker.MlflowClient",
            return_value=fake_client,
        ),
        patch(
            "bat_tracking.run_context.mlflow.start_run",
            return_value=fake_run_cm,
        ) as m_start,
        start_run(experiment_name="exp", run_name="r1") as tracker,
    ):
        assert isinstance(tracker, MLflowTracker)
        # The tracker is bound to the active run id.
        assert tracker._run_id == "ctx-run-77"

    # mlflow.start_run was invoked once with our experiment id + run name.
    m_start.assert_called_once()
    kw = m_start.call_args.kwargs
    assert kw["experiment_id"] == "1"
    assert kw["run_name"] == "r1"
    assert kw["tags"] is None
    # The context manager exited cleanly.
    assert cm_enter.called
    assert cm_exit.called


def test_start_run_forwards_tags() -> None:
    fake_client = MagicMock()
    fake_client.get_experiment_by_name.return_value = _fake_experiment()
    fake_run = SimpleNamespace(info=SimpleNamespace(run_id="r"))
    fake_run_cm = MagicMock()
    fake_run_cm.__enter__ = MagicMock(return_value=fake_run)
    fake_run_cm.__exit__ = MagicMock(return_value=False)

    tags = {"phase": "phase-1", "worker": "F"}

    with (
        patch(
            "bat_tracking.mlflow_tracker.MlflowClient",
            return_value=fake_client,
        ),
        patch(
            "bat_tracking.run_context.mlflow.start_run",
            return_value=fake_run_cm,
        ) as m_start,
        start_run("exp", run_name="r1", tags=tags),
    ):
        pass

    assert m_start.call_args.kwargs["tags"] == tags


def test_start_run_closes_on_exception() -> None:
    """Run is closed even when the body raises."""
    fake_client = MagicMock()
    fake_client.get_experiment_by_name.return_value = _fake_experiment()
    fake_run = SimpleNamespace(info=SimpleNamespace(run_id="r"))
    cm_exit = MagicMock(return_value=False)
    fake_run_cm = MagicMock()
    fake_run_cm.__enter__ = MagicMock(return_value=fake_run)
    fake_run_cm.__exit__ = cm_exit

    with (
        patch(
            "bat_tracking.mlflow_tracker.MlflowClient",
            return_value=fake_client,
        ),
        patch(
            "bat_tracking.run_context.mlflow.start_run",
            return_value=fake_run_cm,
        ),
    ):
        try:
            with start_run("exp"):
                raise ValueError("boom")
        except ValueError:
            pass

    cm_exit.assert_called_once()
    # First positional/keyword arg to __exit__ is the exception type.
    args = cm_exit.call_args.args
    assert args[0] is ValueError
