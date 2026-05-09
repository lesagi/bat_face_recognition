"""Tests for :class:`bat_tracking.MLflowTracker`.

All MLflow client interactions are mocked --- no real tracking server
required.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from bat_core import Tracker
from bat_tracking import MLflowTracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_experiment(experiment_id: str = "1") -> SimpleNamespace:
    return SimpleNamespace(experiment_id=experiment_id, name="exp")


def _make_tracker(
    client: MagicMock | None = None,
    run_id: str = "run-123",
) -> tuple[MLflowTracker, MagicMock]:
    if client is None:
        client = MagicMock()
        client.get_experiment_by_name.return_value = _fake_experiment()
    tracker = MLflowTracker(
        experiment_name="exp",
        run_id=run_id,
        client=client,
    )
    return tracker, client


# ---------------------------------------------------------------------------
# Construction + experiment resolution
# ---------------------------------------------------------------------------


def test_constructor_uses_existing_experiment() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = _fake_experiment("42")
    tracker = MLflowTracker(experiment_name="exp", client=client)
    assert tracker.experiment_id == "42"
    client.create_experiment.assert_not_called()


def test_constructor_creates_missing_experiment() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = "99"
    tracker = MLflowTracker(experiment_name="new-exp", client=client)
    assert tracker.experiment_id == "99"
    client.create_experiment.assert_called_once_with("new-exp")


# ---------------------------------------------------------------------------
# Tracker protocol satisfaction
# ---------------------------------------------------------------------------


def test_satisfies_tracker_protocol() -> None:
    tracker, _ = _make_tracker()
    assert isinstance(tracker, Tracker)


# ---------------------------------------------------------------------------
# log_metrics: section prefixing
# ---------------------------------------------------------------------------


def test_log_metrics_prefixes_train_keys() -> None:
    tracker, client = _make_tracker()
    tracker.log_metrics("train", {"loss": 0.5, "acc": 0.9}, step=3)
    calls = client.log_metric.call_args_list
    keys_logged = [c.args[1] for c in calls]
    assert "train/loss" in keys_logged
    assert "train/acc" in keys_logged
    # raw keys must not leak through
    assert "loss" not in keys_logged
    assert "acc" not in keys_logged


def test_log_metrics_prefixes_val_keys() -> None:
    tracker, client = _make_tracker()
    tracker.log_metrics("val", {"f1": 0.7}, step=10)
    client.log_metric.assert_called_once_with("run-123", "val/f1", 0.7, step=10)


def test_log_metrics_prefixes_test_keys() -> None:
    tracker, client = _make_tracker()
    tracker.log_metrics("test", {"roc_auc": 0.95}, step=0)
    client.log_metric.assert_called_once_with("run-123", "test/roc_auc", 0.95, step=0)


def test_log_metrics_coerces_values_to_float() -> None:
    tracker, client = _make_tracker()
    tracker.log_metrics("train", {"loss": 0}, step=1)  # int
    args = client.log_metric.call_args.args
    assert isinstance(args[2], float)


# ---------------------------------------------------------------------------
# log_params: HP audit applied
# ---------------------------------------------------------------------------


def test_log_params_filters_via_hp_audit() -> None:
    tracker, client = _make_tracker()
    kept = tracker.log_params(
        {
            "lr": 1e-3,
            "mlflow_tracking_uri": "http://example",
            "interpolation": "bilinear",
            "manifest_hash": "abc",
        }
    )
    assert kept == {"lr": 1e-3, "manifest_hash": "abc"}
    logged_keys = {c.args[1] for c in client.log_param.call_args_list}
    assert logged_keys == {"lr", "manifest_hash"}


# ---------------------------------------------------------------------------
# log_artifact: file vs directory dispatch
# ---------------------------------------------------------------------------


def test_log_artifact_dispatches_file(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("hi")
    tracker, client = _make_tracker()
    tracker.log_artifact(f, dest_dir="sub")
    client.log_artifact.assert_called_once_with("run-123", str(f), artifact_path="sub")
    client.log_artifacts.assert_not_called()


def test_log_artifact_dispatches_directory(tmp_path: Path) -> None:
    d = tmp_path / "outdir"
    d.mkdir()
    (d / "a.txt").write_text("a")
    tracker, client = _make_tracker()
    tracker.log_artifact(d)
    client.log_artifacts.assert_called_once_with("run-123", str(d), artifact_path=None)
    client.log_artifact.assert_not_called()


# ---------------------------------------------------------------------------
# log_config: writes YAML artifact
# ---------------------------------------------------------------------------


def test_log_config_writes_yaml_artifact() -> None:
    tracker, client = _make_tracker()
    cfg: dict[str, Any] = {
        "model": {"name": "arcface", "embedding_dim": 512},
        "trainer": {"epochs": 100, "lr": 1e-3},
    }

    captured: dict[str, str] = {}

    def _fake_log_artifact(run_id: str, local_path: str, artifact_path: Any = None) -> None:
        captured["run_id"] = run_id
        captured["local_path"] = local_path
        captured["artifact_path"] = artifact_path
        # Verify the file was actually written and contains our cfg.
        captured["content"] = Path(local_path).read_text(encoding="utf-8")

    client.log_artifact.side_effect = _fake_log_artifact

    tracker.log_config(cfg)

    assert captured["run_id"] == "run-123"
    assert Path(captured["local_path"]).name == "config.yaml"
    assert captured["artifact_path"] is None
    parsed = yaml.safe_load(captured["content"])
    assert parsed == cfg


# ---------------------------------------------------------------------------
# promote_to_champion: criterion comparison
# ---------------------------------------------------------------------------


def _candidate_run(run_id: str, criterion: str, value: float) -> SimpleNamespace:
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(metrics={criterion: value}),
    )


def _model_version(name: str, version: str, run_id: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, version=version, run_id=run_id)


def test_promote_returns_true_when_candidate_beats_incumbent() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = _fake_experiment()
    candidate = _candidate_run("run-NEW", "test/roc_auc", 0.97)
    incumbent = _candidate_run("run-OLD", "test/roc_auc", 0.91)

    def get_run_side(rid: str) -> SimpleNamespace:
        return candidate if rid == "run-NEW" else incumbent

    client.get_run.side_effect = get_run_side
    client.search_model_versions.return_value = [_model_version("arcface", "5", "run-NEW")]
    client.get_latest_versions.return_value = [_model_version("arcface", "4", "run-OLD")]

    tracker = MLflowTracker(experiment_name="exp", client=client)
    result = tracker.promote_to_champion("run-NEW", "test/roc_auc")

    assert result is True
    client.transition_model_version_stage.assert_called_once()
    kw = client.transition_model_version_stage.call_args.kwargs
    assert kw["name"] == "arcface"
    assert kw["version"] == "5"
    assert kw["stage"] == "Production"
    assert kw["archive_existing_versions"] is True


def test_promote_returns_false_when_candidate_loses() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = _fake_experiment()
    candidate = _candidate_run("run-NEW", "test/roc_auc", 0.85)
    incumbent = _candidate_run("run-OLD", "test/roc_auc", 0.91)

    client.get_run.side_effect = lambda rid: candidate if rid == "run-NEW" else incumbent
    client.search_model_versions.return_value = [_model_version("arcface", "5", "run-NEW")]
    client.get_latest_versions.return_value = [_model_version("arcface", "4", "run-OLD")]

    tracker = MLflowTracker(experiment_name="exp", client=client)
    assert tracker.promote_to_champion("run-NEW", "test/roc_auc") is False
    client.transition_model_version_stage.assert_not_called()


def test_promote_returns_false_when_candidate_ties() -> None:
    """Equal scores do not unseat the incumbent."""
    client = MagicMock()
    client.get_experiment_by_name.return_value = _fake_experiment()
    candidate = _candidate_run("run-NEW", "test/roc_auc", 0.91)
    incumbent = _candidate_run("run-OLD", "test/roc_auc", 0.91)

    client.get_run.side_effect = lambda rid: candidate if rid == "run-NEW" else incumbent
    client.search_model_versions.return_value = [_model_version("arcface", "5", "run-NEW")]
    client.get_latest_versions.return_value = [_model_version("arcface", "4", "run-OLD")]

    tracker = MLflowTracker(experiment_name="exp", client=client)
    assert tracker.promote_to_champion("run-NEW", "test/roc_auc") is False
    client.transition_model_version_stage.assert_not_called()


def test_promote_promotes_when_no_incumbent() -> None:
    """First-ever Production deployment has no incumbent to beat."""
    client = MagicMock()
    client.get_experiment_by_name.return_value = _fake_experiment()
    candidate = _candidate_run("run-NEW", "test/roc_auc", 0.7)
    client.get_run.return_value = candidate
    client.search_model_versions.return_value = [_model_version("arcface", "1", "run-NEW")]
    client.get_latest_versions.return_value = []  # no Production stage yet

    tracker = MLflowTracker(experiment_name="exp", client=client)
    assert tracker.promote_to_champion("run-NEW", "test/roc_auc") is True
    client.transition_model_version_stage.assert_called_once()


def test_promote_returns_false_when_criterion_missing() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = _fake_experiment()
    client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-NEW"),
        data=SimpleNamespace(metrics={}),  # no criterion logged
    )
    tracker = MLflowTracker(experiment_name="exp", client=client)
    assert tracker.promote_to_champion("run-NEW", "test/roc_auc") is False
    client.transition_model_version_stage.assert_not_called()


def test_promote_returns_false_when_no_registered_version() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = _fake_experiment()
    client.get_run.return_value = _candidate_run("run-NEW", "test/roc_auc", 0.99)
    client.search_model_versions.return_value = []
    tracker = MLflowTracker(experiment_name="exp", client=client)
    assert tracker.promote_to_champion("run-NEW", "test/roc_auc") is False
    client.transition_model_version_stage.assert_not_called()


# ---------------------------------------------------------------------------
# would_promote: pure predicate (no transition)
# ---------------------------------------------------------------------------


def test_would_promote_reports_beats_and_metrics_when_better() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = _fake_experiment()
    candidate = _candidate_run("run-NEW", "test/roc_auc", 0.94)
    incumbent = _candidate_run("run-OLD", "test/roc_auc", 0.88)
    client.get_run.side_effect = lambda rid: candidate if rid == "run-NEW" else incumbent
    client.search_model_versions.return_value = [_model_version("arcface", "5", "run-NEW")]
    client.get_latest_versions.return_value = [_model_version("arcface", "4", "run-OLD")]

    tracker = MLflowTracker(experiment_name="exp", client=client)
    beats, candidate_metric, incumbent_metric = tracker.would_promote("run-NEW", "test/roc_auc")

    assert beats is True
    assert candidate_metric == pytest.approx(0.94)
    assert incumbent_metric == pytest.approx(0.88)
    client.transition_model_version_stage.assert_not_called()


def test_would_promote_reports_loss_when_candidate_worse() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = _fake_experiment()
    candidate = _candidate_run("run-NEW", "test/roc_auc", 0.7)
    incumbent = _candidate_run("run-OLD", "test/roc_auc", 0.92)
    client.get_run.side_effect = lambda rid: candidate if rid == "run-NEW" else incumbent
    client.search_model_versions.return_value = [_model_version("arcface", "5", "run-NEW")]
    client.get_latest_versions.return_value = [_model_version("arcface", "4", "run-OLD")]

    tracker = MLflowTracker(experiment_name="exp", client=client)
    beats, candidate_metric, incumbent_metric = tracker.would_promote("run-NEW", "test/roc_auc")

    assert beats is False
    assert candidate_metric == pytest.approx(0.7)
    assert incumbent_metric == pytest.approx(0.92)
    client.transition_model_version_stage.assert_not_called()


def test_would_promote_no_incumbent_returns_beats_with_none_incumbent() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = _fake_experiment()
    client.get_run.return_value = _candidate_run("run-NEW", "test/roc_auc", 0.5)
    client.search_model_versions.return_value = [_model_version("arcface", "1", "run-NEW")]
    client.get_latest_versions.return_value = []

    tracker = MLflowTracker(experiment_name="exp", client=client)
    beats, candidate_metric, incumbent_metric = tracker.would_promote("run-NEW", "test/roc_auc")

    assert beats is True
    assert candidate_metric == pytest.approx(0.5)
    assert incumbent_metric is None
    client.transition_model_version_stage.assert_not_called()


def test_would_promote_missing_criterion_returns_no_metrics() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = _fake_experiment()
    client.get_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="run-NEW"),
        data=SimpleNamespace(metrics={}),
    )
    tracker = MLflowTracker(experiment_name="exp", client=client)
    beats, candidate_metric, incumbent_metric = tracker.would_promote("run-NEW", "test/roc_auc")

    assert beats is False
    assert candidate_metric is None
    assert incumbent_metric is None


# ---------------------------------------------------------------------------
# Active-run resolution
# ---------------------------------------------------------------------------


def test_log_metrics_uses_active_run_when_no_explicit_run_id() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = _fake_experiment()
    tracker = MLflowTracker(experiment_name="exp", client=client)  # no run_id

    fake_active = SimpleNamespace(info=SimpleNamespace(run_id="active-9"))
    with patch("bat_tracking.mlflow_tracker.mlflow.active_run", return_value=fake_active):
        tracker.log_metrics("train", {"loss": 0.1}, step=0)

    client.log_metric.assert_called_once_with("active-9", "train/loss", 0.1, step=0)


def test_log_metrics_raises_when_no_active_or_explicit_run() -> None:
    client = MagicMock()
    client.get_experiment_by_name.return_value = _fake_experiment()
    tracker = MLflowTracker(experiment_name="exp", client=client)

    with (
        patch("bat_tracking.mlflow_tracker.mlflow.active_run", return_value=None),
        pytest.raises(RuntimeError),
    ):
        tracker.log_metrics("train", {"loss": 0.1}, step=0)
