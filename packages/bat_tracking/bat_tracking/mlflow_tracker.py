"""MLflow facade implementing the :class:`bat_core.Tracker` protocol.

Wraps :class:`mlflow.MlflowClient` so the rest of the workspace programs
against a single, stable interface. Highlights:

* Metrics are *sectioned* --- every key is prefixed with ``train/``, ``val/``
  or ``test/`` so the MLflow UI groups them automatically.
* Params are filtered through :func:`bat_tracking.hp_audit.filter_params`
  before being sent, so HP audit is enforced at log time.
* Hydra configs are snapshotted as a YAML artifact (``config.yaml``).
* Champion promotion uses
  :meth:`mlflow.MlflowClient.transition_model_version_stage` to move a
  registered model version to "Production" once it beats the incumbent on the
  given criterion.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mlflow
import yaml
from bat_tracking.hp_audit import filter_params
from mlflow.tracking import MlflowClient

if TYPE_CHECKING:
    from bat_core.interfaces import TrackerSection


_PROD_STAGE = "Production"


class MLflowTracker:
    """Concrete :class:`bat_core.Tracker` backed by an MLflow client.

    Parameters
    ----------
    experiment_name:
        MLflow experiment name. Created if it does not exist.
    tracking_uri:
        Optional MLflow tracking URI. ``None`` defers to MLflow's default
        (``file:./mlruns``). Note: while ``mlflow_tracking_uri`` is *banned*
        as a logged HP (see :mod:`bat_tracking.hp_audit`), it is fine as a
        constructor argument --- it stays out of the run's params.
    run_id:
        Optional explicit run id. When provided the tracker logs into the
        existing run rather than relying on the active MLflow run. Tests use
        this to assert against mocks; the :func:`bat_tracking.run_context`
        context manager wires it automatically.
    client:
        Optional pre-built :class:`MlflowClient`. Primarily a test seam.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        run_id: str | None = None,
        client: MlflowClient | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._run_id = run_id

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        self._client: MlflowClient = (
            client if client is not None else MlflowClient(tracking_uri=tracking_uri)
        )

        # Resolve experiment_id eagerly so failures surface at construction time.
        exp = self._client.get_experiment_by_name(experiment_name)
        if exp is None:
            self.experiment_id: str = self._client.create_experiment(experiment_name)
        else:
            self.experiment_id = exp.experiment_id

    # ------------------------------------------------------------------ helpers
    def _resolve_run_id(self) -> str:
        """Return the run id to log against. Prefers the explicit one."""
        if self._run_id is not None:
            return self._run_id
        active = mlflow.active_run()
        if active is None:
            raise RuntimeError(
                "MLflowTracker has no run_id and no active mlflow run; "
                "wrap calls in `with start_run(...) as tracker:` or pass run_id."
            )
        return active.info.run_id

    def set_run_id(self, run_id: str) -> None:
        """Bind subsequent logging calls to *run_id* (used by start_run)."""
        self._run_id = run_id

    # ------------------------------------------------------- Tracker protocol
    def log_metrics(
        self,
        section: TrackerSection,
        metrics: dict[str, float],
        step: int,
    ) -> None:
        """Log *metrics* under the given section, prefixing every key.

        e.g. ``log_metrics("train", {"loss": 0.5}, step=3)`` calls
        ``client.log_metric(run_id, "train/loss", 0.5, step=3)``.
        """
        run_id = self._resolve_run_id()
        for raw_key, value in metrics.items():
            prefixed = f"{section}/{raw_key}"
            self._client.log_metric(run_id, prefixed, float(value), step=step)

    def log_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Log HP-audited params and return the filtered dict actually sent."""
        run_id = self._resolve_run_id()
        kept = filter_params(params)
        for k, v in kept.items():
            # MLflow stringifies params anyway; we coerce explicitly to keep
            # the behaviour deterministic across mlflow versions.
            self._client.log_param(run_id, k, _stringify(v))
        return kept

    def log_artifact(self, path: Path, dest_dir: str | None = None) -> None:
        """Upload a file or directory to the run's artifact store."""
        run_id = self._resolve_run_id()
        p = Path(path)
        if p.is_dir():
            self._client.log_artifacts(run_id, str(p), artifact_path=dest_dir)
        else:
            self._client.log_artifact(run_id, str(p), artifact_path=dest_dir)

    def log_config(self, cfg: dict[str, Any]) -> None:
        """Snapshot a (Hydra) config dict as ``config.yaml`` artifact."""
        run_id = self._resolve_run_id()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            self._client.log_artifact(run_id, str(cfg_path), artifact_path=None)

    def would_promote(self, run_id: str, criterion: str) -> tuple[bool, float | None, float | None]:
        """Return ``(beats, candidate_metric, incumbent_metric)`` without transitioning.

        Mirrors the comparison performed by :meth:`promote_to_champion` but
        does not call ``transition_model_version_stage``. Useful for CLI
        flows that want to prompt the user before mutating the registry.

        ``incumbent_metric`` is ``None`` if no Production-stage version
        exists yet (in which case ``beats`` is True if the candidate has any
        value for ``criterion``). ``candidate_metric`` is ``None`` if the
        run has no value for ``criterion`` (``beats`` is False in that case).
        """
        candidate_run = self._client.get_run(run_id)
        candidate_metric = candidate_run.data.metrics.get(criterion)
        if candidate_metric is None:
            return False, None, None

        mv = _find_model_version_for_run(self._client, run_id)
        if mv is None:
            return False, candidate_metric, None

        incumbents = self._client.get_latest_versions(mv.name, stages=[_PROD_STAGE])
        if not incumbents:
            return True, candidate_metric, None

        incumbent_metric: float | None = None
        for inc in incumbents:
            inc_run = self._client.get_run(inc.run_id)
            value = inc_run.data.metrics.get(criterion)
            if value is None:
                continue
            if incumbent_metric is None or value > incumbent_metric:
                incumbent_metric = value
        if incumbent_metric is None:
            return True, candidate_metric, None
        return candidate_metric > incumbent_metric, candidate_metric, incumbent_metric

    def promote_to_champion(self, run_id: str, criterion: str) -> bool:
        """Promote *run_id*'s registered model to Production if it wins.

        The check compares ``run.data.metrics[criterion]`` of the candidate
        against the same metric on the run that produced the current
        Production-stage model version. Higher is better. Returns True iff a
        transition was performed.

        The model name is read from the run's ``mlflow.runName`` tag's
        ``registered_model`` companion tag, falling back to the run's name.
        Concretely, the trainer is expected to have called
        :func:`bat_tracking.registry.register_model` already; this method only
        performs the *stage transition*.
        """
        beats, _candidate, _incumbent = self.would_promote(run_id, criterion)
        if not beats:
            return False

        mv = _find_model_version_for_run(self._client, run_id)
        if mv is None:
            return False

        self._client.transition_model_version_stage(
            name=mv.name,
            version=mv.version,
            stage=_PROD_STAGE,
            archive_existing_versions=True,
        )
        return True


def _stringify(v: Any) -> str:
    """Stable repr for MLflow params (handles bool/None/numbers/strings)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v)


def _find_model_version_for_run(client: MlflowClient, run_id: str) -> Any | None:
    """Locate the most recent ModelVersion produced by *run_id*."""
    try:
        results = client.search_model_versions(f"run_id='{run_id}'")
    except Exception:  # pragma: no cover - mlflow backend variance
        return None
    if not results:
        return None
    # Highest version number wins.
    return cast(Any, max(results, key=lambda mv: int(mv.version)))


__all__ = ["MLflowTracker"]
