"""End-to-end Optuna sweep orchestrator.

:func:`run_sweep` glues :func:`make_study`, :func:`build_objective` and the
caller-supplied ``build_components`` factory together and finalises by
logging the champion (best) trial back through a :class:`bat_core.Tracker`
(typically :class:`bat_tracking.MLflowTracker`) so the sweep result is
discoverable from MLflow alongside the per-trial training runs.

Cfg shape (mirrors ``configs/sweep/arcface.yaml``)::

    study_name: arcface_search
    n_trials: 20
    direction: maximize
    target_metric: val/roc_auc
    pruner: median
    storage: sqlite:///optuna_studies.db
    search_space:
      loss.margin: { type: float, low: 0.3, high: 0.6 }
      loss.scale:  { type: float, low: 32.0, high: 96.0 }
      trainer.lr:  { type: loguniform, low: 1.0e-3, high: 3.0e-1 }
    base_cfg:    # merged with each trial's sampled overrides
      loss: { ... }
      model: { ... }
      trainer: { ... }

The ``base_cfg`` block is what gets handed to ``build_components`` after
the per-trial overrides are merged in. Everything else is sweep-runtime
configuration consumed by :mod:`bat_sweeps` itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from bat_sweeps.objective import BuildComponentsFn, build_objective
from bat_sweeps.study import DEFAULT_STORAGE, make_study

if TYPE_CHECKING:  # pragma: no cover -- typing only
    import optuna


ChampionLoggerFn = Callable[[Any, Any], None]
"""Signature: ``(study, tracker) -> None``. Custom champion-logging hook."""


def _default_log_champion(study: optuna.Study, tracker: Any | None) -> None:
    """Default champion logger.

    Logs the best trial's params (under section ``"val"``-disjoint --
    they're flushed via ``log_config``) plus the best value (as a
    ``val/`` metric at step ``0``). Quietly no-ops if ``tracker`` is
    ``None`` or no trial has completed.
    """
    if tracker is None:
        return
    try:
        best = study.best_trial
    except ValueError:
        # No completed trials -- nothing to promote.
        return

    payload: dict[str, Any] = {
        "study_name": study.study_name,
        "best_value": float(best.value) if best.value is not None else None,
        "best_trial_number": int(best.number),
        "best_params": dict(best.params),
    }
    # ``log_config`` snapshots the dict as ``config.yaml`` -- handy for
    # reading the champion's params back out of MLflow.
    if hasattr(tracker, "log_config"):
        tracker.log_config(payload)
    if hasattr(tracker, "log_metrics") and best.value is not None:
        tracker.log_metrics(
            section="val",
            metrics={"champion_value": float(best.value)},
            step=0,
        )


def run_sweep(
    cfg: dict[str, Any],
    build_components: BuildComponentsFn,
    *,
    champion_tracker: Any | None = None,
    log_champion: ChampionLoggerFn | None = None,
    n_jobs: int = 1,
) -> optuna.Study:
    """Run an Optuna sweep end-to-end and return the resulting study.

    Args:
        cfg: A dict matching the schema in this module's docstring.
        build_components: Factory that turns a merged cfg dict into a
            :class:`bat_sweeps.TrialComponents`. Same callable as the one
            handed to :func:`build_objective`.
        champion_tracker: Optional :class:`bat_core.Tracker` to receive
            the champion run (typically a fresh MLflow run). When
            provided, ``log_champion`` is called once after
            ``study.optimize`` finishes.
        log_champion: Custom champion-logging hook. ``None`` falls back
            to :func:`_default_log_champion`.
        n_jobs: Forwarded to :meth:`optuna.Study.optimize`.

    Returns:
        The completed :class:`optuna.Study`.
    """
    study_name = cfg.get("study_name") or "bat_sweep"
    storage = cfg.get("storage", DEFAULT_STORAGE)
    direction = cfg.get("direction", "maximize")
    pruner = cfg.get("pruner", "median")
    n_trials = int(cfg.get("n_trials", 1))
    target_metric = cfg.get("target_metric", "val/roc_auc")
    search_space = cfg.get("search_space") or {}
    base_cfg = cfg.get("base_cfg") or {}

    study = make_study(
        name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
    )

    objective = build_objective(
        base_cfg=base_cfg,
        search_space=search_space,
        build_components=build_components,
        target_metric=target_metric,
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    logger = log_champion if log_champion is not None else _default_log_champion
    logger(study, champion_tracker)

    return study


__all__ = ["ChampionLoggerFn", "run_sweep"]
