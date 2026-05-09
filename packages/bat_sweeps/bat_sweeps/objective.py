"""Build an Optuna ``objective`` callable from a Hydra-style cfg dict.

The objective:

1. Samples search-space params from the trial (via :func:`parse_search_space`).
2. Deep-merges them into a copy of ``base_cfg`` (dotted-key overrides).
3. Calls a user-supplied factory ``build_components(merged_cfg)`` that
   returns a :class:`TrialComponents` carrying a model, loss,
   :class:`bat_training.TrainerConfig`, and the train/val loaders. The
   factory is the seam between :mod:`bat_sweeps` and the rest of the
   workspace -- :mod:`bat_sweeps` deliberately avoids importing
   :mod:`bat_models` / :mod:`bat_losses` / :mod:`bat_data` so it can run
   on minimal environments.
4. Wraps the (optional) inner tracker in :class:`OptunaPruningTracker` so
   each per-epoch ``log_metrics(section="val", ...)`` triggers
   ``trial.report(value, step=epoch)`` and ``trial.should_prune()``.
5. Builds a trainer via :func:`bat_training.make_trainer` and runs
   :meth:`Trainer.fit`. **The objective never calls ``trainer.test()``**
   -- the test split is sacred and lives outside the sweep.
6. Returns the final ``val/<target_metric>`` value (the median pruner
   uses the per-epoch ``trial.report`` calls; the returned value is the
   one Optuna optimises against).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bat_sweeps.pruning_callback import OptunaPruningCallback, OptunaPruningTracker
from bat_sweeps.search_space import SearchSpaceSpec, apply_overrides, parse_search_space

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from optuna.trial import Trial


@dataclass
class TrialComponents:
    """Bundle of artefacts the objective needs to run a single trial.

    The ``build_components`` factory the caller supplies is responsible
    for constructing every field. Attributes:

    * ``model``: a :class:`bat_core.FaceModel`-shaped ``nn.Module``.
    * ``loss``: a :class:`bat_core.Loss` (its ``family`` decides the trainer).
    * ``trainer_cfg``: a :class:`bat_training.TrainerConfig`.
    * ``train_loader`` / ``val_loader``: PyTorch loaders (or any iterable
      the trainer accepts). The val loader is **never** the test loader.
    * ``tracker``: optional inner :class:`bat_core.Tracker`. The objective
      wraps it in :class:`OptunaPruningTracker`.
    * ``trainer_kwargs``: extra kwargs forwarded into
      :func:`bat_training.make_trainer` (e.g. ``eval_manifest``).
    """

    model: Any
    loss: Any
    trainer_cfg: Any
    train_loader: Any
    val_loader: Any
    tracker: Any | None = None
    trainer_kwargs: dict[str, Any] | None = None


BuildComponentsFn = Callable[[dict[str, Any]], TrialComponents]
ObjectiveFn = Callable[["Trial"], float]


def _strip_val_prefix(metric: str) -> str:
    """Return the bare metric key (drops a leading ``val/``)."""
    return metric[len("val/") :] if metric.startswith("val/") else metric


def _extract_final_value(
    artifacts: Any,
    bare_metric: str,
    callback: OptunaPruningCallback,
) -> float:
    """Pull the optimisation value out of ``RunArtifacts`` (or fallback).

    Preference order:

    1. ``artifacts.best_metrics[bare_metric]`` if present.
    2. ``artifacts.final_metrics[bare_metric]`` if present.
    3. The last value the pruning callback observed (fallback for
       trainers that don't surface the metric in ``RunArtifacts``).

    Raises:
        ValueError: if no value can be located.
    """
    best = getattr(artifacts, "best_metrics", None) or {}
    final = getattr(artifacts, "final_metrics", None) or {}
    if bare_metric in best:
        return float(best[bare_metric])
    if bare_metric in final:
        return float(final[bare_metric])
    if callback.last_value is not None:
        return float(callback.last_value)
    raise ValueError(
        f"objective could not extract {bare_metric!r} from RunArtifacts; "
        "ensure the trainer's tracker emits it or that best_metrics carries it"
    )


def build_objective(
    base_cfg: dict[str, Any],
    search_space: SearchSpaceSpec,
    build_components: BuildComponentsFn,
    target_metric: str = "val/roc_auc",
) -> ObjectiveFn:
    """Compile an Optuna objective from a base cfg + search space.

    Args:
        base_cfg: The Hydra-style config dict the trial copies + overrides.
        search_space: A :data:`SearchSpaceSpec` (see
            :func:`bat_sweeps.parse_search_space`).
        build_components: Factory callable that takes a merged cfg dict
            and returns a :class:`TrialComponents`. This is the only seam
            into the rest of the workspace.
        target_metric: The val-section metric to optimise. Must start
            with ``val/`` (test/* is rejected -- the test split is
            sacred and never read by the sweep).

    Returns:
        A callable ``objective(trial) -> float`` ready for
        ``study.optimize(...)``.
    """
    if target_metric.startswith("test/"):
        raise ValueError(
            f"target_metric must reference val/* (test split is sacred); got {target_metric!r}"
        )
    if not target_metric.startswith("val/"):
        # Be permissive for callers passing bare metric names.
        target_metric = f"val/{target_metric}"

    bare_metric = _strip_val_prefix(target_metric)
    sampler = parse_search_space(search_space)

    def _objective(trial: Trial) -> float:
        from bat_training.factory import make_trainer

        overrides = sampler(trial)
        merged = apply_overrides(base_cfg, overrides)
        components = build_components(merged)

        callback = OptunaPruningCallback(trial=trial, target_metric=target_metric)
        wrapped_tracker = OptunaPruningTracker(inner=components.tracker, callback=callback)

        trainer_kwargs = dict(components.trainer_kwargs or {})
        trainer_kwargs.setdefault("tracker", wrapped_tracker)
        # Force the wrapper even if the caller passed an explicit tracker;
        # we wrapped their tracker as inner above.
        trainer_kwargs["tracker"] = wrapped_tracker

        trainer = make_trainer(
            cfg=components.trainer_cfg,
            model=components.model,
            loss=components.loss,
            **trainer_kwargs,
        )

        artifacts = trainer.fit(components.train_loader, components.val_loader)

        # Sweep MUST NOT call trainer.test() -- the test split is sacred
        # and used only by the post-sweep champion evaluation.
        return _extract_final_value(artifacts, bare_metric, callback)

    return _objective


__all__ = [
    "BuildComponentsFn",
    "ObjectiveFn",
    "TrialComponents",
    "build_objective",
]
