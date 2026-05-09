"""Optuna pruning hook for :mod:`bat_training`.

Design note (trainer-seam adaptation)
-------------------------------------

:class:`bat_training.PairTrainer` and :class:`bat_training.EmbeddingTrainer`
do not expose an explicit per-epoch callback list -- the only public seam
that fires once per validation pass is::

    self.tracker.log_metrics(section="val", metrics={...}, step=epoch)

So the pruning hook is implemented as a **tracker wrapper** rather than a
free-standing trainer callback. :class:`OptunaPruningTracker` proxies every
:class:`bat_core.Tracker` method through to an inner tracker (or no-ops if
none is provided), and additionally fires ``trial.report(value, step)`` +
``trial.should_prune()`` whenever a ``section="val"`` payload contains the
target metric. If the trial should be pruned the wrapper raises
:class:`optuna.TrialPruned`, which the trainer's caller (the objective) is
responsible for catching.

A standalone :class:`OptunaPruningCallback` is also exported with a
``on_validation_end(epoch, metrics)`` API in case ``bat_training`` ever
exposes a callback list (TODO(phase-2.1)) -- the wrapper delegates to the
callback so the two stay in lockstep.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from optuna.trial import Trial


class OptunaPruningCallback:
    """Per-epoch validation-end hook that reports to an Optuna trial.

    Args:
        trial: The active :class:`optuna.trial.Trial`.
        target_metric: Either a bare metric key (e.g. ``"roc_auc"``) or a
            sectioned key (``"val/roc_auc"``). The ``"val/"`` prefix is
            optional and stripped during lookup -- val is the only
            section the pruner ever consults (test stays sacred).

    Notes:
        ``on_validation_end`` raises :class:`optuna.TrialPruned` when the
        median pruner says so. The objective wraps :meth:`Trainer.fit` in
        a ``try/except optuna.TrialPruned`` so the trial reports its
        last value before unwinding.
    """

    def __init__(self, trial: Trial, target_metric: str = "val/roc_auc") -> None:
        self.trial = trial
        # Accept both "val/roc_auc" and "roc_auc". val/test sectioning
        # lives in the tracker, not in the metric dict the trainer hands us.
        if target_metric.startswith("val/"):
            target_metric = target_metric[len("val/") :]
        elif target_metric.startswith("test/"):
            raise ValueError(
                f"target_metric must reference val/* (test split is sacred); got {target_metric!r}"
            )
        self.target_metric: str = target_metric
        self._last_value: float | None = None

    @property
    def last_value(self) -> float | None:
        """The most recent value reported to Optuna (``None`` if never)."""
        return self._last_value

    def on_validation_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Report ``metrics[target_metric]`` to the trial and maybe prune.

        Raises:
            optuna.TrialPruned: when the pruner decides to abort.
        """
        if self.target_metric not in metrics:
            return
        value = float(metrics[self.target_metric])
        self._last_value = value
        self.trial.report(value, step=int(epoch))
        if self.trial.should_prune():
            import optuna

            raise optuna.TrialPruned(
                f"trial {self.trial.number} pruned at epoch {epoch} "
                f"with {self.target_metric}={value:.6f}"
            )


class OptunaPruningTracker:
    """Tracker wrapper that fans out into an :class:`OptunaPruningCallback`.

    Implements :class:`bat_core.Tracker`. Wraps an inner tracker (or
    ``None`` for a no-op shell) and intercepts ``log_metrics(section="val", ...)``
    to call :meth:`OptunaPruningCallback.on_validation_end`.

    The non-val sections (``train`` / ``test``) and the other tracker
    methods (``log_artifact``, ``log_config``, ``promote_to_champion``)
    pass through untouched. ``test`` payloads in particular are never
    inspected -- test split is sacred.
    """

    def __init__(
        self,
        inner: Any | None,
        callback: OptunaPruningCallback,
    ) -> None:
        self._inner = inner
        self.callback = callback

    # ----------------------------- bat_core.Tracker
    def log_metrics(
        self,
        section: str,
        metrics: dict[str, float],
        step: int,
    ) -> None:
        if self._inner is not None:
            self._inner.log_metrics(section=section, metrics=metrics, step=step)
        if section == "val":
            self.callback.on_validation_end(epoch=step, metrics=metrics)

    def log_artifact(self, path: Path, dest_dir: str | None = None) -> None:
        if self._inner is not None:
            self._inner.log_artifact(path, dest_dir)

    def log_config(self, cfg: dict[str, Any]) -> None:
        if self._inner is not None:
            self._inner.log_config(cfg)

    def promote_to_champion(self, run_id: str, criterion: str) -> bool:
        if self._inner is None:
            return False
        return bool(self._inner.promote_to_champion(run_id, criterion))


__all__ = ["OptunaPruningCallback", "OptunaPruningTracker"]
