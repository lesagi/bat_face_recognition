"""Best-only checkpointing per metric.

This is a port of the ``_artifact_retention`` semantics of
``app/siamese_training/trainer.py``:

* Per-metric "save best" toggles (``save_best_f1``, ``save_best_recall``,
  ``save_best_precision``, ``save_best_loss``).
* Optional final-model dump (``save_final_model``).
* If two metrics improve in the same epoch, only one physical file is
  written; the others are symlinks. This matches the legacy
  ``_save_best_metric_artifact`` policy.

Public surface:

* :class:`BestCheckpointTracker` -- per-metric bookkeeping with
  ``mode = "max" | "min"``.
* :class:`CheckpointManager`     -- writes / overwrites
  ``{output_dir}/{name}.pt`` (or symlinks) and tracks the canonical
  on-disk file for the epoch.
* :func:`save_checkpoint` / :func:`load_checkpoint` -- low-level
  serialise / deserialise of the trainer's state dict (model, optimizer,
  scheduler, scaler, EMA, step).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover -- typing only
    import torch
    from torch import nn

Mode = Literal["max", "min"]


# ---------------------------------------------------------------------------
# Best tracker
# ---------------------------------------------------------------------------


@dataclass
class _MetricState:
    name: str
    mode: Mode
    best_value: float
    best_epoch: int = 0


class BestCheckpointTracker:
    """Track the best value of one or more metrics across epochs.

    Args:
        metrics: Mapping of ``{metric_name: "max" | "min"}``. ``"max"``
            means higher is better (e.g. f1 / recall / precision /
            roc_auc); ``"min"`` means lower is better (e.g. loss).
    """

    def __init__(self, metrics: dict[str, Mode]) -> None:
        self._states: dict[str, _MetricState] = {}
        for name, mode in metrics.items():
            if mode not in ("max", "min"):
                raise ValueError(f"mode must be 'max' or 'min'; got {mode!r}")
            self._states[name] = _MetricState(
                name=name,
                mode=mode,
                best_value=(float("-inf") if mode == "max" else float("inf")),
            )

    def update(self, name: str, value: float, epoch: int) -> bool:
        """Update tracker with a new ``value``; return True iff it improved."""
        if name not in self._states:
            return False
        state = self._states[name]
        if state.mode == "max":
            improved = value > state.best_value
        else:
            improved = value < state.best_value
        if improved:
            state.best_value = float(value)
            state.best_epoch = int(epoch)
        return improved

    def best(self, name: str) -> tuple[float, int]:
        """Return the (best_value, best_epoch) tuple for the metric."""
        s = self._states[name]
        return s.best_value, s.best_epoch

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Return ``{metric: {value, epoch}}`` for all tracked metrics."""
        return {
            n: {"value": s.best_value, "epoch": s.best_epoch}
            for n, s in self._states.items()
        }


# ---------------------------------------------------------------------------
# Checkpoint Manager (best-only files + symlink dedupe)
# ---------------------------------------------------------------------------


def _DEFAULT_RETENTION() -> dict[str, bool]:
    return {
        "save_best_f1": True,
        "save_best_recall": True,
        "save_best_precision": True,
        "save_best_loss": True,
        "save_final_model": False,
    }


class CheckpointManager:
    """Best-only checkpoint writer with same-epoch symlink dedupe.

    Behaviour ported from
    ``app/siamese_training/trainer.py:_save_best_metric_artifact``:

    The first metric to improve in a given epoch wins the canonical
    physical file. Subsequent metrics in the same epoch get a symlink
    to that file. If a metric improves at a later epoch we overwrite
    the (now stale) symlink with a fresh file.
    """

    def __init__(
        self,
        output_dir: str | Path,
        retention: dict[str, bool] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._retention = {**_DEFAULT_RETENTION(), **(retention or {})}

        # Map epoch -> the canonical file written for that epoch.
        self._epoch_canonical: dict[int, str] = {}

    @property
    def retention(self) -> dict[str, bool]:
        return dict(self._retention)

    def write_best(
        self,
        metric_name: str,
        epoch: int,
        state: dict[str, Any],
    ) -> Path | None:
        """Write a best-metric checkpoint or symlink to the epoch's canonical file.

        Args:
            metric_name: One of ``f1`` / ``recall`` / ``precision`` /
                ``loss`` (the keys consumed by the legacy retention map
                are ``save_best_<metric>``).
            epoch: Epoch number.
            state: Trainer state dict to serialise (see
                :func:`save_checkpoint`).

        Returns the on-disk path written, or ``None`` if retention is
        disabled for the metric.
        """
        retention_key = f"save_best_{metric_name}"
        if not self._retention.get(retention_key, False):
            return None

        filename = f"best_model_{metric_name}.pt"
        dest = self.output_dir / filename

        if epoch not in self._epoch_canonical:
            # First improvement this epoch: write a real file.
            _remove_path_for_resave(dest)
            save_checkpoint(state, dest)
            self._epoch_canonical[epoch] = filename
            return dest

        canon = self._epoch_canonical[epoch]
        if canon == filename:
            return dest

        # Subsequent improvement in same epoch: symlink to canonical.
        _remove_path_for_resave(dest)
        try:
            os.symlink(canon, dest)
        except OSError:  # pragma: no cover - filesystems without symlinks
            shutil.copy2(self.output_dir / canon, dest)
        return dest

    def write_final(self, state: dict[str, Any]) -> Path | None:
        """Write the final model under ``output_dir/final_model.pt`` if enabled."""
        if not self._retention.get("save_final_model", False):
            return None
        dest = self.output_dir / "final_model.pt"
        save_checkpoint(state, dest)
        return dest


def _remove_path_for_resave(path: Path) -> None:
    """Replicate the legacy :func:`_remove_path_for_resave` semantics."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


# ---------------------------------------------------------------------------
# Low-level serialise / deserialise
# ---------------------------------------------------------------------------


def save_checkpoint(
    state: dict[str, Any],
    path: str | Path,
) -> Path:
    """Serialise the trainer state via ``torch.save``.

    The expected ``state`` keys are:
      - ``model``     : ``state_dict``
      - ``optimizer`` : ``state_dict`` (optional)
      - ``scheduler`` : ``state_dict`` (optional)
      - ``scaler``    : ``state_dict`` (optional)
      - ``ema``       : EMA shadow dict (optional)
      - ``epoch``     : int
      - ``step``      : int (optional, micro-batch counter)
      - ``metrics``   : dict[str, float] (optional)
    """
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(p))
    return p


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint and return the saved dict."""
    import torch

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    # weights_only=False because we deliberately serialise EMA + scaler
    # state alongside model weights. The trainer is the trusted writer.
    try:
        return torch.load(str(p), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - torch < 2.4 lacks weights_only
        return torch.load(str(p), map_location="cpu")


def restore_into(
    state: dict[str, Any],
    *,
    model: "nn.Module | None" = None,
    optimizer: "torch.optim.Optimizer | None" = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    ema: object | None = None,
) -> None:
    """Apply a loaded ``state`` to the given runtime objects."""
    if model is not None and "model" in state:
        model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])  # type: ignore[attr-defined]
    if scaler is not None and "scaler" in state and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])  # type: ignore[attr-defined]
    if ema is not None and "ema" in state and state["ema"] is not None:
        ema.load_state_dict(state["ema"])  # type: ignore[attr-defined]


__all__ = [
    "BestCheckpointTracker",
    "CheckpointManager",
    "Mode",
    "load_checkpoint",
    "restore_into",
    "save_checkpoint",
]
