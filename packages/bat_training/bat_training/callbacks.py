"""Callbacks for the trainers.

Currently:

* :class:`TrainerCallback` -- per-epoch validation-end hook protocol.
  Trainers iterate ``callbacks`` after each ``_log("val", ...)`` and fire
  ``cb.on_validation_end(epoch, metrics)``. Used by
  :class:`bat_sweeps.OptunaPruningCallback` to fire the Optuna pruner.
* :class:`EarlyStopping` -- patience-based monitor on a configurable
  metric. Mirrors the legacy
  ``app/siamese_training/trainer.py:_check_early_stopping`` semantics:
  best-so-far + min-delta + patience.
* :func:`set_deterministic_mode` -- toggles
  :func:`torch.use_deterministic_algorithms`, cuDNN flags, and seeds.
* :func:`worker_init_fn` -- DataLoader-friendly per-worker seed.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Mode = Literal["max", "min"]


@runtime_checkable
class TrainerCallback(Protocol):
    """Per-epoch validation-end hook trainers iterate over.

    Trainers call ``on_validation_end(epoch, metrics)`` once per epoch
    after the val ``log_metrics`` payload has been computed (``metrics``
    is the same dict logged to the ``"val"`` section). Callbacks may
    raise to abort training (e.g. :class:`optuna.TrialPruned`); the
    trainer does NOT catch — callers wrap ``fit`` appropriately.
    """

    def on_validation_end(self, epoch: int, metrics: dict[str, float]) -> None: ...


@dataclass
class EarlyStoppingState:
    """Public-facing snapshot of an early-stopping callback's state."""

    counter: int
    best_value: float
    best_epoch: int
    triggered: bool


class EarlyStopping:
    """Patience-based early-stopping monitor.

    Args:
        monitor: Metric key the trainer is responsible for forwarding
            in :meth:`step`. Defaults to ``"f1"`` (the legacy choice).
        mode: ``"max"`` (default) for higher-is-better, ``"min"`` for
            loss-style metrics.
        patience: Number of consecutive non-improving epochs before
            triggering. Default ``10``.
        min_delta: Minimum improvement required to reset the counter.
            Default ``1e-3``.
    """

    def __init__(
        self,
        monitor: str = "f1",
        mode: Mode = "max",
        patience: int = 10,
        min_delta: float = 1e-3,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min'; got {mode!r}")
        if patience < 0:
            raise ValueError(f"patience must be >= 0; got {patience}")
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be >= 0; got {min_delta}")
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self._counter = 0
        self._best_value = float("-inf") if mode == "max" else float("inf")
        self._best_epoch = 0
        self._triggered = False

    @property
    def state(self) -> EarlyStoppingState:
        return EarlyStoppingState(
            counter=self._counter,
            best_value=self._best_value,
            best_epoch=self._best_epoch,
            triggered=self._triggered,
        )

    def step(self, value: float, epoch: int) -> bool:
        """Advance the monitor; return True iff training should stop."""
        improved = self._is_improvement(value)
        if improved:
            self._best_value = float(value)
            self._best_epoch = int(epoch)
            self._counter = 0
            return False

        self._counter += 1
        if self._counter >= self.patience:
            self._triggered = True
            return True
        return False

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self._best_value + self.min_delta
        return value < self._best_value - self.min_delta


def set_deterministic_mode(seed: int = 0, *, warn_only: bool = True) -> None:
    """Enable deterministic algorithms across the stack.

    Mirrors the suggested setup in PyTorch reproducibility docs:

    1. Seeds Python / NumPy / Torch.
    2. ``torch.use_deterministic_algorithms(True)`` (with ``warn_only`` so
       operations without a deterministic implementation degrade
       gracefully rather than crashing).
    3. Disables cuDNN benchmark and enables deterministic cuDNN.
    4. Sets ``CUBLAS_WORKSPACE_CONFIG`` for deterministic GEMM.
    """
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover -- numpy is a dep
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover -- env-dependent
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:  # pragma: no cover -- torch is a dep
        pass


def worker_init_fn(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` honoring deterministic mode.

    Each worker gets a fresh torch / numpy / random seed derived from
    the parent generator's seed, so the data pipeline is reproducible.
    """
    try:
        import numpy as np
        import torch

        seed = (torch.initial_seed() + worker_id) % (2**32)
        np.random.seed(seed)
    except ImportError:  # pragma: no cover -- both are deps
        seed = worker_id
    random.seed(seed)


__all__ = [
    "EarlyStopping",
    "EarlyStoppingState",
    "Mode",
    "TrainerCallback",
    "set_deterministic_mode",
    "worker_init_fn",
]
