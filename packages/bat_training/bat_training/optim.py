"""Optimizer + warmup-cosine LR + gradient-accumulation helpers.

These are pure utility functions; the trainers call them, but they do not
own any training state.

Public surface:

* :func:`build_optimizer`   -- factory for ``Adam`` / ``AdamW`` / ``SGD``
  configured with ``lr`` and ``weight_decay``.
* :func:`warmup_cosine_lr`  -- a closed-form schedule
  ``warmup_steps``-long linear ramp from ``0 -> lr`` then a cosine decay
  to ``min_lr`` over the remaining ``total_steps - warmup_steps``.
* :func:`build_scheduler`   -- wraps :func:`warmup_cosine_lr` in a
  ``torch.optim.lr_scheduler.LambdaLR`` so it slots into
  ``Accelerator.prepare(scheduler=...)``.
* :class:`GradientAccumulator` -- tiny helper tracking the current
  micro-batch index and exposing ``.should_step``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover -- typing-only imports
    import torch
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LambdaLR

OptimName = Literal["adam", "adamw", "sgd"]


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    name: OptimName = "adam",
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Optimizer:
    """Build a :class:`torch.optim.Optimizer` from a flat name.

    Notes
    -----
    The ``weight_decay`` hyperparameter is the canonical L2 knob; for
    Siamese parity with the TF ``l2(1e-4)`` regularizer, the trainer
    forwards ``model.recommended_weight_decay`` (``1e-4``) here when the
    attribute is present.
    """
    import torch

    name_l = name.lower()
    if name_l == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    if name_l == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    if name_l == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError(f"unknown optimizer {name!r}; expected adam | adamw | sgd")


def warmup_cosine_lr(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> float:
    """Return the LR multiplier for the given ``step``.

    * ``step in [0, warmup_steps)``: linear ramp from 0 -> 1.
    * ``step in [warmup_steps, total_steps)``: cosine decay from 1 ->
      ``min_lr_ratio``.
    * ``step >= total_steps``: clamped at ``min_lr_ratio``.
    """
    if warmup_steps < 0 or total_steps <= 0:
        raise ValueError(
            f"warmup_steps must be >=0 and total_steps must be >0; "
            f"got warmup={warmup_steps}, total={total_steps}"
        )
    if min_lr_ratio < 0.0 or min_lr_ratio > 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1]; got {min_lr_ratio}")

    if step < 0:
        return 0.0
    if step < warmup_steps:
        # Linear from 0 (at step=0) up to 1 (at step=warmup_steps).
        return float(step + 1) / float(max(1, warmup_steps))
    if step >= total_steps:
        return float(min_lr_ratio)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def build_scheduler(
    optimizer: Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Wrap :func:`warmup_cosine_lr` into a :class:`LambdaLR`."""
    import torch

    def _fn(step: int) -> float:
        return warmup_cosine_lr(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_fn)


class GradientAccumulator:
    """Tracks micro-batch progress for gradient-accumulation training.

    Usage::

        acc = GradientAccumulator(steps=4)
        for batch in loader:
            loss = compute_loss(batch) / acc.steps
            accelerator.backward(loss)
            acc.tick()
            if acc.should_step:
                optimizer.step()
                optimizer.zero_grad()

    The accumulator owns no torch state; it is purely a counter. The trainer
    is responsible for actually scaling the loss by ``1 / steps`` and calling
    ``optimizer.step``.
    """

    def __init__(self, steps: int = 1) -> None:
        if steps < 1:
            raise ValueError(f"steps must be >= 1; got {steps}")
        self.steps = int(steps)
        self._counter = 0

    def tick(self) -> None:
        """Advance the micro-batch counter."""
        self._counter += 1

    def reset(self) -> None:
        self._counter = 0

    @property
    def should_step(self) -> bool:
        """True when the optimizer should step (every ``steps`` ticks)."""
        return (self._counter % self.steps) == 0

    @property
    def counter(self) -> int:
        return self._counter


__all__ = [
    "GradientAccumulator",
    "OptimName",
    "build_optimizer",
    "build_scheduler",
    "warmup_cosine_lr",
]
