"""Shared helpers used by both PairTrainer and EmbeddingTrainer.

These are kept on a private module so the two trainer files stay focused on
their respective loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover -- typing only
    import torch


@dataclass
class TrainerConfig:
    """Common runtime configuration consumed by both trainers.

    Every field has a sensible default so trainers can be constructed
    with minimal boilerplate in tests.
    """

    epochs: int = 1
    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    momentum: float = 0.9
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"  # "no" | "fp16" | "bf16"
    ema_decay: float = 0.0  # 0 disables EMA
    early_stopping_patience: int = 0  # 0 disables ES
    early_stopping_min_delta: float = 1e-3
    early_stopping_monitor: str = "f1"
    early_stopping_mode: str = "max"
    deterministic: bool = False
    seed: int = 0
    output_dir: Path = Path("./outputs/training")
    artifact_retention: dict[str, bool] | None = None
    log_every_n_steps: int = 50


def detach_to_float(t: "torch.Tensor | float | int") -> float:
    """Best-effort ``.item()`` for a 0-D tensor or scalar."""
    try:
        return float(t.detach().item())  # type: ignore[union-attr]
    except (AttributeError, RuntimeError):
        return float(t)  # type: ignore[arg-type]


def looks_like_pair_batch(batch: Any) -> bool:
    """Best-effort detection of pair batches.

    A pair batch is one of:
    - ``(x_a, x_b, label)``
    - ``(x_a, x_b, label, sample_weight_or_class_info)``
    - dict with keys ``{"x_a","x_b","label"}``
    """
    if isinstance(batch, dict):
        return {"x_a", "x_b", "label"}.issubset(batch.keys())
    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        # A batch with 3+ elements might be a pair batch; we trust the caller
        # to wire in the right loader. Returning True here is safe because
        # the trainers only call this for diagnostics.
        return True
    return False


def unpack_pair_batch(batch: Any) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Return ``(x_a, x_b, label)`` from a pair-shaped batch.

    Accepts:
    - ``(x_a, x_b, label)``
    - ``(x_a, x_b, label, *extra)`` -- extras are ignored
    - ``dict`` with keys ``x_a``, ``x_b``, ``label``
    """
    if isinstance(batch, dict):
        return batch["x_a"], batch["x_b"], batch["label"]
    if not isinstance(batch, (tuple, list)) or len(batch) < 3:
        raise TypeError(
            f"PairTrainer expected (x_a, x_b, label[, ...]) batch; got {type(batch).__name__}"
        )
    return batch[0], batch[1], batch[2]


def unpack_embedding_batch(batch: Any) -> "tuple[torch.Tensor, torch.Tensor]":
    """Return ``(image, identity_label)`` from an embedding-shaped batch.

    Accepts:
    - ``(image, label)``
    - ``(image, label, *extra)`` -- extras (e.g. record metadata from
      :class:`bat_data.BatDataset`) are ignored.
    - dict with keys ``image``, ``label``.
    """
    if isinstance(batch, dict):
        return batch["image"], batch["label"]
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError(
            f"EmbeddingTrainer expected (image, label[, ...]) batch; got {type(batch).__name__}"
        )
    return batch[0], batch[1]


__all__ = [
    "TrainerConfig",
    "detach_to_float",
    "looks_like_pair_batch",
    "unpack_embedding_batch",
    "unpack_pair_batch",
]
