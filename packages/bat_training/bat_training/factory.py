"""Trainer factory.

:func:`make_trainer` dispatches on ``loss.family`` to the right concrete
trainer (:class:`PairTrainer` for ``"pair"``, :class:`EmbeddingTrainer` for
``"embedding"``) and validates that ``model.family == loss.family``.

The factory is intentionally tiny: it only owns the dispatch + validation
logic; everything else is in the concrete trainers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bat_core.exceptions import InterfaceViolationError
from bat_training._common import TrainerConfig
from bat_training.embedding_trainer import EmbeddingTrainer
from bat_training.pair_trainer import PairTrainer

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from bat_core import Manifest
    from bat_core.interfaces import Tracker
    from torch import nn


def make_trainer(
    cfg: TrainerConfig,
    model: nn.Module,
    loss: Any,
    *,
    tracker: Tracker | None = None,
    eval_manifest: Manifest | None = None,
    eval_split: str = "test",
    image_size: int = 224,
    accelerator: Any | None = None,
    run_id: str | None = None,
    register_name: str | None = None,
    promote_criterion: str = "test/roc_auc",
    callbacks: list[Any] | None = None,
) -> PairTrainer | EmbeddingTrainer:
    """Build the right trainer based on ``loss.family``.

    Raises :class:`bat_core.InterfaceViolationError` if ``model.family``
    and ``loss.family`` disagree, or if the family is unknown.
    """
    family = getattr(loss, "family", None)
    if family is None:
        raise InterfaceViolationError("loss.family attribute is required to dispatch a trainer")

    m_family = getattr(model, "family", None)
    if m_family is not None and m_family != family:
        raise InterfaceViolationError(
            f"family mismatch: model.family={m_family!r} but loss.family={family!r}"
        )

    if family == "pair":
        return PairTrainer(
            model=model,
            loss=loss,
            cfg=cfg,
            tracker=tracker,
            accelerator=accelerator,
            run_id=run_id,
            register_name=register_name,
            promote_criterion=promote_criterion,
            callbacks=callbacks,
        )
    if family == "embedding":
        return EmbeddingTrainer(
            model=model,
            loss=loss,
            cfg=cfg,
            eval_manifest=eval_manifest,
            eval_split=eval_split,
            image_size=image_size,
            tracker=tracker,
            accelerator=accelerator,
            run_id=run_id,
            register_name=register_name,
            promote_criterion=promote_criterion,
            callbacks=callbacks,
        )
    raise InterfaceViolationError(f"unknown loss.family={family!r}; expected 'pair' or 'embedding'")


__all__ = ["make_trainer"]
