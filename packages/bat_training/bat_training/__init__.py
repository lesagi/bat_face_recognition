"""bat_training -- PairTrainer + EmbeddingTrainer for the bat-face-rec workspace.

Two PyTorch trainer classes (intentionally **not** unified; see the Phase-2
plan):

- :class:`PairTrainer` for ``family="pair"`` losses (BCE / Focal / Triplet).
  Trains on pairs ``(x_a, x_b, label)``; evaluates verification ROC/AUC and
  picks the optimal Youden-J threshold via :mod:`bat_evaluation.verification`.

- :class:`EmbeddingTrainer` for ``family="embedding"`` losses (ArcFace /
  AdaFace / CosFace / SubCenter). Trains on ``(image, identity_label)``; the
  loss expects margined logits from the head -- wired as
  ``head(emb, labels) -> margined_logits`` followed by ``loss(...)``.
  Evaluates both verification (probe-vs-gallery cosine) and identification
  (CMC / top-k / mAP) via :mod:`bat_evaluation.protocols`.

Both implement :class:`bat_core.Trainer`. Construct one through
:func:`make_trainer`, which dispatches on the loss's ``family`` attribute and
validates that ``model.family == loss.family``.

Composable utilities (used by both trainers):

- :mod:`bat_training.optim`           -- optimizer + warmup-cosine LR
- :mod:`bat_training.ema`             -- ExponentialMovingAverage
- :mod:`bat_training.amp`             -- thin Accelerate / autocast wrapper
- :mod:`bat_training.checkpointing`   -- best-only checkpointing per metric
- :mod:`bat_training.callbacks`       -- early stopping + deterministic flags
- :mod:`bat_training.factory`         -- ``make_trainer`` dispatcher
- :mod:`bat_training.permutation_adapter` -- adapter into ``bat_stats``
"""

from __future__ import annotations

from bat_training.amp import AMPContext
from bat_training.callbacks import EarlyStopping, set_deterministic_mode, worker_init_fn
from bat_training.checkpointing import (
    BestCheckpointTracker,
    CheckpointManager,
    load_checkpoint,
    save_checkpoint,
)
from bat_training.ema import ExponentialMovingAverage
from bat_training.embedding_trainer import EmbeddingTrainer
from bat_training.factory import make_trainer
from bat_training.optim import (
    GradientAccumulator,
    build_optimizer,
    build_scheduler,
    warmup_cosine_lr,
)
from bat_training.pair_trainer import PairTrainer
from bat_training.permutation_adapter import (
    create_pair_trainer_factory,
    create_embedding_trainer_factory,
)

__all__ = [
    "AMPContext",
    "BestCheckpointTracker",
    "CheckpointManager",
    "EarlyStopping",
    "EmbeddingTrainer",
    "ExponentialMovingAverage",
    "GradientAccumulator",
    "PairTrainer",
    "build_optimizer",
    "build_scheduler",
    "create_embedding_trainer_factory",
    "create_pair_trainer_factory",
    "load_checkpoint",
    "make_trainer",
    "save_checkpoint",
    "set_deterministic_mode",
    "warmup_cosine_lr",
    "worker_init_fn",
]
