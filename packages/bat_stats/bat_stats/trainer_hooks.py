"""Trainer hooks for the retrain-style permutation test.

The legacy ``app/statistical_tests/permutation_trainer.py`` is a TF-specific
PermutationTrainer that:
- holds a ``SiameseNetwork`` keras model alive across iterations
- re-initializes weights and optimizer slots between permutations
- builds a TF data pipeline with optionally permuted labels
- runs a custom training loop with ``tf.GradientTape`` etc.

That entire module is **intentionally not ported here**: PyTorch trainers live
in :mod:`bat_training`, and ``bat_stats`` sits strictly below ``bat_training``
in the workspace dependency graph (importing the latter would flip the edge).
We therefore only define the *shape* of what the permutation runner needs
from a trainer (the :class:`TrainerProtocol`) and a stub factory so the
runner imports cleanly and tests can pass mocks.

The real ``PermutationTrainerAdapter`` is provided by
:mod:`bat_training.permutation_adapter` (``create_pair_trainer_factory`` /
``create_embedding_trainer_factory``); callers inject it via the
``trainer_factory=`` argument on :func:`bat_stats.runner.run_retrain_test`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TrainerProtocol(Protocol):
    """Minimal interface a permutation-test trainer must expose.

    Note: this is **local to bat_stats** -- it is intentionally narrower than
    :class:`bat_core.Trainer` because permutation tests don't need the full
    ``fit(...) -> RunArtifacts`` / ``test(...) -> EvalReport`` lifecycle, just
    a "train-with-permuted-labels-and-return-metrics" cycle.

    Do **not** add this protocol to ``bat_core``: per the Phase-1 plan, only
    cross-cutting interfaces live there.
    """

    def train_and_evaluate(self) -> dict[str, float]:
        """Train for the configured number of epochs and return final metrics.

        Returned dict must include at least ``f1``, ``accuracy``, ``precision``,
        ``recall`` (and may include ``loss``, ``best_f1``, etc.).
        """

    def reset_for_new_permutation(self) -> None:
        """Re-randomize labels and reinitialize model + optimizer state.

        Called between iterations so a single trainer instance can serve all
        N permutations without leaking optimizer state.
        """


def create_permutation_trainer(
    *,
    permute_labels: bool = True,
    num_epochs: int = 10,
    verbose: bool = False,
    **_kwargs: Any,
) -> TrainerProtocol:
    """Stub factory.  Always raises ``NotImplementedError``.

    ``bat_stats`` cannot construct a real trainer itself without importing
    ``bat_training`` (which would invert the workspace dep graph).  Callers
    must either:
    - use the inference-based path
      (:func:`bat_stats.run_inference_permutation_test`), which doesn't
      retrain, or
    - inject a ``trainer_factory`` from
      :mod:`bat_training.permutation_adapter` (or any callable returning a
      :class:`TrainerProtocol`).
    """

    del permute_labels, num_epochs, verbose  # silence linters

    raise NotImplementedError(
        "Retrain-style permutation training has no default factory in "
        "bat_stats. Either call bat_stats.run_inference_permutation_test, "
        "or pass a trainer_factory built via "
        "bat_training.permutation_adapter.create_pair_trainer_factory / "
        "create_embedding_trainer_factory."
    )


__all__ = ["TrainerProtocol", "create_permutation_trainer"]
