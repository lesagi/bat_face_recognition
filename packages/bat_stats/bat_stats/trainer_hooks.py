"""Trainer hooks for the retrain-style permutation test.

The legacy ``app/statistical_tests/permutation_trainer.py`` is a TF-specific
PermutationTrainer that:
- holds a ``SiameseNetwork`` keras model alive across iterations
- re-initializes weights and optimizer slots between permutations
- builds a TF data pipeline with optionally permuted labels
- runs a custom training loop with ``tf.GradientTape`` etc.

That entire module is **intentionally not ported here**: PyTorch trainers live
in :mod:`bat_training`, which is a Phase-2 deliverable.  Until ``bat_training``
exists we only define the *shape* of what the permutation runner needs from
a trainer (the :class:`TrainerProtocol`) and a stub factory so the runner
imports cleanly and tests can pass mocks.

TODO(phase-2): provide a real ``PermutationTrainerAdapter`` that wraps the
PyTorch ``PairTrainer`` from ``bat_training`` and exposes
``train_and_evaluate`` / ``reset_for_new_permutation``.  At that point the
``create_permutation_trainer`` factory below should be replaced with one that
imports from ``bat_training`` lazily.
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
    """Stub factory.  Raises ``NotImplementedError`` until Phase 2.

    The retrain-style permutation test cannot run end-to-end without a real
    PyTorch trainer.  Callers should either:
    - use the inference-based path (:func:`bat_stats.run_inference_permutation_test`),
      which doesn't retrain, or
    - inject their own ``trainer_factory`` that returns something satisfying
      :class:`TrainerProtocol`.

    TODO(phase-2): replace this with a thin wrapper around
    ``bat_training.PairTrainer`` once it lands.
    """

    del permute_labels, num_epochs, verbose  # silence linters

    raise NotImplementedError(
        "Retrain-style permutation training is not yet wired up. "
        "Use bat_stats.run_inference_permutation_test for now, or pass your "
        "own trainer_factory to PermutationTest.run(). "
        "Phase-2 (bat_training) will provide the concrete trainer."
    )


__all__ = ["TrainerProtocol", "create_permutation_trainer"]
