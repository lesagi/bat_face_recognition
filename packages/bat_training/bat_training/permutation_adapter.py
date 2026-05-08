"""Adapter from :class:`PairTrainer`/:class:`EmbeddingTrainer` to ``bat_stats``.

``bat_stats.runner.run_retrain_test(trainer_factory=...)`` accepts a
callable that returns an object with::

    train_and_evaluate() -> dict[str, float]
    reset_for_new_permutation() -> None

This module exposes :func:`create_pair_trainer_factory` and
:func:`create_embedding_trainer_factory` -- callers pass the result as the
``trainer_factory=`` argument. We do **not** modify ``bat_stats``; this is
the dependency-injection seam Worker F's TODO points at.

Usage from a CLI (Phase 3 sketch)::

    from bat_stats.runner import run_retrain_test
    from bat_training.permutation_adapter import create_pair_trainer_factory

    factory = create_pair_trainer_factory(
        model_factory=lambda: SiameseModel(),
        loss_factory=lambda: BCELoss(),
        train_loader_factory=lambda permute: build_train_loader(permute),
        val_loader_factory=lambda: build_val_loader(),
        cfg=trainer_cfg,
    )
    run_retrain_test(cfg=hydra_cfg, observed_metrics=..., trainer_factory=factory)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from bat_training._common import TrainerConfig
from bat_training.embedding_trainer import EmbeddingTrainer
from bat_training.pair_trainer import PairTrainer

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from bat_core import Manifest
    from torch import nn

ModelFactory = Callable[[], "nn.Module"]
LossFactory = Callable[[], Any]
LoaderFactory = Callable[[bool], Any]
"""``LoaderFactory(permute_labels) -> torch.utils.data.DataLoader``."""

ValLoaderFactory = Callable[[], Any]


class _PermutationAdapter:
    """Minimal :class:`bat_stats.trainer_hooks.TrainerProtocol` impl.

    Holds the ``model_factory`` / ``loss_factory`` / loader factories and
    rebuilds everything between permutations so optimizer state never
    leaks. The only difference per permutation is that the train loader is
    constructed with ``permute=True`` so the underlying labels get
    shuffled (whatever your loader implementation chooses to do).
    """

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        loss_factory: LossFactory,
        train_loader_factory: LoaderFactory,
        val_loader_factory: ValLoaderFactory | None = None,
        cfg: TrainerConfig,
        family: str = "pair",
        permute_labels: bool = True,
        verbose: bool = False,
    ) -> None:
        self._model_factory = model_factory
        self._loss_factory = loss_factory
        self._train_loader_factory = train_loader_factory
        self._val_loader_factory = val_loader_factory
        self._cfg = cfg
        self._family = family
        self._permute_labels = bool(permute_labels)
        self._verbose = bool(verbose)
        self._build()

    def _build(self) -> None:
        model = self._model_factory()
        loss = self._loss_factory()
        if self._family == "pair":
            self._trainer: PairTrainer | EmbeddingTrainer = PairTrainer(
                model=model,
                loss=loss,
                cfg=self._cfg,
            )
        elif self._family == "embedding":
            self._trainer = EmbeddingTrainer(
                model=model,
                loss=loss,
                cfg=self._cfg,
            )
        else:  # pragma: no cover - factory guards this
            raise ValueError(f"unknown family {self._family!r}")
        self._train_loader = self._train_loader_factory(self._permute_labels)
        self._val_loader = (
            self._val_loader_factory() if self._val_loader_factory is not None else None
        )

    # -- TrainerProtocol -------------------------------------------------
    def train_and_evaluate(self) -> dict[str, float]:
        artifacts = self._trainer.fit(self._train_loader, self._val_loader)
        # The protocol asks for at least f1, accuracy, precision, recall.
        # Pull the best-tracked values; PairTrainer exposes f1/recall/
        # precision via its tracker; EmbeddingTrainer exposes accuracy.
        out: dict[str, float] = {}
        for k, v in artifacts.best_metrics.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
        # Provide an "accuracy" alias for the pair-family case (the
        # permutation runner expects it).
        if "accuracy" not in out:
            if "f1" in out:
                out["accuracy"] = out["f1"]
        # Provide an "f1" alias for the embedding-family case if we only
        # tracked accuracy.
        if "f1" not in out and "accuracy" in out:
            out["f1"] = out["accuracy"]
        # Defensive defaults so the runner can compute a null dist.
        for k in ("f1", "accuracy", "precision", "recall"):
            out.setdefault(k, 0.0)
        return out

    def reset_for_new_permutation(self) -> None:
        # Rebuild the trainer + loaders so we start each permutation
        # from a fresh init.
        self._build()


def create_pair_trainer_factory(
    *,
    model_factory: ModelFactory,
    loss_factory: LossFactory,
    train_loader_factory: LoaderFactory,
    val_loader_factory: ValLoaderFactory | None = None,
    cfg: TrainerConfig | None = None,
) -> Callable[..., _PermutationAdapter]:
    """Return a ``trainer_factory`` callable for ``run_retrain_test``.

    The returned callable accepts ``permute_labels``, ``num_epochs``,
    ``verbose`` (and any extra ``**kwargs``) so it matches the signature
    expected by :func:`bat_stats.runner.run_retrain_test`.
    """

    def _factory(
        *,
        permute_labels: bool = True,
        num_epochs: int | None = None,
        verbose: bool = False,
        **_: Any,
    ) -> _PermutationAdapter:
        run_cfg = cfg if cfg is not None else TrainerConfig()
        if num_epochs is not None:
            run_cfg = _replace_epochs(run_cfg, int(num_epochs))
        return _PermutationAdapter(
            model_factory=model_factory,
            loss_factory=loss_factory,
            train_loader_factory=train_loader_factory,
            val_loader_factory=val_loader_factory,
            cfg=run_cfg,
            family="pair",
            permute_labels=permute_labels,
            verbose=verbose,
        )

    return _factory


def create_embedding_trainer_factory(
    *,
    model_factory: ModelFactory,
    loss_factory: LossFactory,
    train_loader_factory: LoaderFactory,
    val_loader_factory: ValLoaderFactory | None = None,
    cfg: TrainerConfig | None = None,
) -> Callable[..., _PermutationAdapter]:
    """Same as :func:`create_pair_trainer_factory` but for embedding losses."""

    def _factory(
        *,
        permute_labels: bool = True,
        num_epochs: int | None = None,
        verbose: bool = False,
        **_: Any,
    ) -> _PermutationAdapter:
        run_cfg = cfg if cfg is not None else TrainerConfig()
        if num_epochs is not None:
            run_cfg = _replace_epochs(run_cfg, int(num_epochs))
        return _PermutationAdapter(
            model_factory=model_factory,
            loss_factory=loss_factory,
            train_loader_factory=train_loader_factory,
            val_loader_factory=val_loader_factory,
            cfg=run_cfg,
            family="embedding",
            permute_labels=permute_labels,
            verbose=verbose,
        )

    return _factory


def _replace_epochs(cfg: TrainerConfig, epochs: int) -> TrainerConfig:
    # dataclasses.replace would be nicer, but TrainerConfig is mutable so
    # a shallow copy is enough.
    new = TrainerConfig(**{**cfg.__dict__})
    new.epochs = epochs
    return new


__all__ = [
    "_PermutationAdapter",
    "create_embedding_trainer_factory",
    "create_pair_trainer_factory",
]
