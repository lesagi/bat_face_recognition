"""End-to-end smoke for PairTrainer.fit on a tiny synthetic fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

from bat_losses import BCELoss
from bat_training import PairTrainer
from bat_training._common import TrainerConfig


def test_pair_trainer_fit_reduces_loss(
    tmp_path: Path, tiny_pair_model_cls: Any, make_pair_loader: Any
) -> None:
    """Two epochs on the 4-pair fixture should reduce the BCE loss."""
    torch.manual_seed(0)
    model = tiny_pair_model_cls()
    loss = BCELoss()
    cfg = TrainerConfig(
        epochs=2,
        lr=1e-2,
        weight_decay=1e-4,
        output_dir=tmp_path / "run",
        artifact_retention={"save_best_f1": True},
    )

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, float], int]] = []

        def log_metrics(self, section: str, metrics: dict[str, float], step: int) -> None:
            self.calls.append((section, dict(metrics), step))

        def log_artifact(self, *args: Any, **kwargs: Any) -> None:
            pass

        def log_config(self, cfg: Any) -> None:
            pass

        def promote_to_champion(self, run_id: str, criterion: str) -> bool:
            return False

    rec = _Recorder()
    trainer = PairTrainer(model=model, loss=loss, cfg=cfg, tracker=rec)
    loader = make_pair_loader()

    artifacts = trainer.fit(loader, val_loader=None)

    train_calls = [c for c in rec.calls if c[0] == "train"]
    assert len(train_calls) == 2
    losses = [c[1]["loss"] for c in train_calls]
    assert losses[1] <= losses[0] + 1e-6

    assert "loss" in artifacts.best_metrics


def test_pair_trainer_rejects_wrong_family(tiny_pair_model_cls: Any) -> None:
    """`PairTrainer` must refuse a loss with family != 'pair'."""

    class _FakeEmbeddingLoss:
        family = "embedding"

        def __call__(
            self, model_output: Any, labels: Any, sample_weights: Any = None
        ) -> Any:  # pragma: no cover -- not reached
            raise AssertionError("should not be called")

    from bat_core.exceptions import InterfaceViolationError

    with pytest.raises(InterfaceViolationError):
        PairTrainer(
            model=tiny_pair_model_cls(),
            loss=_FakeEmbeddingLoss(),
            cfg=TrainerConfig(),
        )
