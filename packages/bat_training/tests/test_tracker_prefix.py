"""Verify the trainers feed `train/`/`val/` sectioned metrics into the tracker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

from bat_losses import BCELoss
from bat_training import PairTrainer
from bat_training._common import TrainerConfig


class _MockTracker:
    """Captures every (section, metrics, step) the trainer emits."""

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


def test_pair_trainer_feeds_train_and_val_sections(
    tmp_path: Path, tiny_pair_model_cls: Any, make_pair_loader: Any
) -> None:
    """The trainer logs section='train' for train metrics and 'val' for val metrics."""
    torch.manual_seed(1)
    cfg = TrainerConfig(epochs=2, lr=1e-2, output_dir=tmp_path / "run")
    tracker = _MockTracker()
    trainer = PairTrainer(model=tiny_pair_model_cls(), loss=BCELoss(), cfg=cfg, tracker=tracker)

    train_loader = make_pair_loader()
    val_loader = make_pair_loader()
    trainer.fit(train_loader, val_loader=val_loader)

    sections = [c[0] for c in tracker.calls]
    assert sections.count("train") == 2
    assert sections.count("val") == 2

    train_metrics = next(c[1] for c in tracker.calls if c[0] == "train")
    assert "loss" in train_metrics
    val_metrics = next(c[1] for c in tracker.calls if c[0] == "val")
    assert "loss" in val_metrics
    assert "roc_auc" in val_metrics
