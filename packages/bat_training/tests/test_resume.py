"""End-to-end test: save a checkpoint mid-training and resume into a fresh trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from bat_losses import BCELoss
from bat_training import PairTrainer
from bat_training._common import TrainerConfig
from bat_training.checkpointing import load_checkpoint


def test_resume_state_dict_keys_round_trip(
    tmp_path: Path, tiny_pair_model_cls: Any
) -> None:
    """``load_checkpoint`` returns a dict with the expected keys and they
    line up with the model / optimizer state-dicts."""
    torch.manual_seed(0)
    cfg = TrainerConfig(epochs=1, lr=1e-2, output_dir=tmp_path / "run")
    trainer = PairTrainer(model=tiny_pair_model_cls(), loss=BCELoss(), cfg=cfg)
    path = trainer.save(tmp_path / "ckpt.pt")
    state = load_checkpoint(path)
    assert set(state.keys()) >= {"model", "optimizer", "epoch", "step"}

    trainer2 = PairTrainer(model=tiny_pair_model_cls(), loss=BCELoss(), cfg=cfg)
    trainer2.load(path)

    assert set(trainer.model.state_dict().keys()) == set(trainer2.model.state_dict().keys())

    sd1 = trainer.optimizer.state_dict()
    sd2 = trainer2.optimizer.state_dict()
    assert set(sd1.keys()) == set(sd2.keys())
    assert len(sd1["param_groups"]) == len(sd2["param_groups"])
