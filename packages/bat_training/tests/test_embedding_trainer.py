"""End-to-end smoke for EmbeddingTrainer.fit on a tiny synthetic fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")

from bat_losses import ArcFaceLoss
from bat_training import EmbeddingTrainer
from bat_training._common import TrainerConfig


class _ToyArcFaceModel(torch.nn.Module):
    """Backbone -> projection -> ArcFace-like head producing margined logits."""

    family = "embedding"

    def __init__(self, embedding_dim: int = 4, num_classes: int = 3) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(8, embedding_dim, bias=False)
        # Cosine-style head weights.
        self.head_weight = torch.nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.scale = 8.0

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        e = self.fc(x)
        return torch.nn.functional.normalize(e, p=2, dim=1)

    def forward_train(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        e = self.forward_embedding(x)
        w = torch.nn.functional.normalize(self.head_weight, p=2, dim=1)
        # Plain cosine logits (skipping the actual margin to keep the
        # synthetic test deterministic).
        return torch.matmul(e, w.t()) * self.scale

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if labels is None:
            return self.forward_embedding(x)
        return self.forward_train(x, labels)

    def export_for_inference(self) -> torch.nn.Module:
        return self.fc


def _make_id_loader() -> Any:
    """6 images, 3 identities (2 each), 8-dim feature inputs.

    Identity i lives in feature dim ``i`` (one-hot-ish), giving the model
    something trivial to learn: argmax over ``W e^T`` is exactly the
    identity once W aligns with the data axes.
    """
    torch.manual_seed(0)
    x = torch.zeros(6, 8)
    x[0, 0] = x[1, 0] = 1.0
    x[2, 1] = x[3, 1] = 1.0
    x[4, 2] = x[5, 2] = 1.0
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    return [(x, labels)]


def test_embedding_trainer_fit_reduces_loss(tmp_path: Path) -> None:
    """Two epochs on the 6-image / 3-identity fixture should reduce loss."""
    torch.manual_seed(0)
    model = _ToyArcFaceModel(embedding_dim=4, num_classes=3)
    loss = ArcFaceLoss()
    cfg = TrainerConfig(
        epochs=2,
        lr=1e-1,
        ema_decay=0.0,  # disable EMA in this synthetic test
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
    trainer = EmbeddingTrainer(model=model, loss=loss, cfg=cfg, tracker=rec)
    loader = _make_id_loader()

    artifacts = trainer.fit(loader, val_loader=None)

    train_calls = [c for c in rec.calls if c[0] == "train"]
    assert len(train_calls) == 2
    losses = [c[1]["loss"] for c in train_calls]
    assert losses[1] <= losses[0] + 1e-6

    assert "loss" in artifacts.best_metrics


def test_embedding_trainer_embed_fn_uses_configured_image_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_build_embed_fn`` must load images at the trained ``image_size``.

    Regression for the eval/train resolution mismatch: embeddings used for
    val/test/permutation were loaded at the loader's 224 default rather than
    the model's trained edge length (112 for ArcFace/AdaFace).
    """
    import bat_data

    captured_sizes: list[int] = []

    def _fake_loader(
        path: Path, image_size: int = 224, normalize: str | None = None
    ) -> torch.Tensor:
        captured_sizes.append(int(image_size))
        return torch.zeros(8)

    monkeypatch.setattr(bat_data, "default_image_loader", _fake_loader)

    model = _ToyArcFaceModel(embedding_dim=4, num_classes=3)
    cfg = TrainerConfig(epochs=1, ema_decay=0.0, output_dir=tmp_path / "run")
    trainer = EmbeddingTrainer(model=model, loss=ArcFaceLoss(), cfg=cfg, image_size=112)

    embed_fn = trainer._build_embed_fn()
    with torch.no_grad():
        embed_fn([Path("a.jpg"), Path("b.jpg")])

    assert captured_sizes == [112, 112]


def test_embedding_trainer_steps_partial_gradient_accumulation_tail(tmp_path: Path) -> None:
    """A short final accumulation window must still update parameters."""
    torch.manual_seed(0)
    model = _ToyArcFaceModel(embedding_dim=4, num_classes=3)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    cfg = TrainerConfig(
        epochs=1,
        lr=1e-1,
        ema_decay=0.0,
        gradient_accumulation_steps=2,
        output_dir=tmp_path / "run",
    )

    trainer = EmbeddingTrainer(model=model, loss=ArcFaceLoss(), cfg=cfg)
    trainer.fit(_make_id_loader(), val_loader=None)

    after = {name: param.detach().cpu() for name, param in model.named_parameters()}
    assert any(not torch.allclose(before[name], after[name]) for name in before)


def test_embedding_trainer_emits_val_roc_auc_when_eval_manifest_present(tmp_path: Path) -> None:
    """val_verification=True + eval_manifest → val/roc_auc + TAR@FAR merged in."""
    torch.manual_seed(0)
    model = _ToyArcFaceModel(embedding_dim=4, num_classes=3)
    cfg = TrainerConfig(
        epochs=1,
        lr=1e-1,
        ema_decay=0.0,
        output_dir=tmp_path / "run",
        val_verification=True,
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
    # eval_manifest just needs to be truthy; _validate_verification is patched below.
    trainer = EmbeddingTrainer(
        model=model, loss=ArcFaceLoss(), cfg=cfg, tracker=rec, eval_manifest=object()
    )
    trainer._validate_verification = lambda *, split: {  # type: ignore[method-assign]
        "roc_auc": 0.873,
        "tar_at_far_1e3": 0.5,
        "tar_at_far_1e4": 0.3,
        "optimal_threshold": 0.42,
        "youden_j": 0.61,
    }

    loader = _make_id_loader()
    trainer.fit(loader, val_loader=loader)

    val_calls = [c for c in rec.calls if c[0] == "val"]
    assert len(val_calls) == 1
    metrics = val_calls[0][1]
    assert metrics["roc_auc"] == pytest.approx(0.873)
    assert metrics["tar_at_far_1e3"] == pytest.approx(0.5)
    assert metrics["tar_at_far_1e4"] == pytest.approx(0.3)
    # Existing classification metrics must still be there.
    assert "loss" in metrics
    assert "accuracy" in metrics


def test_embedding_trainer_skips_val_verification_when_flag_disabled(tmp_path: Path) -> None:
    """val_verification=False → no roc_auc even if eval_manifest is set."""
    torch.manual_seed(0)
    model = _ToyArcFaceModel(embedding_dim=4, num_classes=3)
    cfg = TrainerConfig(
        epochs=1,
        lr=1e-1,
        ema_decay=0.0,
        output_dir=tmp_path / "run",
        val_verification=False,
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
    trainer = EmbeddingTrainer(
        model=model, loss=ArcFaceLoss(), cfg=cfg, tracker=rec, eval_manifest=object()
    )
    called = []
    trainer._validate_verification = lambda *, split: called.append(split) or {}  # type: ignore[method-assign]

    loader = _make_id_loader()
    trainer.fit(loader, val_loader=loader)

    val_calls = [c for c in rec.calls if c[0] == "val"]
    assert len(val_calls) == 1
    assert "roc_auc" not in val_calls[0][1]
    assert called == []  # helper never invoked


def test_embedding_trainer_skips_val_verification_when_eval_manifest_missing(
    tmp_path: Path,
) -> None:
    """val_verification=True but no eval_manifest → no roc_auc, no crash."""
    torch.manual_seed(0)
    model = _ToyArcFaceModel(embedding_dim=4, num_classes=3)
    cfg = TrainerConfig(
        epochs=1,
        lr=1e-1,
        ema_decay=0.0,
        output_dir=tmp_path / "run",
        val_verification=True,
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
    trainer = EmbeddingTrainer(model=model, loss=ArcFaceLoss(), cfg=cfg, tracker=rec)
    # No eval_manifest set; helper should not be called.

    loader = _make_id_loader()
    trainer.fit(loader, val_loader=loader)

    val_calls = [c for c in rec.calls if c[0] == "val"]
    assert len(val_calls) == 1
    assert "roc_auc" not in val_calls[0][1]


def test_embedding_trainer_fires_callbacks_on_validation_end(tmp_path: Path) -> None:
    """callbacks=[cb] -> cb.on_validation_end(epoch, metrics) once per val epoch."""
    torch.manual_seed(0)
    model = _ToyArcFaceModel(embedding_dim=4, num_classes=3)
    cfg = TrainerConfig(
        epochs=2,
        lr=1e-1,
        ema_decay=0.0,
        output_dir=tmp_path / "run",
    )

    class _RecordingCb:
        def __init__(self) -> None:
            self.calls: list[tuple[int, dict[str, float]]] = []

        def on_validation_end(self, epoch: int, metrics: dict[str, float]) -> None:
            self.calls.append((epoch, dict(metrics)))

    cb = _RecordingCb()
    trainer = EmbeddingTrainer(model=model, loss=ArcFaceLoss(), cfg=cfg, callbacks=[cb])
    loader = _make_id_loader()
    trainer.fit(loader, val_loader=loader)

    assert [c[0] for c in cb.calls] == [1, 2]
    # Metrics dict should at minimum carry the classification keys.
    assert "loss" in cb.calls[0][1]
    assert "accuracy" in cb.calls[0][1]


def test_embedding_trainer_callback_can_abort_via_exception(tmp_path: Path) -> None:
    """Callback exceptions propagate out of fit() (trainer does not catch)."""
    torch.manual_seed(0)
    model = _ToyArcFaceModel(embedding_dim=4, num_classes=3)
    cfg = TrainerConfig(
        epochs=5,  # epochs we will never reach
        lr=1e-1,
        ema_decay=0.0,
        output_dir=tmp_path / "run",
    )

    class _Aborter:
        def on_validation_end(self, epoch: int, metrics: dict[str, float]) -> None:
            raise RuntimeError(f"abort at epoch {epoch}")

    trainer = EmbeddingTrainer(model=model, loss=ArcFaceLoss(), cfg=cfg, callbacks=[_Aborter()])
    loader = _make_id_loader()
    with pytest.raises(RuntimeError, match="abort at epoch 1"):
        trainer.fit(loader, val_loader=loader)


def test_embedding_trainer_skips_callbacks_without_val_loader(tmp_path: Path) -> None:
    """No val loader -> callbacks never invoked."""
    torch.manual_seed(0)
    model = _ToyArcFaceModel(embedding_dim=4, num_classes=3)
    cfg = TrainerConfig(
        epochs=1,
        lr=1e-1,
        ema_decay=0.0,
        output_dir=tmp_path / "run",
    )

    class _Bomb:
        def on_validation_end(self, epoch: int, metrics: dict[str, float]) -> None:
            raise AssertionError("should not be called without a val loader")

    trainer = EmbeddingTrainer(model=model, loss=ArcFaceLoss(), cfg=cfg, callbacks=[_Bomb()])
    trainer.fit(_make_id_loader(), val_loader=None)  # must NOT raise


def test_embedding_trainer_rejects_pair_loss() -> None:
    """`EmbeddingTrainer` must refuse a loss with family != 'embedding'."""
    pytest.importorskip("torch")

    class _FakePairLoss:
        family = "pair"

        def __call__(
            self, model_output: Any, labels: Any, sample_weights: Any = None
        ) -> Any:  # pragma: no cover -- not reached
            raise AssertionError("not used")

    from bat_core.exceptions import InterfaceViolationError

    with pytest.raises(InterfaceViolationError):
        EmbeddingTrainer(
            model=_ToyArcFaceModel(),
            loss=_FakePairLoss(),
            cfg=TrainerConfig(),
        )
