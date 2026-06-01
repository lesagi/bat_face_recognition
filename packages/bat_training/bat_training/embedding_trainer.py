"""EmbeddingTrainer: training loop for ``family="embedding"`` losses.

Use this trainer for ArcFace / AdaFace / CosFace / SubCenter ArcFace --
the head produces margined logits, the loss is plain cross-entropy over
those logits. Inputs are ``(image, identity_label)``.

The trainer keeps **only** the loop, optimizer, scheduler, AMP, EMA, and
checkpointing. Identification + verification metrics are produced by
:mod:`bat_evaluation.protocols.run_eval_protocol` against a manifest +
``embed_fn`` closure.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bat_core import EvalReport, Manifest, RunArtifacts
from bat_core.exceptions import InterfaceViolationError
from bat_evaluation.protocols import run_eval_protocol
from bat_training._common import TrainerConfig, detach_to_float, unpack_embedding_batch
from bat_training.amp import AMPContext
from bat_training.callbacks import EarlyStopping, set_deterministic_mode
from bat_training.checkpointing import (
    BestCheckpointTracker,
    CheckpointManager,
    restore_into,
    save_checkpoint,
)
from bat_training.ema import ExponentialMovingAverage
from bat_training.optim import GradientAccumulator, build_optimizer, build_scheduler

if TYPE_CHECKING:  # pragma: no cover -- typing only
    import torch
    from accelerate import Accelerator
    from bat_core.interfaces import Tracker
    from torch import nn


class EmbeddingTrainer:
    """Trainer for embedding-family models (ArcFace / AdaFace / CosFace / SubCenter).

    Wiring contract -- the trainer drives the head with labels::

        emb = model.forward_embedding(x)
        logits = model.head(emb, labels)   # margined when training
        loss_value = loss(logits, labels)

    For models exposing :meth:`forward_train` (which is the convention in
    :mod:`bat_models`), we call ``model.forward_train(x, labels)`` directly
    -- it does the equivalent: backbone -> projection -> head.

    Args:
        model: A :class:`bat_core.FaceModel` with ``family == "embedding"``.
        loss: A :class:`bat_core.Loss` with ``family == "embedding"``.
        cfg: A :class:`TrainerConfig`.
        eval_manifest: Optional :class:`bat_core.Manifest` used by
            :meth:`test`. If ``None`` you must provide one explicitly to
            :meth:`test`.
        eval_split: Which split (``"val"``/``"test"``) :meth:`test` uses
            on the manifest. Defaults to ``"test"``.
        tracker / accelerator / run_id / register_name / promote_criterion:
            same semantics as :class:`PairTrainer`.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Any,
        cfg: TrainerConfig,
        *,
        eval_manifest: Manifest | None = None,
        eval_split: str = "test",
        tracker: Tracker | None = None,
        accelerator: Accelerator | None = None,
        run_id: str | None = None,
        register_name: str | None = None,
        promote_criterion: str = "test/roc_auc",
        callbacks: list[Any] | None = None,
    ) -> None:
        family = getattr(loss, "family", None)
        if family != "embedding":
            raise InterfaceViolationError(
                f"EmbeddingTrainer requires loss.family=='embedding'; got {family!r}"
            )
        m_family = getattr(model, "family", None)
        if m_family is not None and m_family != "embedding":
            raise InterfaceViolationError(
                f"EmbeddingTrainer requires model.family=='embedding'; got {m_family!r}"
            )

        if cfg.deterministic:
            set_deterministic_mode(cfg.seed)

        self.cfg = cfg
        self.model = model
        self.loss = loss
        self.tracker = tracker
        self._run_id = run_id
        self._register_name = register_name
        self._promote_criterion = promote_criterion
        self.eval_manifest = eval_manifest
        self.eval_split = eval_split
        self.callbacks: list[Any] = list(callbacks or [])

        # ArcFace/AdaFace baselines benefit from non-zero weight_decay
        # (the canonical recipe is 5e-4 with SGD); we honor whatever the
        # cfg or the model's recommendation says.
        weight_decay = cfg.weight_decay
        recommended = getattr(model, "recommended_weight_decay", None)
        if recommended is not None and weight_decay == 0.0:
            weight_decay = float(recommended)

        self.optimizer = build_optimizer(
            (p for p in model.parameters() if p.requires_grad),
            name=cfg.optimizer,
            lr=cfg.lr,
            weight_decay=weight_decay,
            momentum=cfg.momentum,
        )
        self.scheduler = None  # built lazily in fit()

        self.accelerator = accelerator
        self._owns_accelerator = accelerator is None

        self.amp = AMPContext(precision=cfg.mixed_precision, enabled=cfg.mixed_precision != "no")
        ema_decay = cfg.ema_decay if cfg.ema_decay is not None else 0.999
        self.ema = ExponentialMovingAverage(model, decay=ema_decay) if ema_decay > 0 else None
        self.grad_accum = GradientAccumulator(steps=max(1, cfg.gradient_accumulation_steps))

        self.best = BestCheckpointTracker(
            metrics={
                "f1": "max",
                "recall": "max",
                "precision": "max",
                "loss": "min",
                "roc_auc": "max",
                "top1": "max",
                "recall_at_far_1e2": "max",
            }
        )
        self.checkpoints = CheckpointManager(
            output_dir=cfg.output_dir,
            retention=cfg.artifact_retention,
        )
        if cfg.early_stopping_patience > 0:
            self.early_stopping: EarlyStopping | None = EarlyStopping(
                monitor=cfg.early_stopping_monitor,
                mode=cfg.early_stopping_mode,  # type: ignore[arg-type]
                patience=cfg.early_stopping_patience,
                min_delta=cfg.early_stopping_min_delta,
            )
        else:
            self.early_stopping = None

        self._global_step: int = 0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _ensure_accelerator(self) -> Accelerator:
        if self.accelerator is None:
            from accelerate import Accelerator

            self.accelerator = Accelerator(
                mixed_precision=self.cfg.mixed_precision,
                gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            )
        return self.accelerator

    def _backward(self, loss: torch.Tensor) -> None:
        if self.accelerator is not None:
            self.accelerator.backward(loss)
            return
        self.amp.backward(loss)

    def _autocast(self) -> Any:
        if self.accelerator is not None:
            return self.accelerator.autocast()
        return self.amp.autocast()

    def _step_optimizer(self) -> None:
        if self.accelerator is not None:
            self.optimizer.step()
        else:
            self.amp.step(self.optimizer)
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema is not None:
            self.ema.update(self.model)

    def _has_pending_accumulation(self) -> bool:
        return self.grad_accum.counter > 0 and self.grad_accum.counter % self.grad_accum.steps != 0

    def _log(self, section: str, metrics: dict[str, float], step: int) -> None:
        if self.tracker is None:
            return
        self.tracker.log_metrics(section=section, metrics=metrics, step=step)  # type: ignore[arg-type]

    def _forward_train(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Run ``model.forward_train(x, labels)`` (or fall back).

        :mod:`bat_models` ArcFaceModel/AdaFaceModel implement
        ``forward_train`` directly; for plain ``nn.Module``\\ s with no such
        method we fall back to ``model.forward_embedding(x)`` followed by
        ``model.head(emb, labels)``.
        """
        if hasattr(self.model, "forward_train"):
            return self.model.forward_train(x, labels)  # type: ignore[no-any-return]
        emb = self.model.forward_embedding(x)  # type: ignore[attr-defined]
        return self.model.head(emb, labels)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # bat_core.Trainer protocol                                          #
    # ------------------------------------------------------------------ #
    def fit(self, train_loader: Any, val_loader: Any | None = None) -> RunArtifacts:
        accelerator = self._ensure_accelerator()
        try:
            steps_per_epoch = len(train_loader)
        except TypeError:
            steps_per_epoch = 1
        optimizer_steps_per_epoch = max(
            1,
            (steps_per_epoch + self.grad_accum.steps - 1) // self.grad_accum.steps,
        )
        total_steps = max(1, optimizer_steps_per_epoch * self.cfg.epochs)
        self.scheduler = build_scheduler(
            self.optimizer,
            warmup_steps=self.cfg.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=self.cfg.min_lr_ratio,
        )

        if val_loader is not None:
            (
                self.model,
                self.optimizer,
                self.scheduler,
                train_loader,
                val_loader,
            ) = accelerator.prepare(
                self.model, self.optimizer, self.scheduler, train_loader, val_loader
            )
        else:
            self.model, self.optimizer, self.scheduler, train_loader = accelerator.prepare(
                self.model, self.optimizer, self.scheduler, train_loader
            )

        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._train_one_epoch(train_loader, epoch=epoch)
            self._log("train", train_metrics, step=epoch)

            val_metrics: dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self._validate(val_loader)
                if self.cfg.val_verification and self.eval_manifest is not None:
                    verif_metrics = self._validate_verification(split="val")
                    if verif_metrics is not None:
                        val_metrics = {**val_metrics, **verif_metrics}
                self._log("val", val_metrics, step=epoch)
                for cb in self.callbacks:
                    cb.on_validation_end(epoch=epoch, metrics=val_metrics)
                self._maybe_save_best(epoch, val_metrics)
            else:
                self._maybe_save_best(epoch, train_metrics)

            if self.early_stopping is not None:
                monitor_metrics = val_metrics or train_metrics
                value = monitor_metrics.get(self.cfg.early_stopping_monitor)
                if value is not None and self.early_stopping.step(value, epoch=epoch):
                    break

        final_state = self._build_state_dict(epoch=epoch)
        final_path = self.checkpoints.write_final(final_state)

        return RunArtifacts(
            run_id=self._run_id or "",
            output_dir=Path(self.cfg.output_dir),
            best_metrics={k: v["value"] for k, v in self.best.summary().items()},  # type: ignore[misc]
            final_metrics={},
            checkpoints={"final": final_path} if final_path is not None else {},
        )

    def test(
        self,
        test_loader: Any | None = None,
        *,
        manifest: Manifest | None = None,
        split: str | None = None,
    ) -> EvalReport:
        """Evaluate on a test loader or a manifest + ``embed_fn`` closure.

        Identification metrics need a gallery / probe split (>= 2 images
        per identity). The protocol module accepts a manifest split and
        an embedding callable. The trainer wraps :meth:`forward_embedding`
        as that callable.

        Args:
            test_loader: Optional unused convenience -- present so the
                signature matches :class:`bat_core.Trainer.test`. If the
                caller passes a loader we still run identification via the
                manifest + closure path.
            manifest: Manifest to use; defaults to ``self.eval_manifest``.
            split: Manifest split; defaults to ``self.eval_split``.
        """
        del test_loader  # reserved -- the protocol uses paths, not loaders
        m = manifest if manifest is not None else self.eval_manifest
        if m is None:
            raise ValueError(
                "EmbeddingTrainer.test() needs a manifest; pass manifest=... "
                "or set eval_manifest in the constructor."
            )
        split_name = split if split is not None else self.eval_split

        embed_fn = self._build_embed_fn()
        if self.ema is not None:
            self.ema.apply_to(self.model)
        try:
            report = run_eval_protocol(m, split=split_name, embed_fn=embed_fn)  # type: ignore[arg-type]
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)

        if self.tracker is not None:
            self._log(
                "test",
                {
                    # Performance metrics (all on [0, 1] scale).
                    "roc_auc": float(report.verification.roc_auc),
                    "youden_j": float(report.verification.youden_j),
                    "tar_at_far_1e3": float(report.verification.tar_at_far_1e3),
                    "tar_at_far_1e4": float(report.verification.tar_at_far_1e4),
                    "recall_at_far_1e2": float(report.verification.recall_at_far_1e2),
                    "top1": float(report.identification.top1) if report.identification else 0.0,
                    "top5": float(report.identification.top5) if report.identification else 0.0,
                    "map": float(report.identification.map) if report.identification else 0.0,
                    # Decision thresholds (not on [0, 1] scale — grouped separately
                    # so MLflow keeps them out of the performance-metric charts).
                    "threshold/optimal_threshold": float(report.verification.optimal_threshold),
                    "threshold/threshold_at_far_1e2": float(
                        report.verification.threshold_at_far_1e2
                    ),
                },
                step=0,
            )
        return report

    # ------------------------------------------------------------------ #
    # Loops                                                              #
    # ------------------------------------------------------------------ #
    def _train_one_epoch(self, loader: Iterable[Any], *, epoch: int) -> dict[str, float]:
        import torch

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        n_batches = 0

        self.optimizer.zero_grad(set_to_none=True)
        self.grad_accum.reset()
        device = self._infer_device()

        for batch in loader:
            x, labels = unpack_embedding_batch(batch)
            x = x.to(device)
            labels_long = labels.to(device=device, dtype=torch.long)

            with self._autocast():
                logits = self._forward_train(x, labels_long)
                loss_value = self.loss(logits, labels_long)
            scaled = loss_value / float(self.grad_accum.steps)

            self._backward(scaled)
            self.grad_accum.tick()

            if self.grad_accum.should_step:
                self._step_optimizer()

            total_loss += detach_to_float(loss_value)
            n_batches += 1
            self._global_step += 1

            with torch.no_grad():
                preds = torch.argmax(logits.detach(), dim=1)
                total_correct += int((preds == labels_long).sum().item())
                total_count += int(labels_long.numel())

        if self._has_pending_accumulation():
            self._step_optimizer()

        accuracy = total_correct / max(1, total_count)
        avg_loss = total_loss / max(1, n_batches)
        return {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            # we don't have f1 in identification training; we report top-1
            # accuracy here for callers monitoring "f1" via early stopping.
            "f1": float(accuracy),
            "precision": float(accuracy),
            "recall": float(accuracy),
        }

    def _validate(self, loader: Iterable[Any]) -> dict[str, float]:
        import torch

        if self.ema is not None:
            self.ema.apply_to(self.model)
        self.model.eval()
        try:
            total_loss = 0.0
            total_correct = 0
            total_count = 0
            n_batches = 0
            device = self._infer_device()
            with torch.no_grad():
                for batch in loader:
                    x, labels = unpack_embedding_batch(batch)
                    x = x.to(device)
                    labels_long = labels.to(device=device, dtype=torch.long)
                    with self._autocast():
                        logits = self._forward_train(x, labels_long)
                        loss_value = self.loss(logits, labels_long)
                    total_loss += detach_to_float(loss_value)
                    n_batches += 1
                    preds = torch.argmax(logits.detach(), dim=1)
                    total_correct += int((preds == labels_long).sum().item())
                    total_count += int(labels_long.numel())
            accuracy = total_correct / max(1, total_count)
            return {
                "loss": float(total_loss / max(1, n_batches)),
                "accuracy": float(accuracy),
                "f1": float(accuracy),
                "precision": float(accuracy),
                "recall": float(accuracy),
            }
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)

    def _validate_verification(self, *, split: str) -> dict[str, float] | None:
        """Run bat_evaluation verification on the chosen manifest split.

        Returns ``{"roc_auc", "tar_at_far_1e3", "tar_at_far_1e4",
        "optimal_threshold", "youden_j"}`` on success, or ``None`` if the
        protocol can't be run (e.g., the split has fewer than 2 identities
        with the gallery/probe minimum). Failure is silent because this is
        a best-effort enrichment of the val metrics dict.
        """
        if self.eval_manifest is None:
            return None
        embed_fn = self._build_embed_fn()
        if self.ema is not None:
            self.ema.apply_to(self.model)
        try:
            report = run_eval_protocol(
                self.eval_manifest,  # type: ignore[arg-type]
                split=split,  # type: ignore[arg-type]
                embed_fn=embed_fn,
            )
        except Exception:  # pragma: no cover -- best-effort enrichment
            return None
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
        v = report.verification
        return {
            "roc_auc": float(v.roc_auc),
            "tar_at_far_1e3": float(v.tar_at_far_1e3),
            "tar_at_far_1e4": float(v.tar_at_far_1e4),
            "optimal_threshold": float(v.optimal_threshold),
            "youden_j": float(v.youden_j),
            "recall_at_far_1e2": float(v.recall_at_far_1e2),
            "threshold_at_far_1e2": float(v.threshold_at_far_1e2),
        }

    def _build_embed_fn(self) -> Callable[[list[Path]], Any]:
        """Return a closure that maps a list of image paths to an embedding tensor.

        Uses :func:`bat_data.default_image_loader` for IO so this trainer
        stays loader-agnostic. Callers wanting a different loader should
        wrap :meth:`model.forward_embedding` themselves and call
        :func:`bat_evaluation.protocols.run_eval_protocol` directly.
        """
        from bat_data import default_image_loader

        model = self.model
        device = self._infer_device()

        def _embed(paths: list[Path]) -> Any:
            import torch

            tensors = [default_image_loader(Path(p)) for p in paths]
            stacked = torch.stack(tensors, dim=0).to(device)
            model.eval()
            with torch.no_grad():
                emb = model.forward_embedding(stacked)  # type: ignore[attr-defined]
            return emb.detach().cpu()

        return _embed

    def _infer_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover -- empty model
            import torch

            return torch.device("cpu")

    def _maybe_save_best(self, epoch: int, metrics: dict[str, float]) -> None:
        for key in ("f1", "recall", "precision", "loss", "roc_auc", "top1", "recall_at_far_1e2"):
            if key not in metrics:
                continue
            improved = self.best.update(key, metrics[key], epoch=epoch)
            if improved:
                self.checkpoints.write_best(
                    metric_name=key,
                    epoch=epoch,
                    state=self._build_state_dict(epoch=epoch),
                )

    def _build_state_dict(self, *, epoch: int) -> dict[str, Any]:
        state: dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": self._global_step,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        return state

    # ------------------------------------------------------------------ #
    # Resume                                                             #
    # ------------------------------------------------------------------ #
    def load(self, path: str | Path) -> None:
        from bat_training.checkpointing import load_checkpoint

        state = load_checkpoint(path)
        restore_into(
            state,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
        )
        self._global_step = int(state.get("step", 0))

    def save(self, path: str | Path) -> Path:
        return save_checkpoint(self._build_state_dict(epoch=0), path)


__all__ = ["EmbeddingTrainer"]
