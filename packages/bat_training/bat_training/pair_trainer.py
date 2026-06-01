"""PairTrainer: training loop for ``family="pair"`` losses.

Use this trainer for Siamese-style verification: BCE / Focal / Triplet on
pair inputs ``(x_a, x_b, label)``. Identification metrics are not produced
here -- pair models do not learn an identity classifier; verification ROC /
AUC + Youden-J threshold are the only meaningful outputs.

The trainer keeps **only** the loop, optimizer, scheduler, AMP, EMA, and
checkpointing. Data wiring is the caller's responsibility (use
:class:`bat_data` and :class:`torch.utils.data.DataLoader`); MLflow goes
through ``tracker.log_metrics(section=..., metrics=..., step=...)``;
verification metrics come from :mod:`bat_evaluation.verification`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bat_core import EvalReport, Predictions, RunArtifacts, VerificationMetrics
from bat_core.exceptions import InterfaceViolationError
from bat_evaluation.verification import evaluate_predictions
from bat_training._common import TrainerConfig, detach_to_float, unpack_pair_batch
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


class PairTrainer:
    """Trainer for pair-family models (Siamese with BCE / Focal / Triplet).

    This implements :class:`bat_core.Trainer`.

    Args:
        model: A :class:`bat_core.FaceModel` with ``family == "pair"``.
        loss: A :class:`bat_core.Loss` with ``family == "pair"``.
        cfg: A :class:`TrainerConfig`.
        tracker: Optional :class:`bat_core.Tracker`. Metrics flow through
            ``tracker.log_metrics(section=..., metrics={...}, step=epoch)``;
            ``train/`` and ``val/`` prefixes are added automatically.
        accelerator: Optional pre-built :class:`accelerate.Accelerator`.
            If ``None``, one is constructed with
            ``Accelerator(mixed_precision=cfg.mixed_precision,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps)``.
        run_id: Optional MLflow run id (forwarded to model registration
            during :meth:`fit` finalization).
        register_name: Model-registry name. If set, registration occurs
            **before** ``tracker.promote_to_champion`` -- ordering is
            mandatory per Worker F's TODO.
        promote_criterion: MLflow metric used to gate champion promotion.
            Defaults to ``"test/roc_auc"``.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Any,
        cfg: TrainerConfig,
        *,
        tracker: Tracker | None = None,
        accelerator: Accelerator | None = None,
        run_id: str | None = None,
        register_name: str | None = None,
        promote_criterion: str = "test/roc_auc",
        callbacks: list[Any] | None = None,
    ) -> None:
        # Family contract: PairTrainer accepts only family="pair" losses.
        family = getattr(loss, "family", None)
        if family != "pair":
            raise InterfaceViolationError(
                f"PairTrainer requires loss.family=='pair'; got {family!r}"
            )
        # Model family is checked too (when present) -- not all wrappers
        # expose .family at runtime, so we accept missing gracefully.
        m_family = getattr(model, "family", None)
        if m_family is not None and m_family != "pair":
            raise InterfaceViolationError(
                f"PairTrainer requires model.family=='pair'; got {m_family!r}"
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
        self.callbacks: list[Any] = list(callbacks or [])

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

        # The scheduler total-steps depends on the number of batches; we
        # build it lazily inside :meth:`fit` once we know ``len(loader)``.
        self.scheduler = None  # set in fit()

        self.accelerator = accelerator
        self._owns_accelerator = accelerator is None

        self.amp = AMPContext(precision=cfg.mixed_precision, enabled=cfg.mixed_precision != "no")
        self.ema = (
            ExponentialMovingAverage(model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None
        )
        self.grad_accum = GradientAccumulator(steps=max(1, cfg.gradient_accumulation_steps))

        self.best = BestCheckpointTracker(
            metrics={
                "f1": "max",
                "recall": "max",
                "precision": "max",
                "loss": "min",
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
        # Standalone (test) path
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

    # ------------------------------------------------------------------ #
    # bat_core.Trainer protocol                                          #
    # ------------------------------------------------------------------ #
    def fit(self, train_loader: Any, val_loader: Any | None = None) -> RunArtifacts:
        """Run the training loop and return :class:`RunArtifacts`."""
        accelerator = self._ensure_accelerator()

        # Materialise the LR scheduler now we know ``len(train_loader)``.
        try:
            steps_per_epoch = len(train_loader)
        except TypeError:  # iterable without __len__
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
                val_metrics = self._evaluate(val_loader)
                self._log("val", val_metrics, step=epoch)
                for cb in self.callbacks:
                    cb.on_validation_end(epoch=epoch, metrics=val_metrics)

                # Best-only checkpointing on val metrics (or train metrics
                # when no val loader is provided).
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

    def test(self, test_loader: Any) -> EvalReport:
        """Evaluate on a test loader and return :class:`EvalReport`.

        Verification only -- pair models do not produce identification
        metrics. Use :class:`EmbeddingTrainer.test` for embeddings.
        """
        if self.accelerator is None:
            self._ensure_accelerator()
        predictions = self._collect_predictions(test_loader)
        verification = evaluate_predictions(predictions)
        if self.tracker is not None:
            tp, fp, _tn, fn = _confusion_at_threshold(
                predictions, threshold=verification.optimal_threshold
            )
            recall, precision, f1 = _prf1(tp, fp, fn)
            self._log_verification(
                verification,
                section="test",
                step=0,
                classification={"f1": f1, "precision": precision, "recall": recall},
            )
        return EvalReport(verification=verification, identification=None, predictions=predictions)

    # ------------------------------------------------------------------ #
    # Loops                                                              #
    # ------------------------------------------------------------------ #
    def _train_one_epoch(self, loader: Iterable[Any], *, epoch: int) -> dict[str, float]:
        import torch

        self.model.train()
        total_loss = 0.0
        n_batches = 0
        running_tp = running_fp = running_tn = running_fn = 0

        self.optimizer.zero_grad(set_to_none=True)
        self.grad_accum.reset()
        device = self._infer_device()

        for batch in loader:
            x_a, x_b, label = unpack_pair_batch(batch)
            x_a = x_a.to(device)
            x_b = x_b.to(device)
            label = label.to(device)

            with self._autocast():
                output = (
                    self.model(x_a, x_b) if self._call_model_with_pair() else self.model((x_a, x_b))
                )
                loss = self.loss(output, label)
            scaled = loss / float(self.grad_accum.steps)

            self._backward(scaled)
            self.grad_accum.tick()

            if self.grad_accum.should_step:
                self._step_optimizer()

            total_loss += detach_to_float(loss)
            n_batches += 1
            self._global_step += 1

            # Cheap precision/recall accumulation for pair sigmoid output
            with torch.no_grad():
                tp, fp, tn, fn = _confusion_pieces(output, label)
                running_tp += tp
                running_fp += fp
                running_tn += tn
                running_fn += fn

        if self._has_pending_accumulation():
            self._step_optimizer()

        recall, precision, f1 = _prf1(running_tp, running_fp, running_fn)
        avg_loss = total_loss / max(1, n_batches)
        return {
            "loss": float(avg_loss),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
        }

    def _evaluate(self, loader: Iterable[Any]) -> dict[str, float]:
        """Validation loop -- returns loss + verification metrics."""
        if self.ema is not None:
            self.ema.apply_to(self.model)
        try:
            predictions, total_loss, n_batches = self._predict_with_loss(loader)
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
        verification = evaluate_predictions(predictions)
        running_tp, running_fp, running_tn, running_fn = _confusion_at_threshold(
            predictions, threshold=verification.optimal_threshold
        )
        recall, precision, f1 = _prf1(running_tp, running_fp, running_fn)
        return {
            "loss": float(total_loss / max(1, n_batches)),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "roc_auc": float(verification.roc_auc),
            "youden_j": float(verification.youden_j),
            "optimal_threshold": float(verification.optimal_threshold),
            "tar_at_far_1e3": float(verification.tar_at_far_1e3),
            "tar_at_far_1e4": float(verification.tar_at_far_1e4),
            "recall_at_far_1e2": float(verification.recall_at_far_1e2),
            "threshold_at_far_1e2": float(verification.threshold_at_far_1e2),
        }

    def _collect_predictions(self, loader: Iterable[Any]) -> Predictions:
        predictions, _, _ = self._predict_with_loss(loader)
        return predictions

    def _predict_with_loss(
        self,
        loader: Iterable[Any],
    ) -> tuple[Predictions, float, int]:
        import torch

        self.model.eval()
        y_true: list[float] = []
        y_score: list[float] = []
        total_loss = 0.0
        n_batches = 0
        device = self._infer_device()
        with torch.no_grad():
            for batch in loader:
                x_a, x_b, label = unpack_pair_batch(batch)
                x_a = x_a.to(device)
                x_b = x_b.to(device)
                label = label.to(device)
                with self._autocast():
                    output = (
                        self.model(x_a, x_b)
                        if self._call_model_with_pair()
                        else self.model((x_a, x_b))
                    )
                    loss = self.loss(output, label)
                total_loss += detach_to_float(loss)
                n_batches += 1
                # ravel
                out = output.detach().reshape(-1).float().cpu().numpy().tolist()
                lab = label.detach().reshape(-1).float().cpu().numpy().tolist()
                y_score.extend(out)
                y_true.extend(lab)
        predictions = Predictions(
            y_true=tuple(int(round(v)) for v in y_true),
            y_score=tuple(float(v) for v in y_score),
        )
        return predictions, total_loss, n_batches

    def _infer_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover -- empty model
            import torch

            return torch.device("cpu")

    # ------------------------------------------------------------------ #
    # Misc                                                               #
    # ------------------------------------------------------------------ #
    def _call_model_with_pair(self) -> bool:
        """Whether the model accepts ``(x_a, x_b)`` as two args.

        :class:`bat_models.SiameseModel` accepts either a tuple or a
        ``(B, 2, C, H, W)`` tensor; we always pass a tuple via the
        single-arg path to avoid signature surprises.
        """
        return False

    def _maybe_save_best(self, epoch: int, metrics: dict[str, float]) -> None:
        # ``loss`` is min-mode; the rest are max-mode (handled by the
        # tracker). The CheckpointManager owns the on-disk symlink dance.
        for key in ("f1", "recall", "precision", "loss", "recall_at_far_1e2"):
            if key not in metrics:
                continue
            improved = self.best.update(key, metrics[key], epoch=epoch)
            if improved:
                self.checkpoints.write_best(
                    metric_name=key,
                    epoch=epoch,
                    state=self._build_state_dict(epoch=epoch),
                )

    def _log_verification(
        self,
        metrics: VerificationMetrics,
        *,
        section: str,
        step: int,
        classification: dict[str, float] | None = None,
    ) -> None:
        # Performance metrics (all on [0, 1] scale).
        payload: dict[str, float] = {
            "roc_auc": float(metrics.roc_auc),
            "youden_j": float(metrics.youden_j),
            "tar_at_far_1e3": float(metrics.tar_at_far_1e3),
            "tar_at_far_1e4": float(metrics.tar_at_far_1e4),
            "recall_at_far_1e2": float(metrics.recall_at_far_1e2),
            # Decision thresholds (not on [0, 1] scale — grouped separately
            # so MLflow keeps them out of the performance-metric charts).
            "threshold/optimal_threshold": float(metrics.optimal_threshold),
            "threshold/threshold_at_far_1e2": float(metrics.threshold_at_far_1e2),
        }
        if classification is not None:
            for key, value in classification.items():
                payload[key] = float(value)
        self._log(
            section,
            payload,
            step=step,
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
    # Resume support                                                     #
    # ------------------------------------------------------------------ #
    def load(self, path: str | Path) -> None:
        """Resume training from a checkpoint produced by :func:`save_checkpoint`."""
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
        """Manual checkpoint dump (mostly for tests)."""
        return save_checkpoint(self._build_state_dict(epoch=0), path)


# ---------------------------------------------------------------------------
# Confusion helpers (kept private to this module)
# ---------------------------------------------------------------------------


def _confusion_pieces(
    output: torch.Tensor,
    label: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, tn, fn)`` at a fixed ``0.5`` threshold."""
    pred = (output.detach().reshape(-1) >= threshold).long()
    lab = (label.detach().reshape(-1) >= 0.5).long()
    tp = int((pred & lab).sum().item())
    fp = int((pred & (1 - lab)).sum().item())
    tn = int(((1 - pred) & (1 - lab)).sum().item())
    fn = int(((1 - pred) & lab).sum().item())
    return tp, fp, tn, fn


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return recall, precision, f1


def _confusion_at_threshold(
    predictions: Predictions,
    *,
    threshold: float,
) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for y_t, y_s in zip(predictions.y_true, predictions.y_score, strict=False):
        pred_pos = float(y_s) >= threshold
        if y_t == 1 and pred_pos:
            tp += 1
        elif y_t == 1 and not pred_pos:
            fn += 1
        elif y_t == 0 and pred_pos:
            fp += 1
        else:
            tn += 1
    return tp, fp, tn, fn


__all__ = ["PairTrainer"]
