"""Runtime helpers behind the ``bat-cli`` command surface.

The CLI intentionally stays thin: this module owns Hydra composition,
component construction, and the small amount of orchestration needed to
connect the workspace packages.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from bat_core import EvalReport, ImageRecord, Manifest, Predictions, SaliencyImage
from bat_training import make_trainer
from bat_training._common import TrainerConfig


class CliRuntimeError(RuntimeError):
    """Raised when CLI inputs cannot be resolved into a runnable experiment."""


# (beats, candidate_metric, incumbent_metric_or_None) -> approved-by-user
PromotionConfirm = Any  # Callable[[bool, float | None, float | None], bool]


@dataclass
class ExplanationArtifacts:
    """Saliency + projection artifacts returned by ``_build_explanations``."""

    saliency_images: list[SaliencyImage]
    projection_images: list[SaliencyImage]


@dataclass
class PromotionDecision:
    """Outcome of the post-training promotion gate."""

    mode: str  # "off" | "auto" | "prompt"
    beats: bool
    candidate_metric: float | None
    incumbent_metric: float | None
    promoted: bool
    declined: bool = False  # True iff prompt-mode user said "no"


@dataclass
class TrainingBundle:
    """Resolved runtime objects used by train/evaluate/sweep commands."""

    cfg: dict[str, Any]
    manifest: Manifest
    model: Any
    loss: Any
    trainer_cfg: TrainerConfig
    train_loader: Any
    val_loader: Any | None
    test_loader: Any | None
    trainer_kwargs: dict[str, Any]


@dataclass
class TrainResult:
    """Summary returned after a train command."""

    run_id: str
    output_dir: Path
    report_path: Path | None
    warnings: list[str]
    eval_report: EvalReport | None = None
    permutation_dir: Path | None = None
    explanations_dir: Path | None = None
    promotion: PromotionDecision | None = None


@dataclass
class PermutationResult:
    """Summary returned after a permutation-test command."""

    output_dir: Path
    n_permutations: int
    metrics_tested: tuple[str, ...]
    p_values: dict[str, float]


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root by walking upward from *start*."""

    here = (start or Path.cwd()).resolve()
    candidates = (here, *here.parents)
    for candidate in candidates:
        if (candidate / "configs" / "config.yaml").exists() and (
            candidate / "pyproject.toml"
        ).exists():
            return candidate
    raise CliRuntimeError(
        f"could not find repo root from {here}; expected configs/config.yaml and pyproject.toml"
    )


def compose_config(
    *,
    config_name: str = "config",
    experiment: str | None = None,
    overrides: Sequence[str] = (),
    root: Path | None = None,
) -> dict[str, Any]:
    """Compose a Hydra config into a plain Python dictionary."""

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    repo = find_project_root(root)
    config_dir = repo / "configs"
    hydra_overrides: list[str] = []
    if experiment:
        hydra_overrides.append(f"+experiment={experiment}")
    hydra_overrides.extend(overrides)

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    plain = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(plain, dict):
        raise CliRuntimeError("Hydra composition did not produce a mapping")
    return plain


def default_output_dir(cfg: Mapping[str, Any], root: Path | None = None) -> Path:
    """Return the default timestamped output directory for a composed config."""

    repo = find_project_root(root)
    base = Path(str(cfg.get("output_root", "outputs/runs")))
    if not base.is_absolute():
        base = repo / base
    model = _mapping(cfg.get("model")).get("arch", "model")
    loss = _mapping(cfg.get("loss")).get("type", _mapping(cfg.get("model")).get("head", "loss"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / f"{stamp}_{model}_{loss}"


def build_bundle(
    cfg: Mapping[str, Any],
    *,
    root: Path | None = None,
    output_dir: Path | None = None,
) -> TrainingBundle:
    """Resolve manifest, model, loss, trainer config, and data loaders."""

    repo = find_project_root(root)
    cfg_dict = dict(cfg)
    manifest = load_manifest(cfg_dict, root=repo)
    family = _model_family(cfg_dict)
    image_size = _resolve_image_size(cfg_dict)
    batch_size = int(
        _trainer_value(cfg_dict, "batch_size", _data_value(cfg_dict, "batch_size", 32))
    )
    num_workers = int(
        _trainer_value(cfg_dict, "num_workers", _data_value(cfg_dict, "num_workers", 0))
    )

    # Reproducibility: with `trainer.deterministic=true`, seed the global
    # RNG stack *before* building the model, and produce a fixed-seed
    # generator for the train DataLoader's shuffler. Without both, two
    # runs with identical Hydra cfg diverge from epoch 1 — the trainer's
    # own `set_deterministic_mode` call fires after the model has
    # already been initialized from the unseeded global generator, and
    # the DataLoader's default shuffler reads from that same generator
    # after it has been advanced.
    train_generator = _build_train_generator(cfg_dict)

    if family == "pair":
        train_ds = ManifestPairDataset(manifest, "train", image_size=image_size)
        val_ds = _optional_pair_dataset(manifest, "val", image_size=image_size)
        test_ds = _optional_pair_dataset(manifest, "test", image_size=image_size)
        train_loader = _loader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=train_generator,
        )
        val_loader = (
            _loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            if val_ds is not None
            else None
        )
        test_loader = (
            _loader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            if test_ds is not None
            else None
        )
        num_classes = 1
        trainer_kwargs: dict[str, Any] = {}
    else:
        from bat_data import BatDataset

        # Pretrained torchvision backbones expect ImageNet-normalised inputs;
        # apply the same normalisation in training and every eval path.
        normalize = (
            "imagenet"
            if _is_pretrained(_mapping(cfg_dict.get("model")).get("pretrained", True))
            else None
        )
        train_ds = BatDataset(manifest, split="train", image_size=image_size, normalize=normalize)
        val_ds = _optional_embedding_dataset(
            manifest, "val", image_size=image_size, normalize=normalize
        )
        test_ds = _optional_embedding_dataset(
            manifest, "test", image_size=image_size, normalize=normalize
        )
        train_loader = _loader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=_embedding_collate,
            generator=train_generator,
        )
        val_loader = (
            _loader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_embedding_collate,
            )
            if val_ds is not None
            else None
        )
        test_loader = (
            _loader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_embedding_collate,
            )
            if test_ds is not None
            else None
        )
        num_classes = train_ds.num_classes
        trainer_kwargs = {
            "eval_manifest": manifest,
            "eval_split": "test",
            # Eval/val/permutation embeddings must use the same edge length
            # and channel normalisation the model trained at.
            "image_size": image_size,
            "normalize": normalize,
        }

    resolved_output = output_dir or default_output_dir(cfg_dict, root=repo)
    trainer_cfg = build_trainer_config(cfg_dict, output_dir=resolved_output)
    trainer_cfg.warmup_steps = _warmup_steps(cfg_dict, trainer_cfg, train_loader)

    return TrainingBundle(
        cfg=cfg_dict,
        manifest=manifest,
        model=build_model(cfg_dict, num_classes=num_classes),
        loss=build_loss(cfg_dict),
        trainer_cfg=trainer_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        trainer_kwargs=trainer_kwargs,
    )


def load_manifest(cfg: Mapping[str, Any], *, root: Path | None = None) -> Manifest:
    """Load the manifest configured by ``data.manifest_path``."""

    from bat_data import manifest_from_csv

    repo = find_project_root(root)
    data_cfg = _mapping(cfg.get("data"))
    manifest_path = data_cfg.get("manifest_path")
    if manifest_path is None:
        raise CliRuntimeError("data.manifest_path is required")
    path = Path(str(manifest_path))
    if not path.is_absolute():
        path = repo / path
    if not path.exists():
        raise CliRuntimeError(f"manifest CSV does not exist: {path}")
    return manifest_from_csv(path)


def build_trainer_config(cfg: Mapping[str, Any], *, output_dir: Path) -> TrainerConfig:
    """Map Hydra trainer config into :class:`bat_training.TrainerConfig`."""

    trainer = _mapping(cfg.get("trainer"))
    early = _mapping(trainer.get("early_stop"))
    monitor = str(early.get("monitor", "val/f1"))
    bare_monitor = monitor.split("/", 1)[1] if "/" in monitor else monitor
    early_enabled = bool(early.get("enabled", False))
    mode = str(early.get("mode", "min" if bare_monitor == "loss" else "max"))

    return TrainerConfig(
        epochs=int(trainer.get("epochs", 1)),
        optimizer=str(trainer.get("optimizer", "adam")),
        lr=float(trainer.get("lr", 1e-4)),
        weight_decay=float(trainer.get("weight_decay", 0.0)),
        momentum=float(trainer.get("momentum", trainer.get("beta_1", 0.9))),
        warmup_steps=int(trainer.get("warmup_steps", 0)),
        min_lr_ratio=float(trainer.get("min_lr_ratio", 0.0)),
        gradient_accumulation_steps=int(trainer.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(trainer.get("mixed_precision", "no")),
        ema_decay=float(trainer.get("ema_decay", 0.0)),
        early_stopping_patience=int(early.get("patience", 0)) if early_enabled else 0,
        early_stopping_min_delta=float(early.get("min_delta", 1e-3)),
        early_stopping_monitor=bare_monitor,
        early_stopping_mode=mode,
        deterministic=bool(trainer.get("deterministic", False)),
        seed=int(cfg.get("seed", 0)),
        output_dir=Path(output_dir),
        artifact_retention=dict(_mapping(trainer.get("artifact_retention"))),
        log_every_n_steps=int(trainer.get("log_every_n_steps", 50)),
        val_verification=bool(trainer.get("val_verification", True)),
    )


def build_model(cfg: Mapping[str, Any], *, num_classes: int) -> Any:
    """Build a model from the composed config."""

    model_cfg = _mapping(cfg.get("model"))
    loss_cfg = _mapping(cfg.get("loss"))
    family = str(model_cfg.get("family", loss_cfg.get("family", "")))
    arch = str(model_cfg.get("arch", "")).lower()

    if family == "pair" or arch.startswith("siamese"):
        from bat_models import SiameseModel

        return SiameseModel()

    from bat_models import AdaFaceModel, ArcFaceModel
    from bat_models.backbones import resnet50_backbone

    if arch and arch != "resnet50":
        raise CliRuntimeError(f"unsupported embedding model arch: {arch!r}")

    backbone = resnet50_backbone(pretrained=_is_pretrained(model_cfg.get("pretrained", True)))
    head = str(model_cfg.get("head", loss_cfg.get("type", "arcface"))).lower()
    embedding_dim = int(model_cfg.get("embedding_dim", 512))
    margin = float(loss_cfg.get("margin", 0.5 if head == "arcface" else 0.4))
    scale = float(loss_cfg.get("scale", 64.0))

    if head == "arcface":
        return ArcFaceModel(
            backbone=backbone,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            scale=scale,
        )
    if head == "adaface":
        return AdaFaceModel(
            backbone=backbone,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
            h=float(loss_cfg.get("h", 0.333)),
            scale=scale,
        )
    raise CliRuntimeError(f"unsupported embedding head: {head!r}")


def build_loss(cfg: Mapping[str, Any]) -> Any:
    """Build a loss from the composed config."""

    loss_cfg = _mapping(cfg.get("loss"))
    loss_type = str(loss_cfg.get("type", loss_cfg.get("head", ""))).lower()
    reduction = str(loss_cfg.get("reduction", "mean"))

    if loss_type == "bce":
        from bat_losses import BCELoss

        return BCELoss(from_logits=bool(loss_cfg.get("from_logits", False)), reduction=reduction)
    if loss_type in {"binary_focal", "focal"}:
        from bat_losses import FocalLoss

        return FocalLoss(
            alpha=float(loss_cfg.get("alpha", 0.75)),
            gamma=float(loss_cfg.get("gamma", 2.0)),
            from_logits=bool(loss_cfg.get("from_logits", False)),
            reduction=reduction,
        )
    if loss_type == "triplet":
        from bat_losses import TripletLoss

        return TripletLoss(
            margin=float(loss_cfg.get("margin", 0.2)),
            p=int(loss_cfg.get("p", 2)),
            reduction=reduction,
        )
    if loss_type == "arcface":
        from bat_losses import ArcFaceLoss

        return ArcFaceLoss(
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
            reduction=reduction,
        )
    if loss_type == "adaface":
        from bat_losses import AdaFaceLoss

        return AdaFaceLoss(
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
            reduction=reduction,
        )
    if loss_type == "cosface":
        from bat_losses import CosFaceLoss

        return CosFaceLoss(
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
            reduction=reduction,
        )
    if loss_type in {"subcenter", "subcenter_arcface", "subcenter-arcface"}:
        from bat_losses import SubCenterArcFaceLoss

        return SubCenterArcFaceLoss(
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
            reduction=reduction,
        )
    raise CliRuntimeError(f"unsupported loss type: {loss_type!r}")


def run_training(
    cfg: Mapping[str, Any],
    *,
    root: Path | None = None,
    output_dir: Path | None = None,
    mlflow_experiment: str = "bat-face-recognition",
    tracking_uri: str | None = None,
    run_name: str | None = None,
    run_tags: Mapping[str, str] | None = None,
    use_mlflow: bool = True,
    promote: bool = False,
    prompt_promote: bool = False,
    promotion_criterion: str = "test/roc_auc",
    promotion_confirm: PromotionConfirm | None = None,
    run_permutation: bool = True,
    permutation_n: int = 1000,
    permutation_seed: int = 42,
    run_explanations: bool = True,
    explanations_per_split: int = 2,
    explanations_max_per_identity: int = 8,
) -> TrainResult:
    """Train, evaluate, render a PDF, and optionally log to MLflow."""

    from bat_reporting import ReportData, build_unified_pdf
    from bat_tracking import start_run

    if promote and prompt_promote:
        raise CliRuntimeError("--promote and --prompt-promote are mutually exclusive")

    repo = find_project_root(root)
    resolved_output = output_dir or default_output_dir(cfg, root=repo)
    bundle = build_bundle(cfg, root=repo, output_dir=resolved_output)
    warnings: list[str] = []

    context = (
        start_run(
            experiment_name=mlflow_experiment,
            run_name=run_name,
            tags={"entrypoint": "bat-cli", **(run_tags or {})},
            tracking_uri=tracking_uri,
        )
        if use_mlflow
        else nullcontext(None)
    )

    with context as tracker:
        run_id = _tracker_run_id(tracker)
        if tracker is not None:
            tracker.log_config(bundle.cfg)
            tracker.log_params(
                _audit_params(bundle.cfg, manifest_hash=bundle.manifest.manifest_hash)
            )

        trainer_kwargs = dict(bundle.trainer_kwargs)
        trainer_kwargs["tracker"] = tracker
        trainer_kwargs["run_id"] = run_id
        trainer = make_trainer(
            cfg=bundle.trainer_cfg,
            model=bundle.model,
            loss=bundle.loss,
            **trainer_kwargs,
        )
        artifacts = trainer.fit(bundle.train_loader, bundle.val_loader)

        eval_report = _safe(
            "test evaluation",
            warnings,
            lambda: _run_test(trainer, bundle, output_dir=resolved_output, warnings=warnings),
        )

        if eval_report is not None:
            figures_dir = resolved_output / "figures"
            from bat_stats import save_test_curves

            written = _safe(
                "test curves",
                warnings,
                lambda: save_test_curves(eval_report, bundle.cfg, figures_dir),
            )
            if tracker is not None and written:
                _safe(
                    "MLflow test curves upload",
                    warnings,
                    lambda: tracker.log_artifact(figures_dir, "figures"),
                )

        explanations: ExplanationArtifacts | None = None
        explanations_dir: Path | None = None
        if run_explanations:
            explanations_dir = resolved_output / "explanations"
            explanation_image_size = _resolve_image_size(bundle.cfg)
            explanation_normalize = (
                "imagenet"
                if (
                    _model_family(bundle.cfg) == "embedding"
                    and _is_pretrained(_mapping(bundle.cfg.get("model")).get("pretrained", True))
                )
                else None
            )
            explanations = _safe(
                "explanations",
                warnings,
                lambda: _build_explanations(
                    bundle.model,
                    bundle.manifest,
                    output_dir=explanations_dir,
                    samples_per_split=explanations_per_split,
                    max_samples_per_identity=explanations_max_per_identity,
                    image_size=explanation_image_size,
                    normalize=explanation_normalize,
                ),
            )

        report_data_kwargs: dict[str, Any] = dict(
            cfg=bundle.cfg,
            run_id=run_id or artifacts.run_id or "local",
            manifest_hash=bundle.manifest.manifest_hash,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            eval_report=eval_report,
            extra={"best_metrics": artifacts.best_metrics},
        )
        if explanations is not None:
            from bat_reporting import EmbeddingProjection

            report_data_kwargs["saliency_images"] = explanations.saliency_images
            if explanations.projection_images:
                report_data_kwargs["embedding_projection"] = EmbeddingProjection(
                    images=explanations.projection_images
                )

        report_path = _safe(
            "PDF report",
            warnings,
            lambda: build_unified_pdf(
                ReportData(**report_data_kwargs),
                resolved_output / "post_training_report.pdf",
            ),
        )

        if tracker is not None and report_path is not None:
            _safe(
                "MLflow report upload",
                warnings,
                lambda: tracker.log_artifact(report_path, "reports"),
            )

        # Val-selected operating threshold + comparable cosine metric (PR6).
        # Computing them once here lets the permutation test use the
        # validation-derived threshold (rather than a test-fit one) and lets
        # the pair family report a cosine metric directly comparable to the
        # embedding families.
        val_report = _safe("val eval", warnings, lambda: _val_eval_report(trainer, bundle))
        if eval_report is not None and eval_report.predictions is not None:
            eval_cfg = _mapping(bundle.cfg.get("evaluation"))
            strategy = str(eval_cfg.get("permutation_threshold", "youden_j")).strip()
            val_thr = _val_threshold(val_report, strategy)
            if val_thr is not None and tracker is not None:
                from bat_evaluation import confusion_at_threshold

                c = confusion_at_threshold(
                    eval_report.predictions.y_true, eval_report.predictions.y_score, val_thr
                )
                _safe(
                    "val-threshold test metrics",
                    warnings,
                    lambda: tracker.log_metrics(
                        "test_val_threshold",
                        {"f1": c.f1, "precision": c.precision, "recall": c.recall},
                        0,
                    ),
                )
        if _model_family(bundle.cfg) == "pair" and tracker is not None:
            cos = _safe(
                "comparable cosine eval",
                warnings,
                lambda: _comparable_cosine_report(trainer, bundle),
            )
            if cos is not None:
                _safe(
                    "MLflow comparable cosine",
                    warnings,
                    lambda: tracker.log_metrics(
                        "test_cosine",
                        {
                            "roc_auc": float(cos.verification.roc_auc),
                            "youden_j": float(cos.verification.youden_j),
                            "recall_at_far_1e2": float(cos.verification.recall_at_far_1e2),
                            "precision_at_recall_0p75": float(
                                cos.verification.precision_at_recall_0p75
                            ),
                        },
                        0,
                    ),
                )

        permutation_dir: Path | None = None
        if run_permutation and eval_report is not None and eval_report.predictions is not None:
            # Pick the threshold the permutation test binarises at. Default
            # Youden-J reproduces legacy behaviour; ``far_1e2`` uses the
            # recall-at-FAR=1e-2 operating point. Without honouring this,
            # embedding-model cosine scores (which cluster within ~1e-7 of
            # 1.0) collapse to all-positive predictions at threshold=0.5 and
            # the null distribution flatlines — masking real ROC-AUC signal.
            # Prefer the validation-derived threshold (PR6); fall back to the
            # test-fit threshold only when no val report is available.
            eval_cfg = _mapping(bundle.cfg.get("evaluation"))
            strategy = str(eval_cfg.get("permutation_threshold", "youden_j")).strip()
            perm_threshold = _val_threshold(val_report, strategy)
            if perm_threshold is None:
                if strategy == "far_1e2":
                    perm_threshold = float(eval_report.verification.threshold_at_far_1e2)
                elif strategy == "recall_0p75":
                    perm_threshold = float(eval_report.verification.threshold_at_recall_0p75)
                else:
                    perm_threshold = float(eval_report.verification.optimal_threshold)
            permutation_result = _safe(
                "permutation test",
                warnings,
                lambda: run_permutation_test(
                    bundle.cfg,
                    predictions=eval_report.predictions,
                    root=repo,
                    output_dir=resolved_output / "permutation",
                    n_permutations=permutation_n,
                    threshold=perm_threshold,
                    seed=permutation_seed,
                    verbose=False,
                ),
            )
            if permutation_result is not None:
                permutation_dir = permutation_result.output_dir
                if tracker is not None:
                    _safe(
                        "MLflow permutation upload",
                        warnings,
                        lambda: tracker.log_artifact(permutation_dir, "permutation"),
                    )

        promotion_decision = _maybe_promote(
            tracker=tracker,
            run_id=run_id,
            criterion=promotion_criterion,
            promote=promote,
            prompt_promote=prompt_promote,
            confirm=promotion_confirm,
            warnings=warnings,
        )

    return TrainResult(
        run_id=run_id,
        output_dir=resolved_output,
        report_path=report_path,
        warnings=warnings,
        eval_report=eval_report,
        permutation_dir=permutation_dir,
        explanations_dir=explanations_dir if explanations is not None else None,
        promotion=promotion_decision,
    )


def _maybe_promote(
    *,
    tracker: Any | None,
    run_id: str,
    criterion: str,
    promote: bool,
    prompt_promote: bool,
    confirm: PromotionConfirm | None,
    warnings: list[str],
) -> PromotionDecision:
    """Apply the post-training champion-promotion policy."""

    if tracker is None or not run_id:
        return PromotionDecision(
            mode="off", beats=False, candidate_metric=None, incumbent_metric=None, promoted=False
        )
    if not promote and not prompt_promote:
        return PromotionDecision(
            mode="off", beats=False, candidate_metric=None, incumbent_metric=None, promoted=False
        )

    probe = _safe(
        "promotion comparison",
        warnings,
        lambda: tracker.would_promote(run_id, criterion),
    )
    if probe is None:
        return PromotionDecision(
            mode="auto" if promote else "prompt",
            beats=False,
            candidate_metric=None,
            incumbent_metric=None,
            promoted=False,
        )
    beats, candidate_metric, incumbent_metric = probe

    if not beats:
        return PromotionDecision(
            mode="auto" if promote else "prompt",
            beats=False,
            candidate_metric=candidate_metric,
            incumbent_metric=incumbent_metric,
            promoted=False,
        )

    if prompt_promote:
        approved = bool(
            confirm(beats, candidate_metric, incumbent_metric)
            if confirm is not None
            else _default_promotion_confirm(beats, candidate_metric, incumbent_metric)
        )
        if not approved:
            return PromotionDecision(
                mode="prompt",
                beats=True,
                candidate_metric=candidate_metric,
                incumbent_metric=incumbent_metric,
                promoted=False,
                declined=True,
            )

    promoted = bool(
        _safe(
            "champion promotion",
            warnings,
            lambda: tracker.promote_to_champion(run_id, criterion),
        )
    )
    return PromotionDecision(
        mode="prompt" if prompt_promote else "auto",
        beats=True,
        candidate_metric=candidate_metric,
        incumbent_metric=incumbent_metric,
        promoted=promoted,
    )


def _default_promotion_confirm(
    beats: bool, candidate: float | None, incumbent: float | None
) -> bool:
    """Default click-based confirm for the prompt-promote flow."""

    import click

    return click.confirm(
        f"Candidate {candidate:.4f} beats champion "
        f"{incumbent if incumbent is None else f'{incumbent:.4f}'} on the criterion. "
        "Promote?",
        default=False,
    )


def run_evaluation(
    cfg: Mapping[str, Any],
    *,
    checkpoint: Path | None = None,
    root: Path | None = None,
    output_dir: Path | None = None,
) -> EvalReport:
    """Load a configured model/trainer and evaluate the test split."""

    resolved_output = output_dir or default_output_dir(cfg, root=root)
    bundle = build_bundle(cfg, root=root, output_dir=resolved_output)
    trainer = make_trainer(
        cfg=bundle.trainer_cfg,
        model=bundle.model,
        loss=bundle.loss,
        **bundle.trainer_kwargs,
    )
    if checkpoint is not None:
        trainer.load(checkpoint)
    report = _run_test(trainer, bundle)
    if report is not None:
        from bat_stats import save_test_curves

        save_test_curves(report, bundle.cfg, resolved_output / "figures")
    return report


def _build_explanations(
    model: Any,
    manifest: Manifest,
    *,
    output_dir: Path,
    samples_per_split: int = 2,
    max_samples_per_identity: int = 8,
    image_size: int,
    normalize: str | None = None,
) -> ExplanationArtifacts:
    """Generate saliency + projection artifacts spanning train/val/test splits.

    Picks ``samples_per_split`` distinct identities per split (deterministic:
    sorted by record path), takes one record each, runs the family-appropriate
    saliency adapter on the union, then runs the embedding projection adapter
    on up to ``max_samples_per_identity`` records per identity (every identity
    stays represented).

    ``image_size`` is the edge length the model was trained at (Siamese: 105,
    embedding ResNet backbones: 112, read from ``model.input_edge_length``).
    Both adapters resize inputs to that edge before invoking the model, since
    their dataclass defaults (``EmbeddingProjectionAdapter.input_size=224``)
    silently mismatch the trained resolution and cascade-fail the explanation
    block.

    Identity strings on the resulting :class:`bat_core.SaliencyImage` are
    prefixed with the source split (``"train:W"``, ``"val:H"``, ``"test:R"``)
    so the unified PDF's saliency composite makes split provenance obvious.
    """

    from bat_interpretability import EmbeddingProjectionAdapter, select_adapter

    output_dir.mkdir(parents=True, exist_ok=True)
    saliency_samples, path_to_split = _sample_explanation_records(manifest, samples_per_split)
    if not saliency_samples:
        return ExplanationArtifacts(saliency_images=[], projection_images=[])

    adapter = replace(select_adapter(model), input_size=image_size)
    raw = list(adapter.explain(model, saliency_samples))
    saliency_images = [
        replace(img, identity=f"{path_to_split.get(img.image_path, '?')}:{img.identity}")
        for img in raw
    ]

    projection_samples = _sample_projection_records(manifest, max_samples_per_identity)
    projection_adapter = EmbeddingProjectionAdapter(
        method="both",
        output_dir=output_dir,
        input_size=image_size,
        normalize=normalize,
    )
    projection_images = (
        list(projection_adapter.explain(model, projection_samples)) if projection_samples else []
    )

    return ExplanationArtifacts(
        saliency_images=saliency_images,
        projection_images=projection_images,
    )


def _sample_explanation_records(
    manifest: Manifest, samples_per_split: int
) -> tuple[list[ImageRecord], dict[Path, str]]:
    """Pick ``samples_per_split`` representative records per split."""

    samples: list[ImageRecord] = []
    path_to_split: dict[Path, str] = {}
    for split in ("train", "val", "test"):
        records = manifest.filter_split(split)  # type: ignore[arg-type]
        if not records:
            continue
        by_identity: dict[str, list[ImageRecord]] = defaultdict(list)
        for r in records:
            by_identity[r.identity].append(r)
        # Identities sorted alphabetically; pick first record (sorted by path) per identity.
        chosen_identities = sorted(by_identity)[:samples_per_split]
        for identity in chosen_identities:
            picked = sorted(by_identity[identity], key=lambda r: str(r.path))[0]
            samples.append(picked)
            path_to_split[picked.path] = split
    return samples, path_to_split


def _sample_projection_records(manifest: Manifest, max_per_identity: int) -> list[ImageRecord]:
    """Sample up to ``max_per_identity`` records per identity for the projection.

    The old global stride-cap could drop entire identities (and routinely left
    only one point per identity), making the t-SNE/UMAP clusters impossible to
    read. Sampling per identity keeps every identity represented with a small,
    even number of points. Selection is deterministic (sorted by path) so the
    same manifest always yields the same projection.
    """

    by_identity: dict[str, list[ImageRecord]] = defaultdict(list)
    for split in ("train", "val", "test"):
        for r in manifest.filter_split(split):  # type: ignore[arg-type]
            by_identity[r.identity].append(r)
    if not by_identity:
        return []
    cap = max(1, int(max_per_identity))
    samples: list[ImageRecord] = []
    for identity in sorted(by_identity):
        records = sorted(by_identity[identity], key=lambda r: str(r.path))
        samples.extend(records[:cap])
    return samples


def run_permutation_test(
    cfg: Mapping[str, Any],
    *,
    predictions: Predictions | None = None,
    checkpoint: Path | None = None,
    root: Path | None = None,
    output_dir: Path | None = None,
    n_permutations: int = 1000,
    significance_level: float = 0.05,
    seed: int = 42,
    threshold: float = 0.5,
    metrics_to_test: Sequence[str] | None = None,
    degradation: bool = False,
    verbose: bool = False,
) -> PermutationResult:
    """Run the inference-mode permutation test on a configured run.

    If ``predictions`` is supplied (e.g. from an in-flight train pipeline),
    we skip re-evaluation and feed them straight to ``bat_stats``. Otherwise
    we recompose the cfg, optionally restore a checkpoint, run test eval,
    and use the resulting ``EvalReport.predictions``.
    """

    from bat_stats import run_inference_test

    repo = find_project_root(root)
    resolved_output = output_dir or (default_output_dir(cfg, root=repo) / "permutation")
    resolved_output.mkdir(parents=True, exist_ok=True)

    if predictions is None:
        report = run_evaluation(cfg, checkpoint=checkpoint, root=repo, output_dir=resolved_output)
        predictions = report.predictions
    if predictions is None:
        raise CliRuntimeError(
            "evaluation produced no predictions; permutation test needs a Predictions dataclass"
        )

    results = run_inference_test(
        cfg,
        predictions,
        threshold=threshold,
        n_permutations=n_permutations,
        significance_level=significance_level,
        metrics_to_test=metrics_to_test,
        degradation=degradation,
        seed=seed,
        output_dir=resolved_output,
        generate_report=True,
        verbose=verbose,
    )

    p_values = {name: float(metric.p_value) for name, metric in results.metrics.items()}
    return PermutationResult(
        output_dir=resolved_output,
        n_permutations=int(results.n_permutations),
        metrics_tested=tuple(p_values),
        p_values=p_values,
    )


def build_manifest_csv(
    *,
    input_dir: Path,
    output_csv: Path,
    species: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    skip_unparseable: bool = True,
) -> tuple[Path, Any]:
    """Build and split a manifest CSV from an image directory."""

    from bat_data import IdentitySplitter, build_manifest, manifest_to_csv

    manifest = build_manifest(
        input_dir,
        species=species,
        skip_unparseable=skip_unparseable,
        default_split="train",
    )
    splitter = IdentitySplitter(
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    split_manifest = splitter.split(manifest)
    path = manifest_to_csv(split_manifest, output_csv)
    return path, splitter.counts(split_manifest)


_SPECIES_LETTER = {"mauritius": "m", "rousettus": "r"}
_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def run_video_extraction(
    *,
    videos_dir: Path,
    output_root: Path,
    species: str,
    n_videos: int,
    seed: int,
    seg_weights: Path,
    pose_weights: Path,
    detector_weights: Path | None = None,
    device: str = "cuda",
    target_frames: int = 53,
    edge_length: int = 224,
    margin_ratio: float = 0.03,
    min_keypoint_confidence: float = 0.35,
    variants: Sequence[str] = ("original_bg", "green_bg", "random_bg", "face_ellipse"),
    frame_oversample: int = 20,
    min_quality: float = 0.0,
    require_pose: bool = True,
    random_bg_style: str = "picsum",
    build_manifests: bool = False,
    manifests_dir: Path | None = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    root: Path | None = None,
    log=print,
) -> dict[str, Any]:
    """Sample ``n_videos`` and extract eye-anchored face crops + bg variants.

    Each video is treated as one identity (the video stem). Models are loaded
    once and reused across all videos. Videos are picked by a seeded shuffle of
    the full set then taking the first ``n_videos`` — so a smaller ``n_videos``
    (e.g. a 3-video preview) is always a subset of a larger run with the same
    seed. Returns a summary dict (selection, per-variant counts, manifests).
    """
    import random

    import numpy as np
    from bat_preprocessing import (
        VideoExtractionConfig,
        VideoExtractor,
        YOLODetector,
        YOLOPoseEstimator,
        YOLOSegmenter,
    )

    repo = find_project_root(root)

    def _resolve(p: Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else repo / p

    videos_dir = _resolve(videos_dir)
    output_root = _resolve(output_root)
    seg_weights = _resolve(seg_weights)
    pose_weights = _resolve(pose_weights)
    detector_weights = _resolve(detector_weights) if detector_weights is not None else None
    if species not in _SPECIES_LETTER:
        raise CliRuntimeError(f"unsupported species {species!r}")
    if not videos_dir.is_dir():
        raise CliRuntimeError(f"videos dir not found: {videos_dir}")
    required = [seg_weights, pose_weights] + ([detector_weights] if detector_weights else [])
    for w in required:
        if not w.exists():
            raise CliRuntimeError(f"weights not found: {w}")

    all_videos = sorted(p for p in videos_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTENSIONS)
    if not all_videos:
        raise CliRuntimeError(f"no videos found under {videos_dir}")
    shuffled = list(all_videos)
    random.Random(seed).shuffle(shuffled)
    selected = shuffled[: min(n_videos, len(shuffled))]
    letter = _SPECIES_LETTER[species]
    variants = list(variants)

    log(f"Loading models on device={device!r} (once, reused across {len(selected)} videos)…")
    segmenter = YOLOSegmenter({"weights": str(seg_weights), "device": device})
    pose = YOLOPoseEstimator({"weights": str(pose_weights), "device": device})
    detector = (
        YOLODetector({"weights": str(detector_weights), "device": device})
        if detector_weights is not None
        else None
    )
    if detector is not None:
        log(f"  detector (presence gate): {detector_weights.name}")

    config = VideoExtractionConfig(
        output_dir=output_root,
        identity="__placeholder__",
        species=species,  # type: ignore[arg-type]
        source="video",
        split="train",
        augmented=False,
        frame_stride=1,
        max_frames=None,  # collect all passers; subsample for temporal spread below
        min_quality=min_quality,
        edge_length=edge_length,
        margin_ratio=margin_ratio,
        require_pose=require_pose,
        align_mode="eye_anchored",
        min_keypoint_confidence=min_keypoint_confidence,
        require_confident_eyes=True,
        variants=variants,
        variant_output_dirs={v: output_root / v for v in variants},
        name_template=f"{letter}--{{identity}}--{{video}}.{{frame}}.jpg",
        output_ext=".jpg",
        random_bg_style=random_bg_style,
        random_seed=seed,
    )
    extractor = VideoExtractor(config, segmenter=segmenter, pose_estimator=pose, detector=detector)

    import cv2

    def _frame_idx(path: Path) -> int:
        # filename "{letter}--{identity}--{video}.{frame}.jpg" → trailing ".{frame}".
        try:
            return int(Path(path).stem.rsplit(".", 1)[-1])
        except (ValueError, IndexError):
            return -1

    per_video: dict[str, int] = {}
    per_variant: dict[str, int] = {v: 0 for v in variants}
    for i, video in enumerate(selected, 1):
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        candidates = max(1, target_frames * max(1, frame_oversample))
        stride = max(1, total // candidates) if total > 0 else 1

        config.identity = video.stem
        config.frame_stride = stride
        records = extractor.extract(video)

        # Group passers by frame index, then evenly subsample to target_frames so
        # the kept frames are spread across the clip (consecutive frames look alike).
        by_frame: dict[int, list] = defaultdict(list)
        for r in records:
            by_frame[_frame_idx(r.path)].append(r)
        frames_sorted = sorted(by_frame)
        if len(frames_sorted) > target_frames:
            pick = np.linspace(0, len(frames_sorted) - 1, target_frames).round().astype(int)
            keep = {frames_sorted[j] for j in sorted({int(p) for p in pick})}
        else:
            keep = set(frames_sorted)
        for fr, recs in by_frame.items():
            if fr in keep:
                for r in recs:
                    per_variant[Path(r.path).parent.name] = (
                        per_variant.get(Path(r.path).parent.name, 0) + 1
                    )
            else:  # drop the non-selected frames' files (all variants)
                for r in recs:
                    Path(r.path).unlink(missing_ok=True)
        per_video[video.stem] = len(keep)
        log(
            f"  [{i}/{len(selected)}] {video.name}: total={total} stride={stride} "
            f"passers={len(frames_sorted)} kept={len(keep)} (spread across clip)"
        )

    selection = {
        "seed": seed,
        "n_videos": len(selected),
        "videos": [v.name for v in selected],
        "species": species,
        "variants": variants,
        "target_frames": target_frames,
        "seg_weights": (
            str(seg_weights.relative_to(repo))
            if seg_weights.is_relative_to(repo)
            else str(seg_weights)
        ),
        "pose_weights": (
            str(pose_weights.relative_to(repo))
            if pose_weights.is_relative_to(repo)
            else str(pose_weights)
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {
        "selection": selection,
        "per_video": per_video,
        "per_variant": per_variant,
        "output_root": str(output_root),
    }

    if build_manifests:
        manifests_dir = _resolve(manifests_dir or Path("data/manifests"))
        manifests_dir.mkdir(parents=True, exist_ok=True)
        manifests: dict[str, Any] = {}
        for v in variants:
            variant_dir = output_root / v
            if not variant_dir.is_dir():
                continue
            csv_path = manifests_dir / f"{species}_aligned_{v}.csv"
            path, counts = build_manifest_csv(
                input_dir=variant_dir,
                output_csv=csv_path,
                species=species,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                seed=seed,
            )
            manifests[v] = {
                "csv": str(path),
                "train": counts.train,
                "val": counts.val,
                "test": counts.test,
                "total": counts.total,
            }
            log(
                f"  manifest {v}: {path} "
                f"(train={counts.train}, val={counts.val}, test={counts.test}, total={counts.total})"
            )
        summary["manifests"] = manifests

    return summary


def eval_report_to_dict(report: EvalReport) -> dict[str, Any]:
    """Convert an EvalReport to a JSON-friendly dictionary."""

    verification = report.verification
    payload: dict[str, Any] = {
        "verification": {
            "roc_auc": verification.roc_auc,
            "youden_j": verification.youden_j,
            "optimal_threshold": verification.optimal_threshold,
            "tar_at_far_1e3": verification.tar_at_far_1e3,
            "tar_at_far_1e4": verification.tar_at_far_1e4,
            "recall_at_far_1e2": verification.recall_at_far_1e2,
            "threshold_at_far_1e2": verification.threshold_at_far_1e2,
            "precision_at_recall_0p75": verification.precision_at_recall_0p75,
            "threshold_at_recall_0p75": verification.threshold_at_recall_0p75,
        }
    }
    if report.identification is not None:
        payload["identification"] = {
            "top1": report.identification.top1,
            "top5": report.identification.top5,
            "map": report.identification.map,
            "cmc": list(report.identification.cmc),
        }
    return payload


def format_json(payload: Mapping[str, Any]) -> str:
    """Pretty JSON formatter used by CLI output."""

    return json.dumps(payload, indent=2, sort_keys=True)


class ManifestPairDataset:
    """Reproducible random pair dataset over a manifest split.

    Each ``__getitem__(idx)`` draws a fresh anchor + partner pair via a
    deterministic per-index RNG (``random.Random(seed + idx)``), so the
    same ``idx`` always yields the same pair (dataloader replay /
    checkpoint resume stays bit-exact) but across an epoch the sampler
    explores all ``C(n, 2)`` positives and cross-class negatives instead
    of the O(n) cycle the previous deterministic walk produced.

    Epoch size is ``pairs_per_record * len(records)`` (default ``20`` →
    9 720 pairs for the 486-record rousettus train split), tuned to
    approach the legacy TF pipeline's pair diversity without exploding
    memory or step count.

    Pair label balance is exactly 50/50 by construction: even indices
    yield positives, odd indices yield negatives.
    """

    def __init__(
        self,
        manifest: Manifest,
        split: str,
        *,
        image_size: int = 105,
        pairs_per_record: int = 20,
        seed: int = 0,
    ) -> None:
        from bat_data import default_image_loader

        self.records = tuple(r for r in manifest.records if r.split == split)
        if not self.records:
            raise ValueError(f"no records found for split {split!r}")
        self.image_size = int(image_size)
        self.loader = lambda p: default_image_loader(Path(p), self.image_size)
        by_identity: dict[str, list[Any]] = defaultdict(list)
        for record in self.records:
            by_identity[record.identity].append(record)
        if len(by_identity) < 2:
            raise ValueError(f"pair dataset split {split!r} needs at least two identities")
        # Drop identities with only one record — they cannot produce a
        # within-identity positive pair, and rotating through them as
        # anchors corrupts the 50/50 label balance.
        self.by_identity = {k: tuple(v) for k, v in by_identity.items() if len(v) >= 2}
        if len(self.by_identity) < 2:
            raise ValueError(
                f"pair dataset split {split!r} needs at least two identities "
                "with >= 2 records each"
            )
        self.identities = tuple(sorted(self.by_identity))
        self._seed = int(seed)
        self._pairs_per_record = max(1, int(pairs_per_record))
        # Keep the eligible record pool aligned with the filtered identities
        # so __getitem__ never lands on a singleton-identity anchor.
        self._anchor_pool = tuple(r for r in self.records if r.identity in self.by_identity)

    def __len__(self) -> int:
        return len(self._anchor_pool) * self._pairs_per_record

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        import random

        import torch

        rng = random.Random(self._seed + index)
        anchor = rng.choice(self._anchor_pool)
        positive = index % 2 == 0
        if positive:
            candidates = [r for r in self.by_identity[anchor.identity] if r is not anchor]
            other = rng.choice(candidates) if candidates else anchor
            label = 1.0
        else:
            other_ids = [i for i in self.identities if i != anchor.identity]
            neg_identity = rng.choice(other_ids)
            other = rng.choice(self.by_identity[neg_identity])
            label = 0.0
        return (
            self.loader(anchor.path),
            self.loader(other.path),
            torch.tensor(label, dtype=torch.float32),
        )


def _val_eval_report(trainer: Any, bundle: TrainingBundle) -> EvalReport | None:
    """Best-effort validation EvalReport, used to source operating-point
    thresholds that are then applied to test (val-selected threshold policy).

    Returns ``None`` if a val report cannot be produced (e.g. no val loader for
    the pair family, or the val split lacks gallery/probe minimums).
    """
    try:
        if _model_family(bundle.cfg) == "embedding":
            return trainer.test(manifest=bundle.manifest, split="val")
        if bundle.val_loader is None:
            return None
        return trainer.test(bundle.val_loader)
    except Exception:  # pragma: no cover -- best-effort enrichment
        return None


def _val_threshold(report: EvalReport | None, strategy: str) -> float | None:
    """Pick the val operating threshold for ``strategy`` (youden_j/far_1e2/recall_0p75)."""
    if report is None:
        return None
    v = report.verification
    if strategy == "far_1e2":
        return float(v.threshold_at_far_1e2)
    if strategy == "recall_0p75":
        return float(v.threshold_at_recall_0p75)
    return float(v.optimal_threshold)


def _comparable_cosine_report(trainer: Any, bundle: TrainingBundle) -> EvalReport | None:
    """Cosine gallery/probe verification on ``forward_embedding`` for any model.

    For the pair family (Siamese) this provides the metric the paper claims all
    families share -- directly comparable to the embedding models -- alongside
    the native pair-head report. Returns ``None`` on any failure.
    """
    model = getattr(trainer, "model", None)
    if model is None or not hasattr(model, "forward_embedding"):
        return None
    image_size = _resolve_image_size(bundle.cfg)
    normalize = (
        "imagenet"
        if (
            _model_family(bundle.cfg) == "embedding"
            and _is_pretrained(_mapping(bundle.cfg.get("model")).get("pretrained", True))
        )
        else None
    )
    try:
        import torch
        from bat_data import default_image_loader
        from bat_evaluation.protocols import run_eval_protocol

        device = next(model.parameters()).device
        model.eval()

        def _embed(paths: list[Path]) -> Any:
            tensors = [default_image_loader(Path(p), image_size, normalize) for p in paths]
            stacked = torch.stack(tensors, dim=0).to(device)
            with torch.no_grad():
                return model.forward_embedding(stacked).detach().cpu()

        return run_eval_protocol(bundle.manifest, split="test", embed_fn=_embed)
    except Exception:  # pragma: no cover -- best-effort enrichment
        return None


def _run_test(
    trainer: Any,
    bundle: TrainingBundle,
    *,
    output_dir: Path | None = None,
    warnings: list[str] | None = None,
) -> EvalReport:
    """Run test eval, restoring a best-* checkpoint if ``evaluation.test_checkpoint`` requests it.

    Default (``test_checkpoint: final``) keeps the legacy behaviour of evaluating
    the in-memory model. ``auto`` resolves to the trainer's early-stop monitor
    metric (e.g. ``roc_auc`` for embedding, ``precision_at_recall_0p75`` for
    pair), so test eval uses the same checkpoint training actually selected --
    not the post-early-stop final/EMA weights. Any other value resolves to
    ``<output_dir>/best_model_<value>.pt`` and is loaded via ``trainer.load``
    before the test pass. Missing files degrade to "final" with a warning so
    that pipelines don't crash on a non-existent best checkpoint.
    """
    eval_cfg = _mapping(bundle.cfg.get("evaluation"))
    requested = str(eval_cfg.get("test_checkpoint", "final")).strip()
    if requested == "auto":
        # Resolve to whatever metric early stopping monitored, so the test
        # checkpoint matches the model selection criterion used in training.
        monitor = str(getattr(getattr(trainer, "cfg", None), "early_stopping_monitor", "") or "")
        requested = monitor.rsplit("/", 1)[-1].strip() or "final"
    if requested and requested != "final" and output_dir is not None:
        ckpt = output_dir / f"best_model_{requested}.pt"
        if ckpt.exists():
            try:
                trainer.load(ckpt)
            except Exception as exc:  # pragma: no cover -- defensive
                msg = f"failed to restore checkpoint {ckpt.name}: {exc}; using final model"
                if warnings is not None:
                    warnings.append(msg)
        else:
            msg = (
                f"evaluation.test_checkpoint={requested!r} requested but "
                f"{ckpt.name} not found; falling back to final model"
            )
            if warnings is not None:
                warnings.append(msg)

    if _model_family(bundle.cfg) == "embedding":
        return trainer.test(manifest=bundle.manifest, split="test")
    if bundle.test_loader is None:
        raise CliRuntimeError("test split is unavailable for pair evaluation")
    return trainer.test(bundle.test_loader)


def _optional_embedding_dataset(
    manifest: Manifest, split: str, *, image_size: int, normalize: str | None = None
) -> Any | None:
    from bat_data import BatDataset

    try:
        return BatDataset(manifest, split=split, image_size=image_size, normalize=normalize)
    except ValueError:
        return None


def _optional_pair_dataset(manifest: Manifest, split: str, *, image_size: int) -> Any | None:
    try:
        return ManifestPairDataset(manifest, split=split, image_size=image_size)
    except ValueError:
        return None


def _loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    collate_fn: Any | None = None,
    generator: Any | None = None,
) -> Any:
    """Wrap :class:`torch.utils.data.DataLoader`.

    ``generator`` is forwarded to ``DataLoader(..., generator=...)`` when
    ``shuffle=True``. Without an explicit generator the loader uses the
    global torch RNG, which has already been advanced by model weight
    init by the time iteration starts -- so identical Hydra configs can
    still produce different batch orders. Pass a fixed-seed generator
    here when ``cfg.trainer.deterministic=true`` to keep batch order
    bit-exact across runs.
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
    )


def _build_train_generator(cfg: Mapping[str, Any]) -> Any | None:
    """Return a fixed-seed `torch.Generator` when deterministic mode is on.

    Also primes the global RNG stack (`set_deterministic_mode`) so model
    weight init -- which runs *before* the trainer's own deterministic
    setup -- gets the same seed as the loader's shuffler.

    Returns ``None`` when ``cfg.trainer.deterministic`` is unset or
    false, leaving DataLoader behavior identical to the pre-fix default.
    """

    trainer_cfg = _mapping(cfg.get("trainer"))
    if not bool(trainer_cfg.get("deterministic", False)):
        return None

    seed = int(cfg.get("seed", 0))

    from bat_training.callbacks import set_deterministic_mode

    set_deterministic_mode(seed)

    import torch

    return torch.Generator().manual_seed(seed)


def _embedding_collate(batch: Sequence[Any]) -> tuple[Any, Any]:
    import torch

    images = [item[0] for item in batch]
    labels = [int(item[1]) for item in batch]
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def _model_family(cfg: Mapping[str, Any]) -> str:
    model_cfg = _mapping(cfg.get("model"))
    loss_cfg = _mapping(cfg.get("loss"))
    family = str(model_cfg.get("family", loss_cfg.get("family", "")))
    if family not in {"pair", "embedding"}:
        raise CliRuntimeError(f"model.family must be 'pair' or 'embedding'; got {family!r}")
    return family


def _resolve_image_size(cfg: Mapping[str, Any]) -> int:
    """Edge length the model trains/evals at, from ``model.input_edge_length``.

    Required, no default: a missing key previously fell back to 224, which a
    torchvision ResNet50 silently accepts (adaptive pool) — corrupting embeddings
    with no error. Fail loud instead.
    """
    model_cfg = _mapping(cfg.get("model"))
    if "input_edge_length" not in model_cfg:
        raise CliRuntimeError(
            "model.input_edge_length is required (e.g. 112 for ArcFace/AdaFace, "
            "105 for Siamese); none found in the composed config"
        )
    size = int(model_cfg["input_edge_length"])
    if size <= 0:
        raise CliRuntimeError(f"model.input_edge_length must be a positive int; got {size!r}")
    return size


def _warmup_steps(cfg: Mapping[str, Any], trainer_cfg: TrainerConfig, train_loader: Any) -> int:
    trainer = _mapping(cfg.get("trainer"))
    if "warmup_steps" in trainer:
        return int(trainer["warmup_steps"])
    warmup_epochs = int(trainer.get("warmup_epochs", 0))
    if warmup_epochs <= 0:
        return trainer_cfg.warmup_steps
    steps_per_epoch = max(1, len(train_loader))
    opt_steps = max(
        1,
        (steps_per_epoch + trainer_cfg.gradient_accumulation_steps - 1)
        // trainer_cfg.gradient_accumulation_steps,
    )
    return warmup_epochs * opt_steps


def _is_pretrained(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() not in {"false", "none", "null", "random", "no", "0"}
    return bool(value)


def _safe(label: str, warnings: list[str], fn: Any) -> Any:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - exercised by real pipeline failures
        warnings.append(f"{label} failed: {exc}")
        return None


def _tracker_run_id(tracker: Any | None) -> str:
    if tracker is None:
        return ""
    resolver = getattr(tracker, "_resolve_run_id", None)
    if resolver is None:
        return ""
    return str(resolver())


def _audit_params(cfg: Mapping[str, Any], manifest_hash: str | None) -> dict[str, Any]:
    """Translate the Hydra cfg into the flat audit allowlist.

    ``bat_tracking.filter_params`` keeps an explicit, curated set of flat keys
    (``model_family``, ``lr``, ``species`` …) plus the ``loss_params.``,
    ``lr_schedule_params.``, ``best_`` and ``test/`` prefixes. The Hydra cfg
    arrives nested under ``model.``, ``loss.``, ``trainer.``, ``data.``; walking
    it with dotted keys produces names the audit drops outright. We pull the
    audit-relevant knobs out by hand so MLflow runs surface the parameters the
    plan actually selected.
    """

    model = _mapping(cfg.get("model"))
    loss = _mapping(cfg.get("loss"))
    trainer = _mapping(cfg.get("trainer"))
    data = _mapping(cfg.get("data"))
    early_stop = _mapping(trainer.get("early_stop"))
    lr_schedule_params = _mapping(trainer.get("lr_schedule_params"))
    loss_params = _mapping(loss.get("params"))

    params: dict[str, Any] = {}

    def _put(key: str, value: Any) -> None:
        if value is not None:
            params[key] = value

    _put("model_family", model.get("family") or loss.get("family"))
    _put("model_arch", model.get("arch"))
    _put("embedding_dim", model.get("embedding_dim"))
    _put("loss_type", loss.get("type"))
    _put("optimizer", trainer.get("optimizer"))
    _put("lr", trainer.get("lr"))
    _put("weight_decay", trainer.get("weight_decay"))
    _put("ema_decay", trainer.get("ema_decay"))
    _put("gradient_accumulation_steps", trainer.get("gradient_accumulation_steps"))
    _put("batch_size", trainer.get("batch_size", data.get("batch_size")))
    _put("epochs", trainer.get("epochs"))
    _put("early_stop_patience", early_stop.get("patience"))
    _put("early_stop_monitor", early_stop.get("monitor"))
    _put("split_mode", data.get("split_mode"))
    _put("split_seed", data.get("split_seed"))
    _put("val_fraction", data.get("val_fraction"))
    _put("test_fraction", data.get("test_fraction"))
    _put("species", data.get("species"))
    _put("background", data.get("background"))
    _put("data_source", data.get("source"))
    _put("augmentation_preset", data.get("augmentation_preset"))
    _put("manifest_hash", manifest_hash)

    # Loss-family sub-params (e.g. focal alpha/gamma, arcface margin/scale,
    # adaface h). The audit allows the ``loss_params.`` prefix to pass
    # through; we also accept the legacy practice of putting these directly
    # on the loss config (`cfg.loss.alpha`) by mirroring known knobs.
    known_loss_keys = ("margin", "scale", "alpha", "gamma", "sub_centers", "h")
    for key in known_loss_keys:
        value = loss_params.get(key, loss.get(key))
        if value is not None:
            params[f"loss_params.{key}"] = value

    # LR schedule sub-params (warmup, T_max, min_lr, ...). Allow both
    # ``cfg.trainer.lr_schedule_params.*`` and the looser
    # ``cfg.trainer.warmup_epochs`` style.
    for key, value in lr_schedule_params.items():
        params[f"lr_schedule_params.{key}"] = value
    if "warmup_epochs" in trainer and "lr_schedule_params.warmup_epochs" not in params:
        params["lr_schedule_params.warmup_epochs"] = trainer["warmup_epochs"]

    return params


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _data_value(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    return _mapping(cfg.get("data")).get(key, default)


def _trainer_value(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    return _mapping(cfg.get("trainer")).get(key, default)
