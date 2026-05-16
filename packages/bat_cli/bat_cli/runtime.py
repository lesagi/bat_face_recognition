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
    image_size = int(_mapping(cfg_dict.get("model")).get("input_edge_length", 224))
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

        train_ds = BatDataset(manifest, split="train", image_size=image_size)
        val_ds = _optional_embedding_dataset(manifest, "val", image_size=image_size)
        test_ds = _optional_embedding_dataset(manifest, "test", image_size=image_size)
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
        trainer_kwargs = {"eval_manifest": manifest, "eval_split": "test"}

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
    explanations_projection_cap: int = 30,
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
            tags={"entrypoint": "bat-cli"},
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

        eval_report = _safe("test evaluation", warnings, lambda: _run_test(trainer, bundle))

        explanations: ExplanationArtifacts | None = None
        explanations_dir: Path | None = None
        if run_explanations:
            explanations_dir = resolved_output / "explanations"
            explanation_image_size = int(
                _mapping(bundle.cfg.get("model")).get("input_edge_length", 224)
            )
            explanations = _safe(
                "explanations",
                warnings,
                lambda: _build_explanations(
                    bundle.model,
                    bundle.manifest,
                    output_dir=explanations_dir,
                    samples_per_split=explanations_per_split,
                    projection_cap=explanations_projection_cap,
                    image_size=explanation_image_size,
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

        permutation_dir: Path | None = None
        if run_permutation and eval_report is not None and eval_report.predictions is not None:
            permutation_result = _safe(
                "permutation test",
                warnings,
                lambda: run_permutation_test(
                    bundle.cfg,
                    predictions=eval_report.predictions,
                    root=repo,
                    output_dir=resolved_output / "permutation",
                    n_permutations=permutation_n,
                    # Use the same Youden-J threshold the eval used. Without
                    # this, embedding-model cosine scores (typically all
                    # above 0.5 even for non-match pairs) collapse to
                    # all-positive predictions and a constant null
                    # distribution — masking real ROC-AUC signal.
                    threshold=eval_report.verification.optimal_threshold,
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
    return _run_test(trainer, bundle)


def _build_explanations(
    model: Any,
    manifest: Manifest,
    *,
    output_dir: Path,
    samples_per_split: int = 2,
    projection_cap: int = 30,
    image_size: int,
) -> ExplanationArtifacts:
    """Generate saliency + projection artifacts spanning train/val/test splits.

    Picks ``samples_per_split`` distinct identities per split (deterministic:
    sorted by record path), takes one record each, runs the family-appropriate
    saliency adapter on the union, then runs the embedding projection adapter
    on a wider pool capped at ``projection_cap``.

    ``image_size`` is the edge length the model was trained at (Siamese: 105,
    embedding ResNet backbones: 224). Both adapters resize inputs to that
    edge before invoking the model, since their dataclass defaults
    (``EmbeddingProjectionAdapter.input_size=224``) silently mismatch a
    pair-family Siamese head and cascade-fail the entire explanation block.

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

    projection_samples = _sample_projection_records(manifest, projection_cap)
    projection_adapter = EmbeddingProjectionAdapter(
        method="both", output_dir=output_dir, input_size=image_size
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


def _sample_projection_records(manifest: Manifest, cap: int) -> list[ImageRecord]:
    """Pool records across splits for the t-SNE/UMAP projection."""

    pool: list[ImageRecord] = []
    for split in ("train", "val", "test"):
        pool.extend(manifest.filter_split(split))  # type: ignore[arg-type]
    if not pool:
        return []
    if len(pool) <= cap:
        return pool
    # Spread evenly: stride sample to keep identity diversity.
    stride = max(1, len(pool) // cap)
    return list(pool[::stride])[:cap]


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


def _run_test(trainer: Any, bundle: TrainingBundle) -> EvalReport:
    if _model_family(bundle.cfg) == "embedding":
        return trainer.test(manifest=bundle.manifest, split="test")
    if bundle.test_loader is None:
        raise CliRuntimeError("test split is unavailable for pair evaluation")
    return trainer.test(bundle.test_loader)


def _optional_embedding_dataset(manifest: Manifest, split: str, *, image_size: int) -> Any | None:
    from bat_data import BatDataset

    try:
        return BatDataset(manifest, split=split, image_size=image_size)
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
