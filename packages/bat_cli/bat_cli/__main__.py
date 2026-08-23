"""Command-line entry point for the PyTorch refactor workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from bat_cli.runtime import (
    KEEP_CHECKPOINTS_CHOICES,
    CliRuntimeError,
    build_manifest_csv,
    compose_config,
    default_output_dir,
    eval_report_to_dict,
    find_project_root,
    format_json,
    run_evaluation,
    run_permutation_test,
    run_training,
    run_video_extraction,
)


def _overrides(_: click.Context, __: click.Parameter, values: tuple[str, ...]) -> tuple[str, ...]:
    for value in values:
        if "=" not in value:
            raise click.BadParameter(f"Hydra override must look like key=value; got {value!r}")
    return values


def _run_tags(_: click.Context, __: click.Parameter, values: tuple[str, ...]) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` MLflow run tags into a dict."""
    tags: dict[str, str] = {}
    for value in values:
        key, sep, val = value.partition("=")
        if not sep or not key:
            raise click.BadParameter(f"run tag must look like key=value; got {value!r}")
        tags[key.strip()] = val.strip()
    return tags


def _apply_fold_overrides(overrides: tuple[str, ...], fold: int | None) -> tuple[str, ...]:
    """Expand ``--fold N`` into the Hydra overrides that enable a re-split.

    The fold index drives ``seed`` as well as the partition: one knob varying
    both the held-out identities and the training seed means each fold is a
    single independent draw, which is what the sensitivity analysis assumes.
    An explicit ``--hydra`` override of any of these keys wins, so a caller can
    still pin the training seed while varying only the split.
    """
    if fold is None:
        return overrides

    if fold < 0:
        raise click.BadParameter("--fold must be >= 0")

    explicit = {o.split("=", 1)[0] for o in overrides}
    defaults = {
        "data.resplit": "true",
        "data.fold_id": str(fold),
        "data.split_size_mode": "seeded",
        "seed": str(fold),
        "trainer.deterministic": "true",
    }
    extra = tuple(f"{k}={v}" for k, v in defaults.items() if k not in explicit)
    return overrides + extra


def _require_gpu(allow_cpu: bool) -> None:
    """Log the training device and fail fast on an accidental CPU fallback.

    ``accelerate`` silently places training on CPU when CUDA is unavailable, which
    would make a multi-day batch crawl unnoticed. Refuse unless ``--allow-cpu`` is
    set. Respects ``CUDA_VISIBLE_DEVICES`` (torch reports only visible devices)."""
    import torch

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
        click.echo(f"GPU: training on CUDA — {n} device(s) visible: {names}")
        return
    if allow_cpu:
        click.echo("GPU: CUDA unavailable — training on CPU (--allow-cpu set).", err=True)
        return
    raise CliRuntimeError(
        "CUDA is not available — refusing to train on CPU (a full run would be "
        "prohibitively slow). Pass --allow-cpu to override, or fix the CUDA/driver "
        "setup / CUDA_VISIBLE_DEVICES."
    )


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Train, sweep, evaluate, compare, and manage bat face-recognition runs."""

    if ctx.invoked_subcommand is None:
        _interactive(ctx)


@main.command()
@click.option("--config-name", default="config", show_default=True)
@click.option("--experiment", help="Hydra experiment config from configs/experiment.")
@click.option(
    "--hydra",
    "overrides",
    multiple=True,
    callback=_overrides,
    help="Hydra override, e.g. trainer.epochs=5. May be passed multiple times.",
)
@click.option(
    "--fold",
    type=int,
    default=None,
    help="Re-split the manifest in memory for fold N (sets data.resplit=true, "
    "data.fold_id=N, data.split_size_mode=seeded). The fold seed drives both the "
    "identity partition and the training seed, so one knob varies everything.",
)
@click.option("--output-dir", type=click.Path(path_type=Path), help="Training output directory.")
@click.option("--mlflow-experiment", default="bat-face-recognition", show_default=True)
@click.option("--tracking-uri", help="Optional MLflow tracking URI.")
@click.option("--run-name", help="Optional MLflow run name.")
@click.option(
    "--run-tag",
    "run_tags",
    multiple=True,
    callback=_run_tags,
    help="MLflow run tag as key=value (e.g. batch=rousettus-2026-07-05). May be passed multiple times.",
)
@click.option("--no-mlflow", is_flag=True, help="Run locally without starting an MLflow run.")
@click.option(
    "--promote",
    is_flag=True,
    help="Auto-promote silently if criterion beats incumbent. Mutually exclusive with --prompt-promote.",
)
@click.option(
    "--prompt-promote",
    is_flag=True,
    help="If criterion beats incumbent, prompt once before promoting. No prompt = no promotion.",
)
@click.option(
    "--promotion-criterion",
    default="test/roc_auc",
    show_default=True,
    help="Metric used by --promote / --prompt-promote.",
)
@click.option(
    "--no-permutation",
    is_flag=True,
    help="Skip the auto inference-mode permutation test step.",
)
@click.option(
    "--permutation-n",
    default=1000,
    show_default=True,
    type=int,
    help="Number of permutations for the auto post-training permutation test.",
)
@click.option(
    "--no-explanations",
    is_flag=True,
    help="Skip the auto saliency / GradCAM / projection step.",
)
@click.option(
    "--explanations-per-split",
    default=2,
    show_default=True,
    type=int,
    help="Identities sampled per split (train/val/test) for the saliency composite.",
)
@click.option(
    "--keep-checkpoints",
    type=click.Choice(KEEP_CHECKPOINTS_CHOICES),
    default="all",
    show_default=True,
    help="Checkpoint retention after the pipeline runs: all | roc_auc | f1 | none. "
    "Batch runs use 'none' (models are re-derivable from the recorded seed + config).",
)
@click.option(
    "--allow-cpu",
    is_flag=True,
    help="Permit training on CPU when CUDA is unavailable (otherwise the run fails fast).",
)
@click.option("--dry-run", is_flag=True, help="Compose and print the run plan without training.")
def train(
    config_name: str,
    experiment: str | None,
    overrides: tuple[str, ...],
    fold: int | None,
    output_dir: Path | None,
    mlflow_experiment: str,
    tracking_uri: str | None,
    run_name: str | None,
    run_tags: dict[str, str],
    no_mlflow: bool,
    promote: bool,
    prompt_promote: bool,
    promotion_criterion: str,
    no_permutation: bool,
    permutation_n: int,
    no_explanations: bool,
    explanations_per_split: int,
    keep_checkpoints: str,
    allow_cpu: bool,
    dry_run: bool,
) -> None:
    """Run training followed by test evaluation, PDF generation, and permutation test."""

    try:
        if promote and prompt_promote:
            raise CliRuntimeError("--promote and --prompt-promote are mutually exclusive")
        root = find_project_root()
        overrides = _apply_fold_overrides(overrides, fold)
        cfg = compose_config(config_name=config_name, experiment=experiment, overrides=overrides)
        resolved_output = output_dir or default_output_dir(cfg, root=root)
        if dry_run:
            click.echo("Dry run OK")
            click.echo(format_json(_plan_payload(cfg, resolved_output, use_mlflow=not no_mlflow)))
            return
        _require_gpu(allow_cpu)
        result = run_training(
            cfg,
            root=root,
            output_dir=resolved_output,
            mlflow_experiment=mlflow_experiment,
            tracking_uri=tracking_uri,
            run_name=run_name,
            run_tags=run_tags,
            use_mlflow=not no_mlflow,
            promote=promote,
            prompt_promote=prompt_promote,
            promotion_criterion=promotion_criterion,
            run_permutation=not no_permutation,
            permutation_n=permutation_n,
            run_explanations=not no_explanations,
            explanations_per_split=explanations_per_split,
            keep_checkpoints=keep_checkpoints,
        )
    except CliRuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Training complete: {result.output_dir}")
    if result.report_path is not None:
        click.echo(f"Report: {result.report_path}")
    if result.explanations_dir is not None:
        click.echo(f"Explanations: {result.explanations_dir}")
    if result.permutation_dir is not None:
        click.echo(f"Permutation: {result.permutation_dir}")
    if result.promotion is not None:
        click.echo(f"Promotion: {_format_promotion(result.promotion)}")
    if result.run_id:
        click.echo(f"MLflow run: {result.run_id}")
    for warning in result.warnings:
        click.echo(f"Warning: {warning}", err=True)


def _format_promotion(decision: Any) -> str:
    if decision.mode == "off":
        return "skipped (no flag)"
    new = "?" if decision.candidate_metric is None else f"{decision.candidate_metric:.4f}"
    inc = "none" if decision.incumbent_metric is None else f"{decision.incumbent_metric:.4f}"
    if not decision.beats:
        return f"skipped (candidate={new}, incumbent={inc})"
    if decision.declined:
        return f"declined (candidate={new}, incumbent={inc})"
    return (
        f"promoted (candidate={new}, incumbent={inc})"
        if decision.promoted
        else f"failed (candidate={new}, incumbent={inc})"
    )


@main.command()
@click.option("--config-name", default="config", show_default=True)
@click.option("--experiment", default="arcface_rousettus_random_bg_video", show_default=True)
@click.option(
    "--sweep-config", default="arcface", show_default=True, help="Config in configs/sweep."
)
@click.option(
    "--hydra",
    "overrides",
    multiple=True,
    callback=_overrides,
    help="Hydra override. May be passed multiple times.",
)
@click.option("--n-jobs", default=1, show_default=True, type=int)
@click.option("--mlflow-experiment", default="bat-face-recognition-sweeps", show_default=True)
@click.option("--tracking-uri", help="Optional MLflow tracking URI.")
@click.option("--dry-run", is_flag=True, help="Compose and print the sweep plan without running.")
def sweep(
    config_name: str,
    experiment: str,
    sweep_config: str,
    overrides: tuple[str, ...],
    n_jobs: int,
    mlflow_experiment: str,
    tracking_uri: str | None,
    dry_run: bool,
) -> None:
    """Run an Optuna sweep on the validation split."""

    try:
        root = find_project_root()
        cfg = compose_config(
            config_name=config_name,
            experiment=experiment,
            overrides=(f"+sweep={sweep_config}", *overrides),
        )
        sweep_cfg = dict(cfg.get("sweep", {}))
        sweep_cfg["base_cfg"] = {k: v for k, v in cfg.items() if k != "sweep"}
        if dry_run:
            click.echo("Dry run OK")
            click.echo(format_json(sweep_cfg))
            return

        from bat_cli.runtime import build_bundle
        from bat_sweeps import TrialComponents, run_sweep
        from bat_tracking import start_run

        def _build_components(trial_cfg: dict[str, Any]) -> TrialComponents:
            bundle = build_bundle(trial_cfg, root=root)
            return TrialComponents(
                model=bundle.model,
                loss=bundle.loss,
                trainer_cfg=bundle.trainer_cfg,
                train_loader=bundle.train_loader,
                val_loader=bundle.val_loader,
                tracker=None,
                trainer_kwargs=bundle.trainer_kwargs,
            )

        with start_run(
            experiment_name=mlflow_experiment,
            run_name=str(sweep_cfg.get("study_name", "bat_sweep")),
            tags={"entrypoint": "bat-cli sweep"},
            tracking_uri=tracking_uri,
        ) as tracker:
            study = run_sweep(
                sweep_cfg,
                _build_components,
                champion_tracker=tracker,
                n_jobs=n_jobs,
            )
    except CliRuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Sweep complete: {study.study_name} best={study.best_value:.6f} "
        f"trial={study.best_trial.number}"
    )


@main.command()
@click.option("--config-name", default="config", show_default=True)
@click.option("--experiment", help="Hydra experiment config from configs/experiment.")
@click.option(
    "--hydra",
    "overrides",
    multiple=True,
    callback=_overrides,
    help="Hydra override. May be passed multiple times.",
)
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", type=click.Path(path_type=Path))
def evaluate(
    config_name: str,
    experiment: str | None,
    overrides: tuple[str, ...],
    checkpoint: Path | None,
    output_dir: Path | None,
) -> None:
    """Evaluate a configured run on the test split."""

    try:
        cfg = compose_config(config_name=config_name, experiment=experiment, overrides=overrides)
        report = run_evaluation(cfg, checkpoint=checkpoint, output_dir=output_dir)
    except CliRuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(format_json(eval_report_to_dict(report)))


@main.command("permutation-test")
@click.option("--config-name", default="config", show_default=True)
@click.option("--experiment", help="Hydra experiment config from configs/experiment.")
@click.option(
    "--hydra",
    "overrides",
    multiple=True,
    callback=_overrides,
    help="Hydra override. May be passed multiple times.",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    help="Optional checkpoint to restore before running test eval.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Where plots / CSV / JSON go. Defaults to <run_output>/permutation.",
)
@click.option(
    "--n-permutations",
    default=1000,
    show_default=True,
    type=int,
)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--threshold", default=0.5, show_default=True, type=float)
@click.option(
    "--degradation",
    is_flag=True,
    help="Also compute the degradation curve (21 fractions, 5%% increments).",
)
@click.option("--verbose", is_flag=True)
def permutation_test(
    config_name: str,
    experiment: str | None,
    overrides: tuple[str, ...],
    checkpoint: Path | None,
    output_dir: Path | None,
    n_permutations: int,
    seed: int,
    threshold: float,
    degradation: bool,
    verbose: bool,
) -> None:
    """Run the inference-mode permutation test for a configured run."""

    try:
        cfg = compose_config(config_name=config_name, experiment=experiment, overrides=overrides)
        result = run_permutation_test(
            cfg,
            checkpoint=checkpoint,
            output_dir=output_dir,
            n_permutations=n_permutations,
            seed=seed,
            threshold=threshold,
            degradation=degradation,
            verbose=verbose,
        )
    except CliRuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Permutation report: {result.output_dir}")
    click.echo(format_json({"n_permutations": result.n_permutations, "p_values": result.p_values}))


@main.command("build-manifest")
@click.option(
    "--input-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True
)
@click.option("--output", "output_csv", type=click.Path(path_type=Path), required=True)
@click.option("--species", type=click.Choice(["mauritius", "rousettus"]), required=True)
@click.option("--val-fraction", default=0.15, show_default=True, type=float)
@click.option("--test-fraction", default=0.15, show_default=True, type=float)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--keep-unparseable", is_flag=True, help="Fail on unparseable filenames.")
def build_manifest(
    input_dir: Path,
    output_csv: Path,
    species: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    keep_unparseable: bool,
) -> None:
    """Build a split manifest CSV from an image directory."""

    try:
        path, counts = build_manifest_csv(
            input_dir=input_dir,
            output_csv=output_csv,
            species=species,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            skip_unparseable=not keep_unparseable,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Wrote {path} "
        f"(train={counts.train}, val={counts.val}, test={counts.test}, total={counts.total})"
    )


@main.command("extract-videos")
@click.option(
    "--videos-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory of raw videos (one video = one identity).",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    required=True,
    help="Root for per-variant output folders (original_bg/green_bg/random_bg/face_ellipse).",
)
@click.option(
    "--species",
    type=click.Choice(["mauritius", "rousettus"]),
    default="mauritius",
    show_default=True,
)
@click.option("--n-videos", default=20, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option(
    "--seg-weights",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="models/preprocessing/face_seg_mauritius_v2.pt",
    show_default=True,
    help="YOLO segmentation weights (retrained mauritius face seg, class 'face').",
)
@click.option(
    "--detector-weights",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="legacy/face_detection/chosen_model/best.pt",
    show_default=True,
    help="YOLO detect weights — used as a face-presence gate before segmenting.",
)
@click.option(
    "--gate/--no-gate",
    default=True,
    show_default=True,
    help="Only process frames where the detector finds a face (presence gate).",
)
@click.option(
    "--pose-weights",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="legacy/face_annotation_eyes_nose/runs/pose/augmented_train/weights/best.pt",
    show_default=True,
    help="YOLO pose weights (eyes+nose), used to straighten the face.",
)
@click.option(
    "--min-keypoint-confidence",
    default=0.35,
    show_default=True,
    type=float,
    help="Require both eyes above this confidence (fixes garbage-angle rotations).",
)
@click.option(
    "--device", default="cuda", show_default=True, help="Torch device for YOLO (cuda/cpu/0)."
)
@click.option(
    "--target-frames",
    default=53,
    show_default=True,
    type=int,
    help="Kept frames per video (~53 × 20 ≈ rousettus count). Spread across the clip.",
)
@click.option("--edge-length", default=224, show_default=True, type=int)
@click.option(
    "--margin-ratio", default=0.03, show_default=True, type=float, help="Margin around mask (3%)."
)
@click.option(
    "--variant",
    "variants",
    multiple=True,
    default=("original_bg", "green_bg", "random_bg", "face_ellipse"),
    show_default=True,
    help="Variant to emit (repeatable).",
)
@click.option(
    "--random-bg-style",
    default="picsum",
    show_default=True,
    help="random_bg source: 'picsum' (natural, pooled) or a generator key like 'blur'.",
)
@click.option("--min-quality", default=0.0, show_default=True, type=float)
@click.option(
    "--build-manifests/--no-build-manifests",
    default=False,
    show_default=True,
    help="Also build one split manifest CSV per variant directory.",
)
@click.option("--val-fraction", default=0.15, show_default=True, type=float)
@click.option("--test-fraction", default=0.15, show_default=True, type=float)
def extract_videos(
    videos_dir: Path,
    output_root: Path,
    species: str,
    n_videos: int,
    seed: int,
    seg_weights: Path,
    detector_weights: Path,
    gate: bool,
    pose_weights: Path,
    min_keypoint_confidence: float,
    device: str,
    target_frames: int,
    edge_length: int,
    margin_ratio: float,
    variants: tuple[str, ...],
    random_bg_style: str,
    min_quality: float,
    build_manifests: bool,
    val_fraction: float,
    test_fraction: float,
) -> None:
    """Extract eye-anchored face crops + background variants from raw videos."""

    try:
        summary = run_video_extraction(
            videos_dir=videos_dir,
            output_root=output_root,
            species=species,
            n_videos=n_videos,
            seed=seed,
            seg_weights=seg_weights,
            detector_weights=detector_weights if gate else None,
            pose_weights=pose_weights,
            min_keypoint_confidence=min_keypoint_confidence,
            device=device,
            target_frames=target_frames,
            edge_length=edge_length,
            margin_ratio=margin_ratio,
            variants=list(variants),
            min_quality=min_quality,
            random_bg_style=random_bg_style,
            build_manifests=build_manifests,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            log=lambda msg: click.echo(msg),
        )
    except CliRuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    per_variant = summary.get("per_variant", {})
    total = sum(per_variant.values())
    click.echo(
        f"Extracted {total} images across {len(per_variant)} variants "
        f"from {summary['selection']['n_videos']} videos → {summary['output_root']}"
    )
    click.echo(format_json({"per_variant": per_variant}))


@main.command()
@click.argument("run_id_a")
@click.argument("run_id_b")
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
def compare(run_id_a: str, run_id_b: str, output_path: Path) -> None:
    """Render a side-by-side MLflow run comparison HTML report."""

    from bat_reporting import compare_runs

    path = compare_runs(run_id_a, run_id_b, output_path)
    click.echo(f"Comparison report: {path}")


@main.command("compare-champion")
@click.argument("candidate_run_id")
@click.option(
    "--model-name",
    required=True,
    help="Registered MLflow model name whose Production-stage version is the champion.",
)
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--tracking-uri", help="Optional MLflow tracking URI.")
def compare_champion(
    candidate_run_id: str,
    model_name: str,
    output_path: Path,
    tracking_uri: str | None,
) -> None:
    """Render a candidate-vs-current-champion HTML comparison report.

    Resolves the current Production-stage version of MODEL_NAME via the
    MLflow registry and compares its source run against CANDIDATE_RUN_ID
    (left column = champion, right column = candidate). Errors loudly if
    no champion is registered yet.
    """
    from bat_reporting import NoChampionError, compare_to_champion

    client = None
    if tracking_uri:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
    try:
        path = compare_to_champion(
            candidate_run_id=candidate_run_id,
            model_name=model_name,
            output_path=output_path,
            mlflow_client=client,
        )
    except NoChampionError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Champion comparison report: {path}")


@main.command()
@click.argument("run_id")
@click.option("--criterion", default="test/roc_auc", show_default=True)
@click.option("--mlflow-experiment", default="bat-face-recognition", show_default=True)
@click.option("--tracking-uri", help="Optional MLflow tracking URI.")
def promote(
    run_id: str,
    criterion: str,
    mlflow_experiment: str,
    tracking_uri: str | None,
) -> None:
    """Promote a registered MLflow model version if it beats the champion."""

    from bat_tracking import MLflowTracker

    tracker = MLflowTracker(mlflow_experiment, tracking_uri=tracking_uri)
    promoted = tracker.promote_to_champion(run_id, criterion)
    click.echo("Promoted" if promoted else "No promotion performed")


_EXPERIMENT_CHOICES = (
    "arcface_rousettus_random_bg_video",
    "siamese_rousettus_random_bg_video",
)


def _interactive(ctx: click.Context) -> None:
    if not click.get_text_stream("stdin").isatty():
        click.echo(ctx.get_help())
        return

    from InquirerPy import inquirer

    action = inquirer.select(
        message="Choose an action",
        choices=[
            "train",
            "sweep",
            "evaluate",
            "permutation-test",
            "compare",
            "compare-champion",
            "promote",
            "build-manifest",
        ],
    ).execute()

    if action == "train":
        _interactive_train(ctx, inquirer)
    elif action == "sweep":
        _interactive_sweep(ctx, inquirer)
    elif action == "evaluate":
        _interactive_evaluate(ctx, inquirer)
    elif action == "permutation-test":
        _interactive_permutation_test(ctx, inquirer)
    elif action == "compare":
        _interactive_compare(ctx, inquirer)
    elif action == "compare-champion":
        _interactive_compare_champion(ctx, inquirer)
    elif action == "promote":
        _interactive_promote(ctx, inquirer)
    elif action == "build-manifest":
        _interactive_build_manifest(ctx, inquirer)


def _interactive_train(ctx: click.Context, inquirer: Any) -> None:
    experiment = inquirer.select(message="Experiment", choices=list(_EXPERIMENT_CHOICES)).execute()
    use_mlflow = inquirer.confirm(message="Log to MLflow?", default=True).execute()
    run_explanations = inquirer.confirm(
        message="Generate saliency + projection (samples train/val/test individuals)?",
        default=True,
    ).execute()
    run_permutation = inquirer.confirm(
        message="Run inference-mode permutation test after training?", default=True
    ).execute()
    promotion_mode = inquirer.select(
        message="Champion promotion",
        choices=[
            {"name": "off (default — never promote)", "value": "off"},
            {"name": "auto (promote silently if criterion improves)", "value": "auto"},
            {"name": "prompt (ask once if criterion improves)", "value": "prompt"},
        ],
        default="off",
    ).execute()
    dry_run = inquirer.confirm(message="Dry run only?", default=False).execute()

    ctx.invoke(
        train,
        config_name="config",
        experiment=experiment,
        overrides=(),
        output_dir=None,
        mlflow_experiment="bat-face-recognition",
        tracking_uri=None,
        run_name=None,
        no_mlflow=not use_mlflow,
        promote=promotion_mode == "auto",
        prompt_promote=promotion_mode == "prompt",
        promotion_criterion="test/roc_auc",
        no_permutation=not run_permutation,
        permutation_n=1000,
        no_explanations=not run_explanations,
        explanations_per_split=2,
        dry_run=dry_run,
    )


def _interactive_sweep(ctx: click.Context, inquirer: Any) -> None:
    experiment = inquirer.select(message="Experiment", choices=list(_EXPERIMENT_CHOICES)).execute()
    sweep_choice = inquirer.select(
        message="Sweep config", choices=["arcface"], default="arcface"
    ).execute()
    n_jobs = int(
        inquirer.text(
            message="Concurrent trials (n_jobs)",
            default="1",
            validate=lambda v: v.isdigit() and int(v) >= 1,
        ).execute()
    )
    dry_run = inquirer.confirm(message="Dry run only?", default=False).execute()

    ctx.invoke(
        sweep,
        config_name="config",
        experiment=experiment,
        sweep_config=sweep_choice,
        overrides=(),
        n_jobs=n_jobs,
        mlflow_experiment="bat-face-recognition-sweeps",
        tracking_uri=None,
        dry_run=dry_run,
    )


def _interactive_evaluate(ctx: click.Context, inquirer: Any) -> None:
    experiment = inquirer.select(message="Experiment", choices=list(_EXPERIMENT_CHOICES)).execute()
    checkpoint_str = inquirer.text(
        message="Checkpoint path (leave blank to evaluate fresh model)",
        default="",
    ).execute()
    checkpoint = Path(checkpoint_str) if checkpoint_str.strip() else None
    if checkpoint is not None and not checkpoint.exists():
        raise click.ClickException(f"checkpoint does not exist: {checkpoint}")

    ctx.invoke(
        evaluate,
        config_name="config",
        experiment=experiment,
        overrides=(),
        checkpoint=checkpoint,
        output_dir=None,
    )


def _interactive_permutation_test(ctx: click.Context, inquirer: Any) -> None:
    experiment = inquirer.select(message="Experiment", choices=list(_EXPERIMENT_CHOICES)).execute()
    checkpoint_str = inquirer.text(
        message="Checkpoint path (blank → recompute predictions from configured model)",
        default="",
    ).execute()
    n_perm = int(
        inquirer.text(
            message="Number of permutations",
            default="1000",
            validate=lambda v: v.isdigit() and int(v) >= 1,
        ).execute()
    )
    degradation = inquirer.confirm(
        message="Compute degradation curve (slower)?", default=False
    ).execute()
    seed = int(
        inquirer.text(
            message="Seed", default="42", validate=lambda v: v.lstrip("-").isdigit()
        ).execute()
    )
    checkpoint = Path(checkpoint_str) if checkpoint_str.strip() else None
    if checkpoint is not None and not checkpoint.exists():
        raise click.ClickException(f"checkpoint does not exist: {checkpoint}")

    ctx.invoke(
        permutation_test,
        config_name="config",
        experiment=experiment,
        overrides=(),
        checkpoint=checkpoint,
        output_dir=None,
        n_permutations=n_perm,
        seed=seed,
        threshold=0.5,
        degradation=degradation,
        verbose=False,
    )


def _interactive_compare(ctx: click.Context, inquirer: Any) -> None:
    run_id_a = inquirer.text(message="MLflow run ID A").execute()
    run_id_b = inquirer.text(message="MLflow run ID B").execute()
    output_str = inquirer.text(
        message="Output path", default="outputs/compare/report.html"
    ).execute()
    ctx.invoke(compare, run_id_a=run_id_a, run_id_b=run_id_b, output_path=Path(output_str))


def _interactive_compare_champion(ctx: click.Context, inquirer: Any) -> None:
    candidate_run_id = inquirer.text(message="Candidate MLflow run ID").execute()
    model_name = inquirer.text(
        message="Registered model name (Production-stage version becomes champion)"
    ).execute()
    output_str = inquirer.text(
        message="Output path", default="outputs/compare/champion.html"
    ).execute()
    ctx.invoke(
        compare_champion,
        candidate_run_id=candidate_run_id,
        model_name=model_name,
        output_path=Path(output_str),
        tracking_uri=None,
    )


def _interactive_promote(ctx: click.Context, inquirer: Any) -> None:
    run_id = inquirer.text(message="MLflow run ID to promote").execute()
    criterion = inquirer.text(
        message="Promotion criterion (metric key)", default="test/roc_auc"
    ).execute()
    mlflow_experiment = inquirer.text(
        message="MLflow experiment", default="bat-face-recognition"
    ).execute()
    ctx.invoke(
        promote,
        run_id=run_id,
        criterion=criterion,
        mlflow_experiment=mlflow_experiment,
        tracking_uri=None,
    )


def _interactive_build_manifest(ctx: click.Context, inquirer: Any) -> None:
    input_dir = Path(inquirer.text(message="Input directory (with images)").execute())
    if not input_dir.exists():
        raise click.ClickException(f"input directory does not exist: {input_dir}")
    output_str = inquirer.text(
        message="Output CSV path", default="data/manifests/manifest.csv"
    ).execute()
    species = inquirer.select(message="Species", choices=["mauritius", "rousettus"]).execute()
    val_fraction = float(
        inquirer.text(
            message="val_fraction",
            default="0.15",
            validate=lambda v: _is_float_in_unit_interval(v),
        ).execute()
    )
    test_fraction = float(
        inquirer.text(
            message="test_fraction",
            default="0.15",
            validate=lambda v: _is_float_in_unit_interval(v),
        ).execute()
    )
    seed = int(
        inquirer.text(
            message="Split seed", default="42", validate=lambda v: v.lstrip("-").isdigit()
        ).execute()
    )
    keep_unparseable = inquirer.confirm(
        message="Fail on unparseable filenames?", default=False
    ).execute()
    ctx.invoke(
        build_manifest,
        input_dir=input_dir,
        output_csv=Path(output_str),
        species=species,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        keep_unparseable=keep_unparseable,
    )


def _is_float_in_unit_interval(value: str) -> bool:
    try:
        v = float(value)
    except ValueError:
        return False
    return 0.0 <= v < 1.0


def _plan_payload(cfg: dict[str, Any], output_dir: Path, *, use_mlflow: bool) -> dict[str, Any]:
    return {
        "model": cfg.get("model"),
        "loss": cfg.get("loss"),
        "trainer": cfg.get("trainer"),
        "data": cfg.get("data"),
        "output_dir": str(output_dir),
        "mlflow": use_mlflow,
    }


if __name__ == "__main__":
    main()
