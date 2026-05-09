"""Command-line entry point for the PyTorch refactor workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from bat_cli.runtime import (
    CliRuntimeError,
    build_manifest_csv,
    compose_config,
    default_output_dir,
    eval_report_to_dict,
    find_project_root,
    format_json,
    run_evaluation,
    run_training,
)


def _overrides(_: click.Context, __: click.Parameter, values: tuple[str, ...]) -> tuple[str, ...]:
    for value in values:
        if "=" not in value:
            raise click.BadParameter(f"Hydra override must look like key=value; got {value!r}")
    return values


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
@click.option("--output-dir", type=click.Path(path_type=Path), help="Training output directory.")
@click.option("--mlflow-experiment", default="bat-face-recognition", show_default=True)
@click.option("--tracking-uri", help="Optional MLflow tracking URI.")
@click.option("--run-name", help="Optional MLflow run name.")
@click.option("--no-mlflow", is_flag=True, help="Run locally without starting an MLflow run.")
@click.option("--promote", is_flag=True, help="Attempt champion promotion after test evaluation.")
@click.option("--dry-run", is_flag=True, help="Compose and print the run plan without training.")
def train(
    config_name: str,
    experiment: str | None,
    overrides: tuple[str, ...],
    output_dir: Path | None,
    mlflow_experiment: str,
    tracking_uri: str | None,
    run_name: str | None,
    no_mlflow: bool,
    promote: bool,
    dry_run: bool,
) -> None:
    """Run training followed by test evaluation and unified PDF generation."""

    try:
        root = find_project_root()
        cfg = compose_config(config_name=config_name, experiment=experiment, overrides=overrides)
        resolved_output = output_dir or default_output_dir(cfg, root=root)
        if dry_run:
            click.echo("Dry run OK")
            click.echo(format_json(_plan_payload(cfg, resolved_output, use_mlflow=not no_mlflow)))
            return
        result = run_training(
            cfg,
            root=root,
            output_dir=resolved_output,
            mlflow_experiment=mlflow_experiment,
            tracking_uri=tracking_uri,
            run_name=run_name,
            use_mlflow=not no_mlflow,
            promote=promote,
        )
    except CliRuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Training complete: {result.output_dir}")
    if result.report_path is not None:
        click.echo(f"Report: {result.report_path}")
    if result.run_id:
        click.echo(f"MLflow run: {result.run_id}")
    for warning in result.warnings:
        click.echo(f"Warning: {warning}", err=True)


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


@main.command()
@click.argument("run_id_a")
@click.argument("run_id_b")
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
def compare(run_id_a: str, run_id_b: str, output_path: Path) -> None:
    """Render a side-by-side MLflow run comparison HTML report."""

    from bat_reporting import compare_runs

    path = compare_runs(run_id_a, run_id_b, output_path)
    click.echo(f"Comparison report: {path}")


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


def _interactive(ctx: click.Context) -> None:
    if not click.get_text_stream("stdin").isatty():
        click.echo(ctx.get_help())
        return

    from InquirerPy import inquirer

    action = inquirer.select(
        message="Choose an action",
        choices=["train", "sweep", "evaluate", "compare", "build-manifest", "promote"],
    ).execute()

    if action in {"train", "sweep"}:
        experiment = inquirer.select(
            message="Experiment",
            choices=["arcface_rousettus_random_bg_video", "siamese_rousettus_random_bg_video"],
        ).execute()
        dry_run = inquirer.confirm(message="Dry run only?", default=False).execute()
        if action == "train":
            ctx.invoke(
                train,
                config_name="config",
                experiment=experiment,
                overrides=(),
                output_dir=None,
                mlflow_experiment="bat-face-recognition",
                tracking_uri=None,
                run_name=None,
                no_mlflow=False,
                promote=False,
                dry_run=dry_run,
            )
        else:
            ctx.invoke(
                sweep,
                config_name="config",
                experiment=experiment,
                sweep_config="arcface",
                overrides=(),
                n_jobs=1,
                mlflow_experiment="bat-face-recognition-sweeps",
                tracking_uri=None,
                dry_run=dry_run,
            )
        return

    click.echo(f"Run `bat-cli {action} --help` for the required arguments.")


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
