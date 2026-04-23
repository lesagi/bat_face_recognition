#!/usr/bin/env python3
"""
Compute ROC-AUC for an existing Siamese training run, identified by MLflow run id
or by on-disk run directory.

This script is a thin wrapper around
`app.siamese_training.auc_evaluator.evaluate_auc_for_run`. It is useful when you
already have a completed training run (the evaluation_v*.csv was produced) but
AUC was never computed/logged at the time.

Usage:
    # By MLflow run id (looks up the `model_output_dir` tag on the run):
    python -m app.compute_auc_from_run --mlflow-run-id <run_id>

    # By run directory directly (no MLflow lookup required):
    python -m app.compute_auc_from_run --run-dir /path/to/.../<experiment_dir>

    # Do not write AUC back to the MLflow run (disk outputs only):
    python -m app.compute_auc_from_run --mlflow-run-id <run_id> --no-mlflow-log
"""

import argparse
import os
import sys
from typing import Optional


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)


def _resolve_tracking_uri(uri: str) -> str:
    """Match `SiameseNetworkTrainer._resolve_mlflow_tracking_uri` without TF."""
    if uri.startswith("file:///"):
        return uri
    if uri.startswith("file:./"):
        uri = uri[5:]
    if uri.startswith("./") or (
        not uri.startswith("/") and not uri.startswith("file://")
    ):
        relative = uri[2:] if uri.startswith("./") else uri
        project_root = os.path.abspath(os.path.join(_THIS_DIR, ".."))
        return f"file://{os.path.join(project_root, relative)}"
    if not uri.startswith("file://"):
        return f"file://{uri}"
    return uri


def _run_dir_from_mlflow_run_id(
    run_id: str, tracking_uri: Optional[str]
) -> str:
    """Look up the `model_output_dir` tag on the given MLflow run."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as e:
        raise ImportError(
            "MLflow is required to resolve a run id. Install with: pip install mlflow"
        ) from e

    if tracking_uri is None:
        try:
            from config.loader import load_config

            tracking_uri = load_config().mlflow.tracking_uri
        except Exception:
            tracking_uri = "file:./mlruns"

    resolved = _resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(resolved)
    client = MlflowClient(resolved)
    run = client.get_run(run_id)

    run_dir = run.data.tags.get("model_output_dir")
    if not run_dir:
        raise ValueError(
            f"MLflow run {run_id} is missing the 'model_output_dir' tag; "
            "cannot locate the evaluation CSV automatically. "
            "Re-run with --run-dir pointing at the run folder."
        )

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(
            f"model_output_dir tag pointed at a non-existent path: {run_dir}"
        )

    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute ROC-AUC for a Siamese training run from its saved "
            "evaluation CSV. Writes roc_curve_v*.png + updates training_summary.json, "
            "and optionally logs test_roc_auc back to MLflow."
        ),
    )
    parser.add_argument(
        "--mlflow-run-id",
        default=None,
        help="MLflow run id to resolve the run directory from (via model_output_dir tag).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Explicit run directory (alternative to --mlflow-run-id). "
            "If both are given, --run-dir wins and MLflow logging still targets --mlflow-run-id."
        ),
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help=(
            "Explicit evaluation_v*.csv path. If omitted, the newest matching "
            "CSV in --run-dir is used."
        ),
    )
    parser.add_argument(
        "--no-mlflow-log",
        action="store_true",
        help="Do not write test_roc_auc / ROC PNG back to any MLflow run (default: write).",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Override MLflow tracking URI (default: read from app config).",
    )

    args = parser.parse_args()

    if not args.mlflow_run_id and not args.run_dir:
        parser.error("one of --mlflow-run-id or --run-dir is required")

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _run_dir_from_mlflow_run_id(args.mlflow_run_id, args.tracking_uri)
        print(f"Resolved run directory from MLflow: {run_dir}")
    elif args.tracking_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(_resolve_tracking_uri(args.tracking_uri))
        except ImportError:
            pass

    # Load auc_evaluator via importlib so we bypass siamese_training/__init__.py
    # (which eagerly imports trainer -> tensorflow and would fail in non-TF envs).
    import importlib.util

    _auc_path = os.path.join(_THIS_DIR, "siamese_training", "auc_evaluator.py")
    _spec = importlib.util.spec_from_file_location(
        "_bfr_auc_evaluator", _auc_path
    )
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Could not load auc_evaluator module from {_auc_path}")
    _auc_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_auc_mod)
    evaluate_auc_for_run = _auc_mod.evaluate_auc_for_run

    result = evaluate_auc_for_run(
        run_dir=run_dir,
        csv_path=args.csv_path,
        mlflow_run_id=args.mlflow_run_id,
        mlflow_log=not args.no_mlflow_log,
    )

    auc = result.get("roc_auc")
    print("")
    print("--- Summary ---")
    print(f"  Run dir:     {run_dir}")
    print(f"  CSV:         {result['csv_path']}")
    print(f"  ROC PNG:     {result['roc_png_path']}")
    if auc is None:
        print("  ROC-AUC:     N/A (single-class labels in CSV)")
    else:
        print(f"  ROC-AUC:     {auc:.4f}")
    print(f"  n_positive:  {result['n_positive']}")
    print(f"  n_negative:  {result['n_negative']}")
    print(f"  n_total:     {result['n_total']}")
    if result.get("summary_path"):
        print(f"  Summary JSON:{result['summary_path']}")
    if result.get("mlflow_status"):
        print(f"  MLflow:      {result['mlflow_status']}")


if __name__ == "__main__":
    main()
