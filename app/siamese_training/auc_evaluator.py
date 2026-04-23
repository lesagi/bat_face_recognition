"""
AUC evaluator for Siamese network predictions.

Consumes the `evaluation_v*.csv` produced by `app/generate_predictions.py`
(`y_true`, `y_hat` columns) and emits:

- A `roc_curve_v*.png` plot next to the evaluation CSV.
- An `auc` block merged into `training_summary.json` in the run directory.
- Opportunistic MLflow logging:
    * If an MLflow run is active (trainer path), log to the active run.
    * Else if an explicit `mlflow_run_id` is provided, use
      `MlflowClient` to log to that specific run.
    * Else skip with an info message.

No TensorFlow import; operates purely on the saved CSV, so this runs quickly
and without needing to reload the model or re-run inference.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve


CSV_PREFIX_RE = re.compile(
    r"^evaluation_v(?P<version>[^_]+)_(?P<bat>[^_]+)_(?P<source>[^_]+)_"
    r"(?P<background>[^_]+)_(?P<timestamp>\d{8}_\d{6})\.csv$"
)


def _read_eval_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read `y_true` and `y_hat` columns from an evaluation CSV."""
    import csv as _csv

    y_true: list = []
    y_hat: list = []
    with open(csv_path, "r", newline="") as f:
        reader = _csv.DictReader(f)
        missing = [c for c in ("y_true", "y_hat") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"Evaluation CSV missing columns {missing}: {csv_path}"
            )
        for row in reader:
            try:
                y_true.append(float(row["y_true"]))
                y_hat.append(float(row["y_hat"]))
            except (TypeError, ValueError):
                continue
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_hat, dtype=np.float64)


def compute_auc_from_arrays(
    y_true: Iterable[float], y_hat: Iterable[float]
) -> Dict[str, Any]:
    """
    Compute ROC curve data and ROC-AUC from binary labels and continuous scores.

    Returns a dict with:
        roc_auc: float in [0, 1]
        fpr, tpr: np.ndarray of the full ROC curve (for plotting)
        n_positive, n_negative, n_total: int class counts
    """
    y_true_arr = np.asarray(list(y_true), dtype=np.float64).ravel()
    y_hat_arr = np.asarray(list(y_hat), dtype=np.float64).ravel()

    if y_true_arr.shape != y_hat_arr.shape:
        raise ValueError(
            f"y_true and y_hat have different shapes: {y_true_arr.shape} vs {y_hat_arr.shape}"
        )

    n_total = int(y_true_arr.size)
    n_positive = int((y_true_arr == 1.0).sum())
    n_negative = int((y_true_arr == 0.0).sum())

    if n_positive == 0 or n_negative == 0:
        # AUC is undefined when a single class is present.
        return {
            "roc_auc": float("nan"),
            "fpr": np.asarray([0.0, 1.0]),
            "tpr": np.asarray([0.0, 1.0]),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_total": n_total,
        }

    auc = float(roc_auc_score(y_true_arr, y_hat_arr))
    fpr, tpr, _ = roc_curve(y_true_arr, y_hat_arr)
    return {
        "roc_auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_total": n_total,
    }


def compute_auc_from_csv(csv_path: str) -> Dict[str, Any]:
    """Thin wrapper: read CSV, then delegate to `compute_auc_from_arrays`."""
    y_true, y_hat = _read_eval_csv(csv_path)
    return compute_auc_from_arrays(y_true, y_hat)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    output_path: str,
    title: str = "Siamese Network ROC Curve",
    subtitle: Optional[str] = None,
) -> str:
    """Render the ROC curve to `output_path` and return that path."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    if np.isnan(auc):
        auc_label = "AUC: N/A (single-class labels)"
    else:
        auc_label = f"ROC (AUC = {auc:.4f})"

    ax.plot(fpr, tpr, color="tab:blue", lw=2, label=auc_label)
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    if subtitle:
        fig.text(
            0.5, 0.92, subtitle,
            ha="center", va="top",
            fontsize=10, color="dimgray",
        )

    plt.tight_layout()
    if subtitle:
        plt.subplots_adjust(top=0.88)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _derive_roc_png_path(csv_path: str) -> str:
    """Build `roc_curve_v*.png` path from an `evaluation_v*.csv` path."""
    directory = os.path.dirname(csv_path)
    basename = os.path.basename(csv_path)
    m = CSV_PREFIX_RE.match(basename)
    if m is None:
        # Fall back to a generic name sitting next to the CSV.
        stem, _ = os.path.splitext(basename)
        return os.path.join(directory, f"roc_curve__{stem}.png")
    return os.path.join(
        directory,
        basename.replace("evaluation_v", "roc_curve_v", 1).replace(".csv", ".png"),
    )


def _parse_csv_context(csv_path: str) -> Dict[str, str]:
    """Best-effort parse of the CSV filename to populate the plot subtitle."""
    basename = os.path.basename(csv_path)
    m = CSV_PREFIX_RE.match(basename)
    if not m:
        return {}
    return {
        "model_version": m.group("version"),
        "bat_type": m.group("bat"),
        "source": m.group("source"),
        "background": m.group("background"),
        "timestamp": m.group("timestamp"),
    }


def _find_latest_eval_csv(run_dir: str) -> Optional[str]:
    """Find the newest `evaluation_v*.csv` in `run_dir`, or None."""
    candidates = sorted(
        glob.glob(os.path.join(run_dir, "evaluation_v*.csv")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None


def _merge_summary_auc(
    run_dir: str, auc_block: Dict[str, Any]
) -> Optional[str]:
    """Merge the `auc` block into `training_summary.json` if the file exists."""
    summary_path = os.path.join(run_dir, "training_summary.json")
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except Exception as exc:
        print(f"WARN: could not read {summary_path}: {exc}")
        return None

    summary["auc"] = auc_block
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary_path
    except Exception as exc:
        print(f"WARN: could not write {summary_path}: {exc}")
        return None


def _log_to_mlflow(
    auc_block: Dict[str, Any],
    roc_png_path: str,
    mlflow_run_id: Optional[str],
) -> Optional[str]:
    """
    Opportunistic MLflow logging.

    Returns a short status string describing what happened, or None if logging
    was skipped due to MLflow import failure.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        print(f"INFO: MLflow not available, skipping AUC logging: {exc}")
        return None

    metrics = {
        "test_roc_auc": auc_block["roc_auc"],
        "roc_auc_n_positive": float(auc_block["n_positive"]),
        "roc_auc_n_negative": float(auc_block["n_negative"]),
    }
    # Drop NaN AUC to avoid polluting the tracker.
    if not np.isfinite(metrics["test_roc_auc"]):
        metrics.pop("test_roc_auc")

    active = mlflow.active_run()
    if active is not None:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        if os.path.exists(roc_png_path):
            mlflow.log_artifact(roc_png_path, artifact_path="evaluation")
        return f"logged to active run {active.info.run_id}"

    if mlflow_run_id:
        client = MlflowClient()
        for key, value in metrics.items():
            client.log_metric(mlflow_run_id, key, value)
        if os.path.exists(roc_png_path):
            client.log_artifact(mlflow_run_id, roc_png_path, artifact_path="evaluation")
        return f"logged to run {mlflow_run_id}"

    return "skipped (no active run and no --mlflow-run-id provided)"


def evaluate_auc_for_run(
    run_dir: str,
    csv_path: Optional[str] = None,
    roc_png_path: Optional[str] = None,
    mlflow_run_id: Optional[str] = None,
    mlflow_log: bool = True,
    title: str = "Siamese Network ROC Curve",
) -> Dict[str, Any]:
    """
    Compute AUC from the evaluation CSV and produce the ROC PNG +
    `training_summary.json` update + optional MLflow logging.

    Parameters
    ----------
    run_dir : Training/evaluation run directory; also where outputs are written.
    csv_path : Explicit CSV; if None, picks the newest `evaluation_v*.csv` in run_dir.
    roc_png_path : Explicit output PNG; if None, derived from the CSV filename.
    mlflow_run_id : Target MLflow run id when no run is currently active.
    mlflow_log : Set False to skip MLflow entirely.
    title : Plot title.
    """
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    if csv_path is None:
        csv_path = _find_latest_eval_csv(run_dir)
        if csv_path is None:
            raise FileNotFoundError(
                f"No evaluation_v*.csv found in {run_dir} to compute AUC from."
            )

    if roc_png_path is None:
        roc_png_path = _derive_roc_png_path(csv_path)

    ctx = _parse_csv_context(csv_path)
    subtitle = None
    if ctx:
        subtitle = (
            f"v{ctx['model_version']} | bat: {ctx['bat_type']} | "
            f"source: {ctx['source']} | bg: {ctx['background']} | {ctx['timestamp']}"
        )

    print(f"Computing ROC-AUC from {csv_path}")
    result = compute_auc_from_csv(csv_path)

    plot_roc_curve(
        fpr=result["fpr"],
        tpr=result["tpr"],
        auc=result["roc_auc"],
        output_path=roc_png_path,
        title=title,
        subtitle=subtitle,
    )
    print(f"ROC curve saved to: {roc_png_path}")

    auc_block: Dict[str, Any] = {
        "roc_auc": None if np.isnan(result["roc_auc"]) else result["roc_auc"],
        "n_positive": result["n_positive"],
        "n_negative": result["n_negative"],
        "n_total": result["n_total"],
        "source_csv": os.path.basename(csv_path),
        "roc_png": os.path.basename(roc_png_path),
    }
    summary_path = _merge_summary_auc(run_dir, auc_block)
    if summary_path:
        print(f"Updated auc block in {summary_path}")

    mlflow_status: Optional[str] = None
    if mlflow_log:
        mlflow_status = _log_to_mlflow(auc_block, roc_png_path, mlflow_run_id)
        if mlflow_status:
            print(f"MLflow: {mlflow_status}")
    else:
        print("MLflow: disabled via --no-mlflow-log")

    if np.isnan(result["roc_auc"]):
        print("ROC-AUC: N/A (single-class labels in CSV)")
    else:
        print(
            f"ROC-AUC: {result['roc_auc']:.4f} "
            f"(n_pos={result['n_positive']}, n_neg={result['n_negative']}, "
            f"n_total={result['n_total']})"
        )

    return {
        "roc_auc": auc_block["roc_auc"],
        "n_positive": result["n_positive"],
        "n_negative": result["n_negative"],
        "n_total": result["n_total"],
        "csv_path": csv_path,
        "roc_png_path": roc_png_path,
        "summary_path": summary_path,
        "mlflow_status": mlflow_status,
    }
