#!/usr/bin/env python3
"""
Prune one siamese training run directory to four best-metric SavedModels + training_summary.json.

Resolves best epochs from MLflow (tag model_output_dir) or from training.log, then rebuilds
best_model_{loss,f1,recall,precision} from siamesemodelv2_v{epoch} when present, with
same-epoch symlink deduplication. Deletes checkpoints/, siamesemodelv2_v*, and stray
siamesemodelv2 by default.

Dry-run is default; use --execute to apply.

If every best-metric epoch (loss, F1, recall, precision) is strictly below 5, the run is treated as
too early / unreliable: the script removes the entire experiment directory instead of normal pruning.

Usage:
  python scripts/prune_siamese_run.py /path/to/run_dir
  python scripts/prune_siamese_run.py /path/to/run_dir --execute
  python scripts/prune_siamese_run.py /path/to/run_dir --use-log --nearest-missing --execute
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_ROOT / "app"))

ROLE_ORDER = ("loss", "f1", "recall", "precision")
# If all four best epochs are < this value, delete the whole run (no partial prune).
EARLY_EXPERIMENT_EPOCH_THRESHOLD = 5
ROLE_TO_DIR = {
    "loss": "best_model_loss",
    "f1": "best_model_f1",
    "recall": "best_model_recall",
    "precision": "best_model_precision",
}

TEST_LINE = re.compile(
    r"Test results - Loss:\s*([\d.eE+-]+).*?Recall:\s*([\d.eE+-]+).*?Precision:\s*([\d.eE+-]+).*?F1:\s*([\d.eE+-]+)"
)


def _remove_path(path: str) -> None:
    if not os.path.lexists(path):
        return
    if os.path.islink(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def parse_training_log(log_path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = TEST_LINE.search(line)
            if m:
                rows.append(
                    {
                        "loss": float(m.group(1)),
                        "recall": float(m.group(2)),
                        "precision": float(m.group(3)),
                        "f1": float(m.group(4)),
                    }
                )
    return rows


def all_best_epochs_strictly_below(
    epochs: Dict[str, Tuple[int, float]], threshold: int
) -> bool:
    """True when each role's best epoch is < threshold (e.g. all in 1..4 when threshold is 5)."""
    if len(epochs) < 4:
        return False
    return all(epochs[k][0] < threshold for k in ROLE_ORDER)


def epochs_from_log_rows(rows: List[Dict[str, float]]) -> Dict[str, Tuple[int, float]]:
    if not rows:
        return {}
    n = len(rows)
    loss_i = min(range(n), key=lambda i: rows[i]["loss"])
    f1_i = max(range(n), key=lambda i: rows[i]["f1"])
    rec_i = max(range(n), key=lambda i: rows[i]["recall"])
    prec_i = max(range(n), key=lambda i: rows[i]["precision"])

    def ep(i: int) -> int:
        return i + 1

    return {
        "loss": (ep(loss_i), rows[loss_i]["loss"]),
        "f1": (ep(f1_i), rows[f1_i]["f1"]),
        "recall": (ep(rec_i), rows[rec_i]["recall"]),
        "precision": (ep(prec_i), rows[prec_i]["precision"]),
    }


def _best_metric_step(
    client: Any, run_id: str, key: str, mode: str
) -> Optional[Tuple[int, float]]:
    hist = client.get_metric_history(run_id, key)
    if not hist:
        return None
    if mode == "min":
        best_val = min(m.value for m in hist)
        cands = [m for m in hist if m.value == best_val]
    else:
        best_val = max(m.value for m in hist)
        cands = [m for m in hist if m.value == best_val]
    m0 = min(cands, key=lambda m: m.step)
    return int(m0.step), float(m0.value)


def resolve_mlflow_uri(uri: str) -> str:
    """Match SiameseNetworkTrainer._resolve_mlflow_tracking_uri logic without importing TF."""
    if uri.startswith("file:///"):
        return uri
    if uri.startswith("file:./"):
        uri = uri[5:]
    if uri.startswith("./") or (not uri.startswith("/") and not uri.startswith("file://")):
        relative_path = uri[2:] if uri.startswith("./") else uri
        absolute_path = os.path.join(str(_APP_ROOT), relative_path)
        return f"file://{absolute_path}"
    if not uri.startswith("file://"):
        return f"file://{uri}"
    return uri


def epochs_from_mlflow(run_dir: str, tracking_uri: str) -> Optional[Dict[str, Tuple[int, float]]]:
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("MLflow not installed; use --use-log or install mlflow.", file=sys.stderr)
        return None

    client = MlflowClient(tracking_uri)
    run_dir_abs = os.path.abspath(run_dir)
    run_dir_alt = run_dir_abs.rstrip(os.sep)
    exps = client.search_experiments()
    exp_ids = [e.experiment_id for e in exps]
    if not exp_ids:
        return None

    # Prefer tag (newer trainer); fall back to param output_dir (older runs store path here).
    filters = [
        f"tags.model_output_dir = '{run_dir_abs}'",
        f"tags.model_output_dir = '{run_dir_alt}'",
        f"params.output_dir = '{run_dir_abs}'",
        f"params.output_dir = '{run_dir_alt}'",
    ]
    runs = []
    for filt in filters:
        runs = client.search_runs(experiment_ids=exp_ids, filter_string=filt, max_results=10)
        if runs:
            break
    if not runs:
        return None

    run_id = runs[0].info.run_id
    out: Dict[str, Tuple[int, float]] = {}
    bl = _best_metric_step(client, run_id, "test_loss", "min")
    if bl:
        out["loss"] = bl
    bf = _best_metric_step(client, run_id, "test_f1", "max")
    if bf:
        out["f1"] = bf
    br = _best_metric_step(client, run_id, "test_recall", "max")
    if br:
        out["recall"] = br
    bp = _best_metric_step(client, run_id, "test_precision", "max")
    if bp:
        out["precision"] = bp
    return out if len(out) == 4 else None


def list_version_dirs(run_dir: str) -> List[Tuple[int, str]]:
    pat = re.compile(r"^siamesemodelv2_v(\d+)$")
    found: List[Tuple[int, str]] = []
    for name in os.listdir(run_dir):
        m = pat.match(name)
        if m and os.path.isdir(os.path.join(run_dir, name)):
            found.append((int(m.group(1)), os.path.join(run_dir, name)))
    return sorted(found, key=lambda x: x[0])


def resolve_source_dir(
    run_dir: str,
    epoch: int,
    nearest_missing: bool,
) -> Tuple[Optional[str], List[str]]:
    warnings: List[str] = []
    vdir = os.path.join(run_dir, f"siamesemodelv2_v{epoch}")
    if os.path.isdir(vdir):
        return vdir, warnings

    if not nearest_missing:
        warnings.append(f"No siamesemodelv2_v{epoch} for requested epoch {epoch}.")
        return None, warnings

    versions = list_version_dirs(run_dir)
    if not versions:
        warnings.append(f"No siamesemodelv2_v* dirs; cannot satisfy epoch {epoch}.")
        return None, warnings

    best_e, best_path = min(versions, key=lambda t: abs(t[0] - epoch))
    warnings.append(
        f"Using nearest periodic snapshot siamesemodelv2_v{best_e} instead of epoch {epoch} "
        f"(off by {abs(best_e - epoch)})."
    )
    return best_path, warnings


def materialize_roles(
    run_dir: str,
    epochs: Dict[str, Tuple[int, float]],
    nearest_missing: bool,
    execute: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    """Rebuild best_model_* dirs; first physical save per epoch wins, others symlink."""
    epoch_canonical: Dict[int, str] = {}
    all_warnings: List[str] = []
    layout: Dict[str, Any] = {}

    for role in ROLE_ORDER:
        ep, val = epochs[role]
        subdir = ROLE_TO_DIR[role]
        dest = os.path.join(run_dir, subdir)

        if ep in epoch_canonical:
            canon = epoch_canonical[ep]
            layout[role] = {
                "epoch": ep,
                "value": val,
                "action": "symlink",
                "target": canon,
                "path": subdir,
            }
            if execute:
                _remove_path(dest)
                os.symlink(canon, dest, target_is_directory=True)
            continue

        src, w = resolve_source_dir(run_dir, ep, nearest_missing)
        all_warnings.extend(w)
        if not src:
            layout[role] = {
                "epoch": ep,
                "value": val,
                "error": "no_source_savedmodel",
                "path": subdir,
            }
            continue

        layout[role] = {
            "epoch": ep,
            "value": val,
            "action": "copytree",
            "from": os.path.basename(src),
            "path": subdir,
        }
        if execute:
            _remove_path(dest)
            shutil.copytree(src, dest)
            epoch_canonical[ep] = subdir

    return layout, all_warnings


def collect_delete_candidates(run_dir: str) -> List[str]:
    out: List[str] = []
    ck = os.path.join(run_dir, "checkpoints")
    if os.path.isdir(ck):
        out.append(ck)

    sm = os.path.join(run_dir, "siamesemodelv2")
    if os.path.isdir(sm):
        out.append(sm)

    for name in os.listdir(run_dir):
        if re.match(r"^siamesemodelv2_v\d+$", name):
            p = os.path.join(run_dir, name)
            if os.path.isdir(p):
                out.append(p)
    return sorted(out)


def write_training_summary(
    run_dir: str,
    epochs: Dict[str, Tuple[int, float]],
    layout: Dict[str, Any],
    warnings: List[str],
) -> str:
    def describe(subdir: str) -> Dict[str, Any]:
        p = os.path.join(run_dir, subdir)
        o: Dict[str, Any] = {"relative_path": subdir, "exists": os.path.lexists(p)}
        if os.path.lexists(p):
            o["is_symlink"] = os.path.islink(p)
            if o["is_symlink"]:
                o["symlink_target"] = os.readlink(p)
        return o

    summary = {
        "model_output_dir": run_dir,
        "best_loss": {"epoch": epochs["loss"][0], "value": epochs["loss"][1], **describe("best_model_loss")},
        "best_f1": {"epoch": epochs["f1"][0], "value": epochs["f1"][1], **describe("best_model_f1")},
        "best_recall": {
            "epoch": epochs["recall"][0],
            "value": epochs["recall"][1],
            **describe("best_model_recall"),
        },
        "best_precision": {
            "epoch": epochs["precision"][0],
            "value": epochs["precision"][1],
            **describe("best_model_precision"),
        },
        "prune_layout": layout,
        "warnings": warnings,
    }
    path = os.path.join(run_dir, "training_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune one siamese run to four best SavedModels.")
    parser.add_argument("run_dir", type=str, help="Path to a single training output directory")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply changes (default: dry-run only)",
    )
    parser.add_argument(
        "--use-log",
        action="store_true",
        help="Use training.log instead of MLflow",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: from app config)",
    )
    parser.add_argument(
        "--nearest-missing",
        action="store_true",
        help="Use nearest siamesemodelv2_v* when exact epoch folder is missing",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        return 1

    epochs: Optional[Dict[str, Tuple[int, float]]] = None
    source = ""

    if args.use_log:
        log_path = os.path.join(run_dir, "training.log")
        if not os.path.isfile(log_path):
            print(f"No training.log at {log_path}", file=sys.stderr)
            return 1
        rows = parse_training_log(log_path)
        epochs = epochs_from_log_rows(rows)
        source = "training.log"
    else:
        from config.loader import ConfigLoader

        cfg = ConfigLoader()
        uri = args.tracking_uri or cfg.mlflow.tracking_uri
        resolved = resolve_mlflow_uri(uri)
        epochs = epochs_from_mlflow(run_dir, resolved)
        source = "mlflow"
        if epochs is None and os.path.isfile(os.path.join(run_dir, "training.log")):
            rows = parse_training_log(os.path.join(run_dir, "training.log"))
            epochs = epochs_from_log_rows(rows)
            source = "training.log (fallback after MLflow miss)"

    if not epochs or len(epochs) < 4:
        print(
            "Could not resolve four best epochs (need MLflow metrics or training.log).",
            file=sys.stderr,
        )
        return 1

    print(f"Resolved epochs from {source}:")
    for k in ROLE_ORDER:
        ep, val = epochs[k]
        print(f"  {k}: epoch={ep} value={val}")

    if all_best_epochs_strictly_below(epochs, EARLY_EXPERIMENT_EPOCH_THRESHOLD):
        print(
            f"\nAll best epochs are < {EARLY_EXPERIMENT_EPOCH_THRESHOLD}: "
            "removing entire experiment directory (early / unreliable run)."
        )
        if args.execute:
            shutil.rmtree(run_dir)
            print(f"Deleted: {run_dir}")
        else:
            print(f"Dry-run: would delete entire directory: {run_dir}")
        return 0

    layout, mat_warnings = materialize_roles(
        run_dir, epochs, args.nearest_missing, execute=args.execute
    )
    for w in mat_warnings:
        print(f"  WARNING: {w}")

    deletes = collect_delete_candidates(run_dir)
    print("\nPlanned deletions:" if not args.execute else "\nDeleting:")
    for p in deletes:
        print(f"  {p}")

    all_warnings = list(mat_warnings)
    if args.execute:
        for p in deletes:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.lexists(p):
                _remove_path(p)
        summary_path = write_training_summary(run_dir, epochs, layout, all_warnings)
        print(f"\nWrote {summary_path}")
    else:
        print("\nDry-run: no changes made. Pass --execute to apply.")
        print("Would write training_summary.json after deletes.")

    print("\nMaterialize plan:")
    print(json.dumps(layout, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
