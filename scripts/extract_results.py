"""Pull test metrics for an experiment matrix from MLflow.

Reads a local MLflow experiment and prints a tidy results table (one row per
model x dataset condition) plus the raw values needed for the paper tables.

  # original PR11/PR12 re-run matrix (defaults):
  uv run python scripts/extract_results.py
  # mauritius 16-bat set:
  uv run python scripts/extract_results.py --experiment-name mauritius-16bat \
      --log-dir outputs/mauritius_logs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mlflow
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")

FAMILY = {"binary_focal": "Siamese", "arcface": "ArcFace", "adaface": "AdaFace"}


def _authoritative_run_ids(summaries: list[Path]) -> dict[str, str]:
    """experiment -> run_id from the driver summary files (final clean run)."""
    mapping: dict[str, str] = {}
    for s in summaries:
        if not s.exists():
            continue
        for line in s.read_text().splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 3 and parts[1] == "OK" and parts[2] not in ("", "NA"):
                mapping[parts[0]] = parts[2]
    return mapping


#: Pre-declared primary metric for the significance claim. Mirrors
#: ``bat_stats.PRIMARY_METRIC`` (kept literal so this script has no import-time
#: dependency on the workspace).
PERM_PRIMARY_METRIC = "roc_auc"

#: Fallback order for older logs, which predate roc_auc being permuted.
PERM_FALLBACK_METRICS = ("f1", "accuracy", "precision", "recall")


def _perm_pvalues(log_dir: Path, experiment: str) -> dict[str, float]:
    """Parse every permutation p-value from the experiment log, by metric."""
    log = log_dir / f"{experiment}.log"
    if not log.exists():
        return {}
    out: dict[str, float] = {}
    for line in log.read_text().splitlines():
        m = re.match(
            r"^(roc_auc|f1|accuracy|precision|recall)\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+<?([\d.]+)",
            line,
        )
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def _perm_primary_pvalue(log_dir: Path, experiment: str) -> float | None:
    """Permutation p-value for the *pre-declared* primary metric.

    Was ``min()`` over all five metrics, which is a multiple-comparisons
    cherry-pick: taking the best of five correlated tests inflates the
    false-positive rate well past the nominal alpha. Fix the metric in advance
    instead. Falls back through the thresholded metrics for logs written before
    roc_auc was added to the permutation test.
    """
    pvals = _perm_pvalues(log_dir, experiment)
    if not pvals:
        return None
    if PERM_PRIMARY_METRIC in pvals:
        return pvals[PERM_PRIMARY_METRIC]
    for metric in PERM_FALLBACK_METRICS:
        if metric in pvals:
            return pvals[metric]
    return None


def _get(row: pd.Series, key: str) -> float | None:
    col = f"metrics.{key}"
    if col in row and pd.notna(row[col]):
        return float(row[col])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--experiment-name",
        default="bat-rerun-gpu",
        help="MLflow experiment to aggregate (falls back to 'bat-rerun').",
    )
    ap.add_argument("--log-dir", default="outputs/rerun_logs", type=Path)
    ap.add_argument(
        "--summaries",
        nargs="*",
        default=None,
        help="Driver summary TSVs (default: <log-dir>/summary_{a,b}.tsv).",
    )
    ap.add_argument(
        "--output", default=None, type=Path, help="CSV out (default <log-dir>/results_table.csv)."
    )
    args = ap.parse_args()

    log_dir: Path = args.log_dir
    summaries = (
        [Path(s) for s in args.summaries]
        if args.summaries is not None
        else [log_dir / "summary_a.tsv", log_dir / "summary_b.tsv"]
    )
    output = args.output or (log_dir / "results_table.csv")

    exp = mlflow.get_experiment_by_name(args.experiment_name)
    if exp is None and args.experiment_name == "bat-rerun-gpu":
        exp = mlflow.get_experiment_by_name("bat-rerun")
    if exp is None:
        raise SystemExit(f"experiment {args.experiment_name!r} not found")
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id])

    authoritative = _authoritative_run_ids(summaries)
    id_to_exp = {rid: e for e, rid in authoritative.items()}
    df = df[df["run_id"].isin(set(authoritative.values()))]

    rows = []
    for _, r in df.iterrows():
        model = FAMILY.get(str(r.get("params.loss_type")), str(r.get("params.loss_type")))
        experiment = id_to_exp.get(r.get("run_id"), "")
        rows.append(
            {
                "model": model,
                "species": r.get("params.species"),
                "bg": r.get("params.background"),
                "roc_auc": _get(r, "test/roc_auc"),
                "f1_valthr": _get(r, "test_val_threshold/f1"),
                "top1": _get(r, "test/top1"),
                "map": _get(r, "test/map"),
                "tar_1e3": _get(r, "test/tar_at_far_1e3"),
                "recall_far_1e2": _get(r, "test/recall_at_far_1e2"),
                "cosine_roc_auc": _get(r, "test_cosine/roc_auc"),
                "val_roc_auc": _get(r, "val/roc_auc"),
                "perm_p": _perm_primary_pvalue(log_dir, experiment),
                "run_id": r.get("run_id"),
            }
        )

    out = pd.DataFrame(rows).sort_values(["species", "bg", "model"]).reset_index(drop=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}" if pd.notna(v) else "NA")
    print(out.to_string(index=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
