"""Aggregate the 20-fold matrix from MLflow into one per-run Parquet table.

Unlike ``aggregate_matrix.py``, which collapses seeds into mean±std per cell,
this keeps **one row per run**. The split-size sensitivity analysis regresses the
metric against the realised ``n_test_ids``, so pre-aggregating would throw away
the independent variable.

Coverage is reported explicitly and never silently truncated: a missing cell
means a run failed, and reading a mean over 14 folds as if it were 20 is exactly
the sort of quiet error this table exists to prevent.

Usage:
    uv run python scripts/aggregate_kfold.py --experiment-name kfold20-20260804
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")

FAMILY = {"binary_focal": "Siamese", "arcface": "ArcFace", "adaface": "AdaFace"}
MODEL_ORDER = {"Siamese": 0, "ArcFace": 1, "AdaFace": 2}

# Metrics pulled per run. Absent ones become NaN (Siamese has no top1/map).
METRIC_COLUMNS = {
    "roc_auc": "metrics.test/roc_auc",
    "top1": "metrics.test/top1",
    "map": "metrics.test/map",
    "f1_valthr": "metrics.test_val_threshold/f1",
    "cosine_roc_auc": "metrics.test_cosine/roc_auc",
    "val_roc_auc": "metrics.val/roc_auc",
    # Permutation p-value on the pre-declared primary metric. `perm_p` is the
    # one the significance analysis uses; taking min() across the five permuted
    # metrics would be a multiple-comparisons cherry-pick.
    "perm_p": "metrics.test/perm_p_roc_auc",
    "perm_p_f1": "metrics.test/perm_p_f1",
    "perm_n": "metrics.test/perm_n_permutations",
}

# Split provenance logged by the runtime re-split (see bat_cli.runtime.SplitInfo).
SPLIT_COLUMNS = {
    "fold_id": "params.split.fold_id",
    "split_seed": "params.split.split_seed",
    "size_mode": "params.split.size_mode",
    "n_train_ids": "params.split.n_train_ids",
    "n_val_ids": "params.split.n_val_ids",
    "n_test_ids": "params.split.n_test_ids",
    "n_train_imgs": "params.split.n_train_imgs",
    "n_val_imgs": "params.split.n_val_imgs",
    "n_test_imgs": "params.split.n_test_imgs",
    "assignment_sha256": "params.split.assignment_sha256",
    "source_manifest_hash": "params.split.source_manifest_hash",
}

INT_COLUMNS = (
    "fold_id",
    "split_seed",
    "n_train_ids",
    "n_val_ids",
    "n_test_ids",
    "n_train_imgs",
    "n_val_imgs",
    "n_test_imgs",
)

EXPECTED_FOLDS = 20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs/kfold_results.parquet"))
    ap.add_argument(
        "--expected-folds",
        type=int,
        default=EXPECTED_FOLDS,
        help="Folds each cell should have; shortfalls are reported.",
    )
    args = ap.parse_args()

    exp = mlflow.get_experiment_by_name(args.experiment_name)
    if exp is None:
        raise SystemExit(f"experiment {args.experiment_name!r} not found")

    raw = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if raw.empty:
        raise SystemExit("experiment has no runs")

    df = raw[raw["metrics.test/roc_auc"].notna()].copy()
    print(f"runs in experiment : {len(raw)}")
    print(f"with test/roc_auc  : {len(df)}")
    if df.empty:
        raise SystemExit("no finished runs with test/roc_auc yet")

    out = pd.DataFrame(index=df.index)
    out["run_id"] = df["run_id"]
    out["species"] = df.get("params.species")
    out["bg"] = df.get("params.background")
    out["model"] = df["params.loss_type"].map(FAMILY).fillna(df["params.loss_type"])

    for name, column in {**METRIC_COLUMNS, **SPLIT_COLUMNS}.items():
        out[name] = df[column] if column in df.columns else pd.NA

    # fold_id also lives in a run tag; fall back to it, then to the run name.
    if out["fold_id"].isna().all():
        out["fold_id"] = df.get("tags.fold")
    if out["fold_id"].isna().all():
        out["fold_id"] = df["tags.mlflow.runName"].str.extract(r"-f(\d+)$")[0]

    for column in INT_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")

    missing_split = int(out["n_test_ids"].isna().sum())
    if missing_split:
        print(
            f"! {missing_split} runs have no split.* params — they predate the "
            "runtime re-split, or ran without --fold"
        )

    # A re-run can leave two runs for the same cell+fold; keep the newest.
    if "end_time" in df.columns:
        out["end_time"] = df["end_time"]
        out = out.sort_values("end_time")
    before = len(out)
    out = out.drop_duplicates(subset=["species", "model", "bg", "fold_id"], keep="last")
    if len(out) != before:
        print(f"deduped {before - len(out)} superseded run(s) by (species, model, bg, fold)")

    out["_m"] = out["model"].map(MODEL_ORDER).fillna(9)
    out = (
        out.sort_values(["species", "bg", "_m", "fold_id"])
        .drop(columns=["_m"])
        .reset_index(drop=True)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"\nwrote {args.output} ({len(out)} rows)")

    # ---- coverage, stated explicitly ----
    print(f"\n=== coverage (expected {args.expected_folds} folds per cell) ===")
    counts = out.groupby(["species", "model", "bg"]).size().rename("n_folds").reset_index()
    short = counts[counts["n_folds"] < args.expected_folds]
    print(f"  cells present : {len(counts)}")
    print(f"  runs total    : {len(out)} / {len(counts) * args.expected_folds} expected")
    if short.empty:
        print("  all cells complete")
    else:
        print(f"  INCOMPLETE cells ({len(short)}):")
        for row in short.itertuples(index=False):
            print(f"    {row.species:<10} {row.model:<9} {row.bg:<9} {row.n_folds}/{args.expected_folds}")

    # ---- per-cell summary, for a quick eyeball ----
    print("\n=== test ROC-AUC per cell (mean ± std over folds) ===")
    summary = (
        out.groupby(["species", "model", "bg"])
        .agg(
            n=("roc_auc", "size"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            n_test_ids_min=("n_test_ids", "min"),
            n_test_ids_max=("n_test_ids", "max"),
        )
        .reset_index()
    )
    for row in summary.itertuples(index=False):
        std = "  n/a" if pd.isna(row.roc_auc_std) else f"{row.roc_auc_std:.3f}"
        print(
            f"  {row.species:<10} {row.model:<9} {row.bg:<9} n={row.n:2d}  "
            f"{row.roc_auc_mean:.3f} ± {std}  "
            f"n_test_ids {row.n_test_ids_min}-{row.n_test_ids_max}"
        )

    print("\n=== realised held-out identity counts ===")
    if out["n_test_ids"].notna().any():
        for species, group in out.groupby("species"):
            dist = group["n_test_ids"].value_counts().sort_index()
            spread = ", ".join(f"{k}:{v}" for k, v in dist.items())
            print(f"  {species:<10} {spread}")
    else:
        print("  none recorded")

    # ---- permutation p-value availability + floor censoring ----
    print("\n=== permutation p-values (primary metric: roc_auc) ===")
    if out["perm_p"].notna().any():
        have = int(out["perm_p"].notna().sum())
        print(f"  present : {have}/{len(out)} runs")
        n_perm = out["perm_n"].dropna()
        if not n_perm.empty:
            b = int(n_perm.mode().iloc[0])
            floor = 1.0 / (b + 1)
            censored = int((out["perm_p"] <= floor + 1e-12).sum())
            print(f"  B       : {b}  -> resolution floor {floor:.6f}")
            print(
                f"  censored: {censored}/{have} runs sit ON the floor "
                "(their p is an upper bound, not an estimate)"
            )
    else:
        print("  none recorded — runs predate permutation metrics being logged;")
        print("  the p-values are still inside each run's permutation/ artifact.")


if __name__ == "__main__":
    main()
