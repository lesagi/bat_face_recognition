"""Aggregate the full multi-seed matrix (multiseed-full-*) from MLflow into the
report CSV (outputs/multiseed_full_results.csv) consumed by gen_results_assets.py.

Groups the Siamese/ArcFace/AdaFace × {green,original,random} × 5-seed runs by
(species, model, background) and reports test ROC-AUC / top-1 / mAP as mean ± std.

Old and new runs coexist in the experiment (the re-run reused the experiment
name), so we dedup by (species, model, bg, seed) keeping the newest — the fresh
runs win over the July batch, exactly as aggregate_resolution.py does.

  conda activate frec && uv run python scripts/aggregate_matrix.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")

FAMILY = {"binary_focal": "Siamese", "arcface": "ArcFace", "adaface": "AdaFace"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-name", default="multiseed-full-20260706")
    ap.add_argument("--output", default="outputs/multiseed_full_results.csv", type=Path)
    args = ap.parse_args()

    exp = mlflow.get_experiment_by_name(args.experiment_name)
    if exp is None:
        raise SystemExit(f"experiment {args.experiment_name!r} not found")
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    df = df[df["metrics.test/roc_auc"].notna()].copy()
    if df.empty:
        raise SystemExit("no finished runs with test/roc_auc yet")

    df["model"] = df["params.loss_type"].map(FAMILY).fillna(df["params.loss_type"])
    # Seed lives in a run tag; fall back to the run-name suffix (...-s<seed>).
    seed = df.get("tags.seed")
    if seed is None:
        seed = df["tags.mlflow.runName"].str.extract(r"-s(\d+)$")[0]
    df["seed"] = seed

    # Keep the newest run per (species, model, bg, seed): the re-run's fresh runs
    # supersede the July batch that lingers in the same experiment.
    if "end_time" in df.columns:
        df = df.sort_values("end_time")
    df = df.drop_duplicates(
        subset=["params.species", "model", "params.background", "seed"], keep="last"
    )

    keys = ["params.species", "model", "params.background"]
    agg = (
        df.groupby(keys)
        .agg(
            n=("metrics.test/roc_auc", "size"),
            roc_auc_mean=("metrics.test/roc_auc", "mean"),
            roc_auc_std=("metrics.test/roc_auc", "std"),
            top1_mean=("metrics.test/top1", "mean"),
            top1_std=("metrics.test/top1", "std"),
            map_mean=("metrics.test/map", "mean"),
            map_std=("metrics.test/map", "std"),
        )
        .reset_index()
        .rename(columns={"params.species": "species", "params.background": "bg"})
    )
    # Column order + row order to match the report's expectations.
    model_order = {"Siamese": 0, "ArcFace": 1, "AdaFace": 2}
    agg["_m"] = agg["model"].map(model_order).fillna(9)
    agg = (
        agg.sort_values(["species", "bg", "_m"])
        .drop(columns="_m")[
            [
                "species",
                "model",
                "bg",
                "n",
                "roc_auc_mean",
                "roc_auc_std",
                "top1_mean",
                "top1_std",
                "map_mean",
                "map_std",
            ]
        ]
        .reset_index(drop=True)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.output, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}" if pd.notna(v) else "NA")
    print("=== full matrix: test metrics (mean ± std over seeds) ===")
    print(agg.to_string(index=False))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
