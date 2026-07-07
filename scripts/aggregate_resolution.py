"""Aggregate the resolution study (multiseed-res-*) from MLflow.

Groups the ArcFace/AdaFace × {green,original,random} × {224,320}px × 5-seed runs
by (species, model, background, edge) and reports test ROC-AUC / top-1 as
mean ± std, plus the 320-minus-224 delta per cell. Recognition is expected NOT
to improve with resolution (the prior 112→224 study found mean ΔROC-AUC ≈ −0.03);
this quantifies the 224→320 increment on the fresh higher-resolution lineage.

  uv run python scripts/aggregate_resolution.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")

MODEL = {"arcface": "ArcFace", "adaface": "AdaFace"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-name", default="multiseed-res-20260707")
    ap.add_argument("--output", default="outputs/resolution_results.csv", type=Path)
    args = ap.parse_args()

    exp = mlflow.get_experiment_by_name(args.experiment_name)
    if exp is None:
        raise SystemExit(f"experiment {args.experiment_name!r} not found")
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    df = df[df["metrics.test/roc_auc"].notna()].copy()
    if df.empty:
        raise SystemExit("no finished runs with test/roc_auc yet")

    df["model"] = df["tags.model"].map(MODEL).fillna(df["tags.model"])
    df["edge"] = df["tags.edge"].astype(int)
    # One row per (species,model,bg,edge,seed): disk-full casualties from the
    # first matrix pass linger as separate runs; keep the newest complete one.
    if "end_time" in df.columns:
        df = df.sort_values("end_time")
    df = df.drop_duplicates(
        subset=["tags.species", "model", "tags.background", "edge", "tags.seed"], keep="last"
    )
    keys = ["tags.species", "model", "tags.background", "edge"]
    agg = (
        df.groupby(keys)
        .agg(
            roc_auc_mean=("metrics.test/roc_auc", "mean"),
            roc_auc_std=("metrics.test/roc_auc", "std"),
            top1_mean=("metrics.test/top1", "mean"),
            top1_std=("metrics.test/top1", "std"),
            n=("metrics.test/roc_auc", "size"),
        )
        .reset_index()
        .rename(columns={"tags.species": "species", "tags.background": "bg"})
        .sort_values(["species", "model", "bg", "edge"])
        .reset_index(drop=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.output, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}" if pd.notna(v) else "NA")
    print("=== per-cell test metrics (mean ± std over seeds) ===")
    print(agg.to_string(index=False))

    # 320 - 224 delta per (species, model, bg)
    piv = agg.pivot_table(index=["species", "model", "bg"], columns="edge", values="roc_auc_mean")
    if 224 in piv.columns and 320 in piv.columns:
        piv["delta_320_224"] = piv[320] - piv[224]
        print("\n=== ROC-AUC: 224 vs 320 (delta = 320 - 224) ===")
        print(piv.to_string())
        print(f"\nmean delta(320-224) ROC-AUC across all cells: {piv['delta_320_224'].mean():+.3f}")
        by_bg = piv.groupby("bg")["delta_320_224"].mean()
        print("mean delta by background:")
        print(by_bg.to_string())
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
