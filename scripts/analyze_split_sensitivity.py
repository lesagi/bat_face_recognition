"""How much does the number of held-out identities drive the result?

Falls out of the seeded k-fold design for free, and answers a question the
project has been working around rather than measuring: the mauritius test split
has 2 identities and a single run's ROC-AUC can swing by >0.1, so every number
is reported as mean±std over seeds. But seeds only varied the training RNG — the
held-out bats were always the same two. With ``--fold``, both the membership and
the *size* of the test split vary, which makes two things measurable:

1. **Accuracy vs test size** -- does the mean metric drift as more identities are
   held out? (It should not, if the metric is unbiased; a drift means the score
   depends on how many strangers the model is asked to tell apart.)
2. **Stability vs test size** -- how fast does the spread shrink? This is the
   practical payoff: it says how many test identities a *trustworthy* number
   needs, and therefore how much of the published ±0.1 is just small-test noise.

Confound stated up front, not buried: the identity pool is fixed, so
``n_train_ids`` and ``n_test_ids`` move in opposite directions by construction.
A drift in the metric cannot be cleanly attributed to one or the other. The
regression reports both coefficients, and the honest reading is "the *split
shape* matters this much", not "test size specifically causes it".

Usage:
    uv run python scripts/analyze_split_sensitivity.py
    uv run python scripts/analyze_split_sensitivity.py --metric top1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bat_stats import bootstrap_ci, build_filename  # noqa: E402

DEFAULT_INPUT = Path("outputs/kfold_results.parquet")
DEFAULT_OUT = Path("outputs/kfold_split_sensitivity.json")
DEFAULT_FIG_DIR = Path("outputs/quality/figures")

SPECIES_COLOURS = {"mauritius": "#A2582C", "rousettus": "#2F6B7A"}

# A level needs at least this many folds before its spread means anything.
MIN_FOLDS_PER_LEVEL = 3


def ols_with_two_predictors(
    y: np.ndarray, x1: np.ndarray, x2: np.ndarray
) -> dict[str, float]:
    """Least squares ``y ~ 1 + x1 + x2``, returning coefficients and R².

    Hand-rolled rather than pulled from statsmodels: the design is tiny and this
    keeps the script's dependencies to what the workspace already has.
    """
    design = np.column_stack([np.ones_like(x1, dtype=float), x1, x2])
    # x1 and x2 are near-collinear by construction (fixed identity pool), so use
    # lstsq, which handles a rank-deficient design without blowing up.
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual_ss = float(np.sum((y - fitted) ** 2))
    total_ss = float(np.sum((y - y.mean()) ** 2))
    return {
        "intercept": float(coefficients[0]),
        "slope_n_test_ids": float(coefficients[1]),
        "slope_n_train_ids": float(coefficients[2]),
        "r_squared": float(1.0 - residual_ss / total_ss) if total_ss > 0 else float("nan"),
        "design_rank": int(rank),
        "n": int(y.size),
    }


def per_level_stats(group: pd.DataFrame, metric: str) -> list[dict[str, Any]]:
    """Mean, spread and CI width of *metric* at each held-out identity count."""
    rows: list[dict[str, Any]] = []
    for level, sub in group.groupby("n_test_ids"):
        values = sub[metric].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        entry: dict[str, Any] = {
            "n_test_ids": int(level),
            "n_folds": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        if values.size >= MIN_FOLDS_PER_LEVEL:
            ci = bootstrap_ci(values, statistic=np.mean, n_boot=5000)
            entry["ci_low"] = ci.ci_low
            entry["ci_high"] = ci.ci_high
            entry["ci_width"] = ci.ci_high - ci.ci_low
        rows.append(entry)
    return rows


def analyse(df: pd.DataFrame, metric: str) -> dict[str, Any]:
    report: dict[str, Any] = {"metric": metric, "species": {}}

    for species, group in df.groupby("species"):
        group = group.dropna(subset=[metric, "n_test_ids", "n_train_ids"])
        if group.empty:
            continue

        y = group[metric].to_numpy(dtype=float)
        n_test = group["n_test_ids"].to_numpy(dtype=float)
        n_train = group["n_train_ids"].to_numpy(dtype=float)

        entry: dict[str, Any] = {
            "n_runs": int(len(group)),
            "correlation_test_vs_train_ids": (
                float(np.corrcoef(n_test, n_train)[0, 1]) if len(group) > 2 else float("nan")
            ),
            "regression": ols_with_two_predictors(y, n_test, n_train),
            "levels": per_level_stats(group, metric),
            "pooled_std": float(y.std(ddof=1)) if y.size > 1 else float("nan"),
        }

        # Spearman-style rank correlation between the level and its spread: the
        # stability question in one number. Negative means more held-out
        # identities give a tighter estimate.
        levels = [lv for lv in entry["levels"] if np.isfinite(lv.get("std", np.nan))]
        if len(levels) >= 3:
            xs = np.array([lv["n_test_ids"] for lv in levels], dtype=float)
            stds = np.array([lv["std"] for lv in levels], dtype=float)
            entry["std_vs_test_size_correlation"] = float(
                np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(stds)))[0, 1]
            )
        else:
            entry["std_vs_test_size_correlation"] = float("nan")

        report["species"][species] = entry

    return report


def plot(df: pd.DataFrame, report: dict[str, Any], metric: str, fig_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel 1: every fold, metric vs held-out identity count.
    ax = axes[0]
    for species, group in df.groupby("species"):
        group = group.dropna(subset=[metric, "n_test_ids"])
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.12, 0.12, len(group))
        ax.scatter(
            group["n_test_ids"].to_numpy(dtype=float) + jitter,
            group[metric],
            s=22,
            alpha=0.65,
            color=SPECIES_COLOURS.get(species, "#666"),
            edgecolor="white",
            linewidth=0.4,
            label=species,
        )
    ax.axhline(0.5, color="#999", linestyle=":", linewidth=1, label="chance")
    ax.set_xlabel("held-out identities")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per fold", fontsize=10)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 2: mean with CI per level -- the accuracy question.
    ax = axes[1]
    for species, entry in report["species"].items():
        levels = [lv for lv in entry["levels"] if "ci_low" in lv]
        if not levels:
            continue
        xs = [lv["n_test_ids"] for lv in levels]
        means = [lv["mean"] for lv in levels]
        lo = [lv["mean"] - lv["ci_low"] for lv in levels]
        hi = [lv["ci_high"] - lv["mean"] for lv in levels]
        ax.errorbar(
            xs,
            means,
            yerr=[lo, hi],
            marker="o",
            capsize=3,
            color=SPECIES_COLOURS.get(species, "#666"),
            label=species,
        )
    ax.set_xlabel("held-out identities")
    ax.set_ylabel(f"mean {metric}")
    ax.set_title("mean ± 95% CI per level", fontsize=10)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 3: spread per level -- the stability question, the payoff.
    ax = axes[2]
    for species, entry in report["species"].items():
        levels = [lv for lv in entry["levels"] if np.isfinite(lv.get("std", np.nan))]
        if not levels:
            continue
        ax.plot(
            [lv["n_test_ids"] for lv in levels],
            [lv["std"] for lv in levels],
            marker="s",
            color=SPECIES_COLOURS.get(species, "#666"),
            label=species,
        )
    ax.set_xlabel("held-out identities")
    ax.set_ylabel(f"std of {metric} across folds")
    ax.set_title("stability: spread per level", fontsize=10)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Split-size sensitivity — {metric} over seeded identity-disjoint folds",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    fig_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "species": "both",
        "source": "video",
        "background": "all",
        "model": "all",
        "loss": "all",
    }
    path = fig_dir / build_filename(f"split_sensitivity_{metric}", cfg)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--metric", default="roc_auc")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument(
        "--per-cell",
        action="store_true",
        help="Also break the analysis down by (model, background) instead of pooling.",
    )
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"{args.input} not found — run scripts/kfold_matrix.sh then "
            "scripts/aggregate_kfold.py first"
        )
    df = pd.read_parquet(args.input)
    if args.metric not in df.columns:
        raise SystemExit(f"metric {args.metric!r} not in {args.input}")
    if df["n_test_ids"].isna().all():
        raise SystemExit(
            "no split.n_test_ids recorded — these runs predate the runtime re-split"
        )

    report = analyse(df, args.metric)
    report["input"] = str(args.input)
    report["n_runs_total"] = int(len(df))

    print(f"=== split-size sensitivity: {args.metric} ===")
    print(f"input: {args.input}  ({len(df)} runs)\n")

    for species, entry in report["species"].items():
        reg = entry["regression"]
        print(f"--- {species}  ({entry['n_runs']} runs)")
        print(
            f"    corr(n_test_ids, n_train_ids) = {entry['correlation_test_vs_train_ids']:+.3f}"
            "   (mechanically negative: fixed identity pool)"
        )
        print(
            f"    regression  {args.metric} ~ n_test_ids + n_train_ids:  "
            f"slope_test={reg['slope_n_test_ids']:+.4f}  "
            f"slope_train={reg['slope_n_train_ids']:+.4f}  R²={reg['r_squared']:.3f}"
        )
        print(f"    pooled std across all folds = {entry['pooled_std']:.4f}")
        print(
            f"    rank corr(level, spread) = {entry['std_vs_test_size_correlation']:+.3f}"
            "   (negative = larger test splits are more stable)"
        )
        print(f"    {'n_test':>6} {'folds':>6} {'mean':>8} {'std':>8} {'CI width':>9}")
        for level in entry["levels"]:
            std = "     n/a" if not np.isfinite(level.get("std", np.nan)) else f"{level['std']:8.4f}"
            width = (
                f"{level['ci_width']:9.4f}"
                if "ci_width" in level
                else "      n/a"
            )
            print(
                f"    {level['n_test_ids']:6d} {level['n_folds']:6d} "
                f"{level['mean']:8.4f} {std} {width}"
            )
        print()

    if args.per_cell:
        print("=== per (model, background) cell ===")
        report["cells"] = {}
        for (model, bg), group in df.groupby(["model", "bg"]):
            cell = analyse(group, args.metric)
            report["cells"][f"{model}|{bg}"] = cell["species"]
            for species, entry in cell["species"].items():
                print(
                    f"  {model:<9} {bg:<9} {species:<10} n={entry['n_runs']:2d} "
                    f"slope_test={entry['regression']['slope_n_test_ids']:+.4f} "
                    f"pooled_std={entry['pooled_std']:.4f}"
                )
        print()

    figure = plot(df, report, args.metric, args.fig_dir)
    report["figure"] = str(figure)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
