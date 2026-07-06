"""Build the report's results table (LaTeX booktabs) + bar figure from the
multi-seed matrix CSV (outputs/multiseed_full_results.csv)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["Siamese", "ArcFace", "AdaFace"]
BGS = ["green", "original", "random"]
SPECIES = ["mauritius", "rousettus"]


def cell(df, sp, model, bg, metric):
    r = df[(df.species == sp) & (df.model == model) & (df.bg == bg)]
    if r.empty or not np.isfinite(r.iloc[0][f"{metric}_mean"]):
        return None
    return float(r.iloc[0][f"{metric}_mean"]), float(r.iloc[0][f"{metric}_std"])


def latex_table(df) -> str:
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{Mauritius} & \multicolumn{2}{c}{Rousettus} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Model & Bg & ROC-AUC & top-1 & ROC-AUC & top-1 \\",
        r"\midrule",
    ]

    def fmt(v):
        return "--" if v is None else f"{v[0]:.3f}\\,$\\pm$\\,{v[1]:.3f}"

    for model in MODELS:
        for bg in BGS:
            cells = [
                fmt(cell(df, "mauritius", model, bg, "roc_auc")),
                fmt(cell(df, "mauritius", model, bg, "top1")),
                fmt(cell(df, "rousettus", model, bg, "roc_auc")),
                fmt(cell(df, "rousettus", model, bg, "top1")),
            ]
            lines.append(f"{model} & {bg} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def bar_figure(df, dest: Path) -> None:
    colors = {"Siamese": "#4C78A8", "ArcFace": "#F58518", "AdaFace": "#54A24B"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    x = np.arange(len(BGS))
    w = 0.26
    for ax, sp in zip(axes, SPECIES, strict=True):
        for i, model in enumerate(MODELS):
            means = [(cell(df, sp, model, bg, "roc_auc") or (np.nan, 0))[0] for bg in BGS]
            stds = [(cell(df, sp, model, bg, "roc_auc") or (np.nan, 0))[1] for bg in BGS]
            ax.bar(
                x + (i - 1) * w, means, w, yerr=stds, capsize=3, label=model, color=colors[model]
            )
        ax.axhline(0.5, ls="--", lw=0.8, color="gray")
        ax.set_title(sp)
        ax.set_xticks(x)
        ax.set_xticklabels(BGS)
        ax.set_ylim(0.3, 1.0)
        ax.set_xlabel("background")
    axes[0].set_ylabel("test ROC-AUC (mean ± std, 5 seeds)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/multiseed_full_results.csv")
    ap.add_argument("--tex-out", default="reports/generated/results_table.tex")
    ap.add_argument("--fig-out", default="reports/figures/multiseed_matrix.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    Path(args.tex_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.fig_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tex_out).write_text(latex_table(df) + "\n")
    bar_figure(df, Path(args.fig_out))
    print(f"wrote {args.tex_out} and {args.fig_out}")


if __name__ == "__main__":
    main()
