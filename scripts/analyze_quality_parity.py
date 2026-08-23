"""Test whether the two species' image datasets are comparable in quality.

Answers the reviewer's "is there an SNR-like measure showing no significant
difference between the species?" — and the honest answer requires two tests per
metric, not one:

* a **difference** test (Mann-Whitney + Cliff's delta + Hedges' g), which can
  only ever detect a difference; and
* an **equivalence** test (TOST against a declared margin), which is the only
  way to actively support "these datasets are the same for our purposes".

A non-significant difference test is *not* evidence of parity. A metric is only
reported as equivalent when TOST says so.

Unit of analysis
----------------
The **identity**, never the image. 520 vs 1092 images live inside 16 vs 12
individuals, and images from one bat come from one video, so per-image tests
would be pseudoreplication — they return p < 1e-30 for essentially any metric.
Each identity contributes its median, giving n = 16 vs n = 12.

Which rows to read
------------------
Sharpness and noise claims should be read off the **green** background, where
the flat background lets the battery restrict every metric to the face ROI. On
``original`` and ``random`` the metrics include background texture, so a
gradient measure partly reports scenery.

Usage:
    uv run python scripts/analyze_quality_parity.py
    uv run python scripts/analyze_quality_parity.py --margin-g 0.8
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
from bat_data.quality_metrics import BATTERY_METRICS  # noqa: E402
from bat_stats import (  # noqa: E402
    benjamini_hochberg,
    bootstrap_ci,
    build_filename,
    compare_groups,
    tost,
)

SPECIES_A = "mauritius"
SPECIES_B = "rousettus"

DEFAULT_BATTERY = Path("outputs/quality/battery.parquet")
DEFAULT_OUT = Path("outputs/quality/parity_report.json")
DEFAULT_FIG_DIR = Path("outputs/quality/figures")

# Metrics whose ratio is easier to read than their difference, and which the
# resolution story turns on. Ordered for the report.
PRIMARY_METRICS = (
    "native_side_px",
    "laplacian_var",
    "tenengrad",
    "gradient_energy",
    "noise_sigma_immerkaer",
    "noise_sigma_mad",
    "snr",
    "rms_contrast",
    "michelson_contrast",
    "mean_luminance",
    "shannon_entropy",
    "colorfulness",
)

# Human-readable grouping, so the report says what each block is evidence about.
METRIC_GROUPS = {
    "native_side_px": "effective resolution",
    "laplacian_var": "sharpness",
    "tenengrad": "sharpness",
    "gradient_energy": "sharpness",
    "noise_sigma_immerkaer": "noise",
    "noise_sigma_mad": "noise",
    "snr": "signal-to-noise",
    "rms_contrast": "contrast",
    "michelson_contrast": "contrast",
    "mean_luminance": "exposure",
    "shannon_entropy": "information",
    "colorfulness": "colour",
}


def species_cfg(background: str) -> dict[str, str]:
    """Naming cfg for a cross-species dataset figure.

    Every filename routes through ``bat_stats.naming`` (project invariant).
    These plots compare species and involve no trained model, so ``species`` is
    ``both`` and model/loss are ``none`` — degraded but honest and still
    self-identifying.
    """
    return {
        "species": "both",
        "source": "video",
        "background": background,
        "model": "none",
        "loss": "none",
    }


def per_identity_medians(df: pd.DataFrame, metric: str) -> dict[str, np.ndarray]:
    """Collapse each identity to its median *metric*, grouped by species."""
    out: dict[str, np.ndarray] = {}
    for species in (SPECIES_A, SPECIES_B):
        sub = df[df["species"] == species]
        if sub.empty or metric not in sub.columns:
            out[species] = np.asarray([], dtype=float)
            continue
        medians = sub.groupby("identity")[metric].median().dropna()
        out[species] = medians.to_numpy(dtype=float)
    return out


def analyse_metric(
    df: pd.DataFrame,
    metric: str,
    *,
    margin_g: float,
    alpha: float,
    n_boot: int,
) -> dict[str, Any] | None:
    groups = per_identity_medians(df, metric)
    a, b = groups[SPECIES_A], groups[SPECIES_B]
    if a.size < 2 or b.size < 2:
        return None

    comparison = compare_groups(a, b)
    equivalence = tost(a, b, margin=margin_g, margin_in_g_units=True, alpha=alpha)
    ci_a = bootstrap_ci(a, n_boot=n_boot)
    ci_b = bootstrap_ci(b, n_boot=n_boot)

    ratio = float(np.median(a) / np.median(b)) if abs(np.median(b)) > 1e-12 else float("nan")

    # A metric can be (a) different, (b) equivalent within margin, or (c)
    # inconclusive — under-powered to call either way. Naming (c) explicitly
    # stops it being silently read as parity.
    if comparison.p_value < alpha:
        verdict = "different"
    elif equivalence.equivalent:
        verdict = "equivalent"
    else:
        verdict = "inconclusive"

    return {
        "metric": metric,
        "group": METRIC_GROUPS.get(metric, "other"),
        "ratio_mauritius_over_rousettus": ratio,
        "verdict": verdict,
        "difference_test": comparison.to_dict(),
        "equivalence_test": equivalence.to_dict(),
        "bootstrap_mauritius": ci_a.to_dict(),
        "bootstrap_rousettus": ci_b.to_dict(),
    }


def plot_background(
    df: pd.DataFrame,
    background: str,
    results: list[dict[str, Any]],
    fig_dir: Path,
) -> Path:
    """One panel per metric: per-identity values for both species."""
    metrics = [r["metric"] for r in results]
    n = len(metrics)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.1 * rows))
    axes = np.atleast_1d(axes).ravel()

    colours = {SPECIES_A: "#A2582C", SPECIES_B: "#2F6B7A"}
    rng = np.random.default_rng(0)

    for ax, result in zip(axes, results):
        metric = result["metric"]
        groups = per_identity_medians(df, metric)
        for x, species in enumerate((SPECIES_A, SPECIES_B)):
            vals = groups[species]
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.08, 0.08, vals.size)
            ax.scatter(
                np.full(vals.size, x) + jitter,
                vals,
                s=26,
                alpha=0.8,
                color=colours[species],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.hlines(np.median(vals), x - 0.22, x + 0.22, color="black", linewidth=2, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["mauritius", "rousettus"], fontsize=8)
        verdict = result["verdict"]
        delta = result["difference_test"]["cliffs_delta"]
        ax.set_title(
            f"{metric}\n{verdict}  (delta={delta:+.2f}, {result['ratio_mauritius_over_rousettus']:.2f}x)",
            fontsize=9,
        )
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Per-identity quality metrics — background={background} "
        f"(n={df[df['species'] == SPECIES_A]['identity'].nunique()} vs "
        f"{df[df['species'] == SPECIES_B]['identity'].nunique()} identities)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / build_filename("quality_parity", species_cfg(background))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--battery", type=Path, default=DEFAULT_BATTERY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument(
        "--margin-g",
        type=float,
        default=0.5,
        help=(
            "TOST equivalence margin in pooled-SD (Hedges' g) units. "
            "0.5 = 'differences under half a between-identity SD do not matter'. "
            "Declare this before looking at the results."
        ),
    )
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()

    if not args.battery.exists():
        raise SystemExit(
            f"{args.battery} not found — run scripts/build_quality_battery.py first"
        )
    df = pd.read_parquet(args.battery)

    metrics = [m for m in PRIMARY_METRICS if m in df.columns]
    extra = [m for m in BATTERY_METRICS if m in df.columns and m not in metrics]
    metrics += extra

    report: dict[str, Any] = {
        "equivalence_margin_hedges_g": args.margin_g,
        "alpha": args.alpha,
        "unit_of_analysis": "identity (per-identity median)",
        "n_identities": {
            SPECIES_A: int(df[df["species"] == SPECIES_A]["identity"].nunique()),
            SPECIES_B: int(df[df["species"] == SPECIES_B]["identity"].nunique()),
        },
        "backgrounds": {},
    }

    print("=== quality parity: mauritius vs rousettus ===")
    print(f"unit of analysis : identity medians (n={report['n_identities']})")
    print(f"TOST margin      : |g| < {args.margin_g} (declared, not data-derived)")
    print(f"alpha            : {args.alpha}\n")

    for background in sorted(df["background"].unique()):
        sub = df[df["background"] == background]
        roi = "/".join(sorted(sub["roi_source"].dropna().unique()))

        results = []
        for metric in metrics:
            entry = analyse_metric(
                sub, metric, margin_g=args.margin_g, alpha=args.alpha, n_boot=args.n_boot
            )
            if entry is not None:
                results.append(entry)

        # Multiplicity: one difference test per metric within this background.
        q_values = benjamini_hochberg([r["difference_test"]["p_value"] for r in results])
        for entry, q in zip(results, q_values):
            entry["difference_test"]["p_value_bh"] = float(q)
            if entry["verdict"] == "different" and q >= args.alpha:
                entry["verdict"] = "different_uncorrected_only"

        figure = plot_background(sub, background, results, args.fig_dir)
        report["backgrounds"][background] = {
            "roi_source": roi,
            "n_metrics": len(results),
            "figure": str(figure),
            "metrics": results,
        }

        print(f"--- background={background}  (roi={roi})")
        header = (
            f"    {'metric':<22} {'group':<20} {'mau':>10} {'rou':>10} "
            f"{'ratio':>7} {'delta':>7} {'p_bh':>9} {'p_tost':>8}  verdict"
        )
        print(header)
        for entry in results:
            diff = entry["difference_test"]
            print(
                f"    {entry['metric']:<22} {entry['group']:<20} "
                f"{diff['median_a']:10.3f} {diff['median_b']:10.3f} "
                f"{entry['ratio_mauritius_over_rousettus']:7.2f} "
                f"{diff['cliffs_delta']:+7.2f} {diff['p_value_bh']:9.5f} "
                f"{entry['equivalence_test']['p_tost']:8.4f}  {entry['verdict']}"
            )
        print(f"    figure: {figure}\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")

    # Verdict tally, so the headline is not buried in the table.
    print("\n=== summary (green background = face-ROI-only, the valid comparison) ===")
    green = report["backgrounds"].get("green")
    if green:
        by_verdict: dict[str, list[str]] = {}
        for entry in green["metrics"]:
            by_verdict.setdefault(entry["verdict"], []).append(entry["metric"])
        for verdict in ("different", "different_uncorrected_only", "equivalent", "inconclusive"):
            if verdict in by_verdict:
                print(f"  {verdict:<28} {', '.join(by_verdict[verdict])}")


if __name__ == "__main__":
    main()
