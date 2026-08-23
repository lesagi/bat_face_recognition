"""Is the species difference real? (reviewer comments 2 and 4)

Three separate questions, each needing a different test:

1. **Per species, is the result above chance?** (comment 4) Each fold already has
   its own permutation p-value. Combining 20 of them needs a combiner that
   tolerates dependence — folds share training data — so the default is the
   **harmonic-mean p** (Wilson 2019), not Fisher. Then Benjamini-Hochberg across
   the 18 species x model x background cells, because 18 simultaneous tests at
   alpha = 0.05 expect roughly one false positive by themselves.

2. **Does one model beat another, within a species?** Folds are shared, so the
   comparison is paired — but a plain paired t-test over 20 overlapping folds
   understates the variance badly. Uses the **Nadeau-Bengio corrected resampled
   t-test**.

3. **Do the two species differ?** (comment 2) This one *cannot* be paired: the
   species have different animals, so there is no fold to pair on. Uses an
   unpaired comparison plus Cliff's delta, with the fold as the unit.

The caveat that governs how question 3 may be reported: per ``docs/quality_parity.md``
the two datasets differ on 11 of 12 image-quality metrics and their effective
resolutions do not overlap at all (Cliff's delta = 1.00, 1.97x). A raw
species difference therefore measures "mauritius-dataset vs rousettus-dataset",
not "species". The script refuses to print a bare species verdict without
labelling it as confounded, and reports the matched-arm requirement alongside.

Usage:
    uv run python scripts/analyze_species_significance.py
    uv run python scripts/analyze_species_significance.py --combine fisher
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bat_stats import (
    benjamini_hochberg,
    combine_pvalues,
    compare_groups,
    corrected_resampled_ttest,
)

DEFAULT_INPUT = Path("outputs/kfold_results.parquet")
DEFAULT_OUT = Path("outputs/kfold_species_significance.json")

SPECIES_A = "mauritius"
SPECIES_B = "rousettus"

CONFOUND_NOTE = (
    "CONFOUNDED: the two datasets differ on 11/12 image-quality metrics and their "
    "effective resolutions do not overlap (Cliff's delta = 1.00, 1.97x). This tests "
    "mauritius-dataset vs rousettus-dataset, not species. See docs/quality_parity.md; "
    "a resolution-matched arm is required before attributing this to the animals."
)


def per_cell_permutation_summary(
    df: pd.DataFrame, *, method: str, alpha: float
) -> list[dict[str, Any]]:
    """Combine each cell's per-fold permutation p-values, then BH across cells."""
    if "perm_p" not in df.columns or df["perm_p"].isna().all():
        return []

    rows: list[dict[str, Any]] = []
    for (species, model, bg), group in df.groupby(["species", "model", "bg"]):
        p_values = group["perm_p"].dropna().to_numpy(dtype=float)
        if p_values.size == 0:
            continue
        # A p-value of exactly 0 is impossible from (Z+1)/(B+1); clip anyway so a
        # malformed input cannot make the combiner divide by zero.
        p_values = np.clip(p_values, 1e-12, 1.0)
        rows.append(
            {
                "species": species,
                "model": model,
                "bg": bg,
                "n_folds": int(p_values.size),
                "p_min": float(p_values.min()),
                "p_median": float(np.median(p_values)),
                "p_combined": combine_pvalues(p_values, method=method),  # type: ignore[arg-type]
                "combine_method": method,
            }
        )

    if rows:
        q_values = benjamini_hochberg([r["p_combined"] for r in rows])
        for row, q in zip(rows, q_values):
            row["p_combined_bh"] = float(q)
            row["significant"] = bool(q < alpha)
    return rows


def model_comparisons(df: pd.DataFrame, metric: str) -> list[dict[str, Any]]:
    """Paired model-vs-model comparisons within each (species, background)."""
    rows: list[dict[str, Any]] = []
    for (species, bg), group in df.groupby(["species", "bg"]):
        wide = group.pivot_table(index="fold_id", columns="model", values=metric)
        for model_a, model_b in combinations(sorted(wide.columns), 2):
            paired = wide[[model_a, model_b]].dropna()
            if len(paired) < 2:
                continue
            differences = (paired[model_a] - paired[model_b]).to_numpy(dtype=float)
            # Test/train sizes for the Nadeau-Bengio correction: use the mean
            # realised image counts for this cell.
            n_train = float(group["n_train_imgs"].dropna().mean() or 0)
            n_test = float(group["n_test_imgs"].dropna().mean() or 0)
            if n_train <= 0 or n_test <= 0:
                continue
            try:
                result = corrected_resampled_ttest(
                    differences, n_train=int(n_train), n_test=int(n_test)
                )
            except ValueError as exc:
                rows.append(
                    {
                        "species": species,
                        "bg": bg,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_folds": int(len(paired)),
                        "error": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "species": species,
                    "bg": bg,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_folds": int(len(paired)),
                    **result.to_dict(),
                }
            )
    return rows


def species_comparisons(df: pd.DataFrame, metric: str, *, alpha: float) -> list[dict[str, Any]]:
    """Unpaired species comparison per (model, background) cell."""
    rows: list[dict[str, Any]] = []
    for (model, bg), group in df.groupby(["model", "bg"]):
        a = group.loc[group["species"] == SPECIES_A, metric].dropna().to_numpy(dtype=float)
        b = group.loc[group["species"] == SPECIES_B, metric].dropna().to_numpy(dtype=float)
        if a.size < 2 or b.size < 2:
            continue
        result = compare_groups(a, b)
        rows.append(
            {
                "model": model,
                "bg": bg,
                "confound": CONFOUND_NOTE,
                **result.to_dict(),
            }
        )
    if rows:
        q_values = benjamini_hochberg([r["p_value"] for r in rows])
        for row, q in zip(rows, q_values):
            row["p_value_bh"] = float(q)
            row["significant_uncorrected_for_quality"] = bool(q < alpha)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--metric", default="roc_auc")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--combine",
        choices=("hmp", "fisher", "stouffer"),
        default="hmp",
        help="Combiner for per-fold permutation p-values. hmp tolerates dependence.",
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

    report: dict[str, Any] = {
        "input": str(args.input),
        "metric": args.metric,
        "alpha": args.alpha,
        "combine_method": args.combine,
        "n_runs": int(len(df)),
        "species_confound": CONFOUND_NOTE,
    }

    print(f"=== species significance: {args.metric} ===")
    print(f"input: {args.input} ({len(df)} runs)   combiner: {args.combine}\n")

    # --- 1. per-species above-chance (comment 4) ---
    perm = per_cell_permutation_summary(df, method=args.combine, alpha=args.alpha)
    report["per_cell_permutation"] = perm
    if perm:
        print("--- above chance, per species per cell (permutation p, combined across folds)")
        print(
            f"    {'species':<10} {'model':<9} {'bg':<9} {'folds':>5} "
            f"{'p_median':>9} {'p_comb':>9} {'p_BH':>9}  sig"
        )
        for row in perm:
            print(
                f"    {row['species']:<10} {row['model']:<9} {row['bg']:<9} "
                f"{row['n_folds']:5d} {row['p_median']:9.4f} {row['p_combined']:9.2e} "
                f"{row['p_combined_bh']:9.2e}  {'yes' if row['significant'] else 'no'}"
            )
        print()
    else:
        print("--- no per-fold permutation p-values in the table (perm_p column absent)\n")

    # --- 2. paired model comparisons ---
    models = model_comparisons(df, args.metric)
    report["model_comparisons"] = models
    if models:
        print("--- model vs model, within species+background (Nadeau-Bengio corrected)")
        print(
            f"    {'species':<10} {'bg':<9} {'A':<9} {'B':<9} {'folds':>5} "
            f"{'mean_diff':>10} {'t':>7} {'p':>8}"
        )
        for row in models:
            if "error" in row:
                print(
                    f"    {row['species']:<10} {row['bg']:<9} {row['model_a']:<9} "
                    f"{row['model_b']:<9} {row['n_folds']:5d}  skipped: {row['error']}"
                )
                continue
            print(
                f"    {row['species']:<10} {row['bg']:<9} {row['model_a']:<9} "
                f"{row['model_b']:<9} {row['n_folds']:5d} "
                f"{row['mean_difference']:+10.4f} {row['t_statistic']:7.2f} "
                f"{row['p_value']:8.4f}"
            )
        print()

    # --- 3. species comparison (comment 2) ---
    species = species_comparisons(df, args.metric, alpha=args.alpha)
    report["species_comparisons"] = species
    if species:
        print("--- mauritius vs rousettus, per cell (UNPAIRED; see confound below)")
        print(
            f"    {'model':<9} {'bg':<9} {'mau':>8} {'rou':>8} {'delta':>7} "
            f"{'p':>8} {'p_BH':>8}  magnitude"
        )
        for row in species:
            print(
                f"    {row['model']:<9} {row['bg']:<9} {row['median_a']:8.3f} "
                f"{row['median_b']:8.3f} {row['cliffs_delta']:+7.2f} "
                f"{row['p_value']:8.4f} {row['p_value_bh']:8.4f}  {row['delta_magnitude']}"
            )
        print()
        print("    !! " + CONFOUND_NOTE)
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
