"""What is the smallest difference this design could possibly detect?

Result 3 of the k-fold sweep was that no model family significantly beats
another. That is easy to misread as "the models are equivalent". It is not — it
is a statement about the experiment's resolving power, and this script quantifies
that separately from any particular comparison, so the two never get confused.

The **minimum detectable difference** (MDE) is the smallest true effect that
would be found significant 80% of the time. Anything smaller is invisible to the
design no matter how carefully it is analysed. Comparing the MDE against the
differences actually observed says whether "not significant" means "no effect" or
"we could never have seen it".

The pivotal quantity is the Nadeau-Bengio variance inflation,
``(1/n + n_test/n_train) / (1/n)``. Two things follow that are worth internalising:

* **More folds barely help.** With ``n_test/n_train`` around 0.5, the ``1/n``
  term is a twentieth of the total. Going from 20 folds to 100 changes almost
  nothing. The usual instinct — run more folds to gain power — is wrong here.
* **The test *fraction* is what matters.** Holding out fewer individuals shrinks
  the ratio and therefore the penalty. That pulls in the opposite direction from
  the stability question in ``analyze_split_sensitivity.py``, and the tension is
  real rather than a mistake: small test sets give noisier single estimates but
  more power to compare two models on the same folds.

Usage:
    uv run python scripts/analyze_power_ceiling.py
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_INPUT = Path("outputs/kfold_results.parquet")
DEFAULT_OUT = Path("outputs/kfold_power_ceiling.json")

ALPHA = 0.05
POWER = 0.80


def paired_difference_sds(df: pd.DataFrame) -> list[float]:
    """SD of per-fold model-vs-model differences, one per (species, bg, pair)."""
    sds: list[float] = []
    for _, group in df.groupby(["species", "bg"]):
        wide = group.pivot_table(index="fold_id", columns="model", values="roc_auc")
        for model_a, model_b in combinations(sorted(wide.columns), 2):
            diff = (wide[model_a] - wide[model_b]).dropna()
            if len(diff) > 1:
                sds.append(float(diff.std(ddof=1)))
    return sds


def mde(sd: float, n_folds: int, ratio: float, alpha: float, power: float) -> float:
    """Minimum detectable paired difference under Nadeau-Bengio."""
    if n_folds < 2:
        return float("nan")
    df = n_folds - 1
    se = sd * np.sqrt(1.0 / n_folds + ratio)
    return float((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * se)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--power", type=float, default=POWER)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"{args.input} not found — run scripts/aggregate_kfold.py first")
    df = pd.read_parquet(args.input)
    df["fold_id"] = df["fold_id"].astype(int)
    df["n_test_ids"] = df["n_test_ids"].astype(int)

    sds = paired_difference_sds(df)
    sd = float(np.median(sds))
    n_folds = int(df["fold_id"].nunique())
    ratio = float((df["n_test_imgs"] / df["n_train_imgs"]).mean())
    inflation = (1.0 / n_folds + ratio) / (1.0 / n_folds)

    naive = mde(sd, n_folds, 0.0, args.alpha, args.power)
    corrected = mde(sd, n_folds, ratio, args.alpha, args.power)

    observed = []
    for (species, bg), group in df.groupby(["species", "bg"]):
        wide = group.pivot_table(index="fold_id", columns="model", values="roc_auc")
        means = wide.mean()
        observed.append(
            {
                "species": species,
                "bg": bg,
                "gap": float(means.max() - means.min()),
                "best": str(means.idxmax()),
                "worst": str(means.idxmin()),
            }
        )
    largest = max(observed, key=lambda r: r["gap"])

    print("=== power ceiling for paired model comparisons ===")
    print(f"  folds                       : {n_folds}")
    print(f"  SD of per-fold differences  : {sd:.4f}   (median over {len(sds)} model pairs)")
    print(f"  n_test / n_train            : {ratio:.3f}")
    print(f"  Nadeau-Bengio inflation     : {inflation:.1f}x the naive variance")
    print(f"\n  MDE if folds were independent : {naive:.3f} ROC-AUC  (wrong, shown for contrast)")
    print(f"  MDE, corrected                : {corrected:.3f} ROC-AUC   <-- the real ceiling")
    print(
        f"\n  Largest gap actually observed : {largest['gap']:.3f} "
        f"({largest['species']}/{largest['bg']}, {largest['best']} vs {largest['worst']})"
    )
    verdict = (
        "every observed difference is below the ceiling — 'not significant' here means "
        "'undetectable by this design', NOT 'no difference'"
        if largest["gap"] < corrected
        else "some observed differences exceed the ceiling and could in principle be resolved"
    )
    print(f"  Verdict: {verdict}")

    # Isolate the ratio effect: hold folds at the real value so small-n noise in
    # the per-level fold counts cannot masquerade as a trend.
    print(f"\n=== how the ceiling depends on the held-out fraction (SD and folds held fixed) ===")
    print(f"  {'n_test_ids':>11}{'n_test/n_train':>16}{'inflation':>11}{'MDE':>8}")
    levels = []
    for level, group in df.groupby("n_test_ids"):
        lvl_ratio = float((group["n_test_imgs"] / group["n_train_imgs"]).mean())
        lvl_mde = mde(sd, n_folds, lvl_ratio, args.alpha, args.power)
        lvl_infl = (1.0 / n_folds + lvl_ratio) / (1.0 / n_folds)
        levels.append(
            {
                "n_test_ids": int(level),
                "ratio": lvl_ratio,
                "inflation": lvl_infl,
                "mde": lvl_mde,
                "n_runs": int(len(group)),
            }
        )
        print(f"  {level:>11}{lvl_ratio:>16.3f}{lvl_infl:>10.1f}x{lvl_mde:>8.3f}")
    print("\n  Fewer held-out bats -> smaller penalty -> more power to compare models.")
    print("  This opposes the stability argument; both are real, and they trade off.")

    report: dict[str, Any] = {
        "n_folds": n_folds,
        "sd_paired_difference": sd,
        "n_test_over_n_train": ratio,
        "nadeau_bengio_inflation": inflation,
        "alpha": args.alpha,
        "power": args.power,
        "mde_naive": naive,
        "mde_corrected": corrected,
        "largest_observed_gap": largest,
        "observed_gaps": observed,
        "by_test_size": levels,
        "interpretation": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
