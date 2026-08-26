"""Does recognition depend on the eye/nose discs, or on the periphery?

Executes `docs/occlusion_preregistration.md` (with Amendment 1) and nothing else:
the metric, the test, the equivalence margin, the positive control and the
decision table were all fixed before the datasets were built.

The question. The pre-registered saliency test found that bat-specific training
moves attribution away from the eye/nose discs and toward the periphery in 38 of
40 models. Attribution can be wrong about causal dependence, so the claim is
converted into an ablation and the pixels are actually destroyed.

The design problem is area, and it is why `matched` exists. The eye+nose discs are
~24% of the crop and the periphery is ~76%, so comparing "remove the discs"
against "remove the periphery" would mostly measure how many pixels were
destroyed. `matched` destroys an identical pixel count, on the animal, furthest
from the discs. **`centre` vs `matched` is the primary comparison; `periphery` is
a positive control, not a hypothesis.**

Why H3 is an equivalence test. H3 predicts *no* difference, and a non-significant
ordinary test would only mean underpowered -- this design's minimum detectable
difference is 0.440 ROC-AUC. TOST at |Hedges' g| < 0.5 can actively support
"these are the same", which a difference test cannot.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from bat_stats import (
    benjamini_hochberg,
    bootstrap_ci,
    cliffs_delta,
    corrected_resampled_ttest,
    hedges_g,
    tost,
)

ARMS = ("none", "centre", "matched", "periphery")
BG = {a: f"green_occ_{a}" for a in ARMS}
SPECIES = ("mauritius", "rousettus")
DEFAULT_IN = pathlib.Path("outputs/occlusion/roi_occlusion_results.parquet")
DEFAULT_OUT = pathlib.Path("outputs/occlusion/roi_occlusion_analysis.json")
POWER = pathlib.Path("outputs/kfold_power_ceiling.json")

# All pre-registered.
TOST_MARGIN_G = 0.5
POSITIVE_CONTROL_MIN_DROP = 0.05
METRIC = "roc_auc"


def _series(df: pd.DataFrame, species: str, arm: str) -> pd.Series:
    s = df[(df.species == species) & (df.bg == BG[arm])]
    return s.set_index("fold_id")[METRIC].dropna().sort_index()


def _drops(df: pd.DataFrame, species: str, arm: str) -> pd.Series:
    """Per-fold cost of the ablation. Positive means the ablation hurt.

    Paired by fold: the arms share manifest identities and fold seed, so fold N
    holds out the same bats in both.
    """
    base, abl = _series(df, species, "none"), _series(df, species, arm)
    common = base.index.intersection(abl.index)
    return (base.loc[common] - abl.loc[common]).rename(arm)


def analyse(df: pd.DataFrame) -> dict[str, Any]:
    mde = json.loads(POWER.read_text())["mde_corrected"] if POWER.exists() else None
    report: dict[str, Any] = {
        "metric": METRIC,
        "preregistration": "docs/occlusion_preregistration.md",
        "mde_corrected_roc_auc": mde,
        "tost_margin_hedges_g": TOST_MARGIN_G,
        "positive_control_min_drop": POSITIVE_CONTROL_MIN_DROP,
        "arm_means": {},
        "ablation_cost": [],
        "primary_h3": [],
        "h4": [],
        "positive_control": [],
    }

    for species in SPECIES:
        report["arm_means"][species] = {
            arm: (
                {
                    "n_folds": int(_series(df, species, arm).size),
                    "median": float(_series(df, species, arm).median()),
                    "mean": float(_series(df, species, arm).mean()),
                    "sd": float(_series(df, species, arm).std(ddof=1)),
                }
                if _series(df, species, arm).size
                else None
            )
            for arm in ARMS
        }

    # --- what each ablation costs, paired against `none`
    rows = []
    for species in SPECIES:
        n_train = float(df[df.species == species].n_train_imgs.dropna().mean() or 1)
        n_test = float(df[df.species == species].n_test_imgs.dropna().mean() or 1)
        for arm in ("centre", "matched", "periphery"):
            d = _drops(df, species, arm)
            if d.size < 3:
                continue
            tt = corrected_resampled_ttest(d.to_numpy(), n_train=int(n_train), n_test=int(n_test))
            ci = bootstrap_ci(d.to_numpy(), statistic=np.median)
            rows.append(
                {
                    "species": species, "arm": arm, "n_folds": int(d.size),
                    "median_drop": float(np.median(d)), "mean_drop": float(d.mean()),
                    "ci_low": ci.ci_low, "ci_high": ci.ci_high,
                    "p_value_nadeau_bengio": tt.p_value,
                    "folds_where_ablation_hurt": int((d > 0).sum()),
                }
            )
    if rows:
        q = benjamini_hochberg([r["p_value_nadeau_bengio"] for r in rows])
        for r, qq in zip(rows, q, strict=True):
            r["p_bh"] = float(qq)
    report["ablation_cost"] = rows

    # --- positive control: destroying 76% must cost something, or nothing is claimable
    for species in SPECIES:
        d = _drops(df, species, "periphery")
        if d.size < 3:
            continue
        med = float(np.median(d))
        report["positive_control"].append(
            {
                "species": species, "median_drop": med,
                "required_min_drop": POSITIVE_CONTROL_MIN_DROP,
                "passes": bool(med >= POSITIVE_CONTROL_MIN_DROP),
            }
        )

    # --- H3 (primary): is the cost of losing the discs EQUIVALENT to losing a
    #     matched area of periphery?
    for species in SPECIES:
        dc, dm = _drops(df, species, "centre"), _drops(df, species, "matched")
        common = dc.index.intersection(dm.index)
        if len(common) < 3:
            continue
        a, b = dc.loc[common].to_numpy(), dm.loc[common].to_numpy()
        eq = tost(a, b, margin=TOST_MARGIN_G, margin_in_g_units=True)
        n_train = float(df[df.species == species].n_train_imgs.dropna().mean() or 1)
        n_test = float(df[df.species == species].n_test_imgs.dropna().mean() or 1)
        diff = corrected_resampled_ttest(a - b, n_train=int(n_train), n_test=int(n_test))
        ci = bootstrap_ci(a - b, statistic=np.median)
        equivalent, significant = bool(eq.equivalent), bool(diff.p_value < 0.05)
        verdict = (
            "H3 supported — the central features are not special"
            if equivalent and not significant
            else "H3 refuted — the discs cost more than matched periphery"
            if significant and not equivalent
            else "inconclusive — underpowered, neither equivalent nor different"
        )
        report["primary_h3"].append(
            {
                "species": species, "n_folds": int(len(common)),
                "median_drop_centre": float(np.median(a)),
                "median_drop_matched": float(np.median(b)),
                "median_difference": float(np.median(a - b)),
                "ci_low": ci.ci_low, "ci_high": ci.ci_high,
                "tost_equivalent": equivalent, "tost_p": eq.p_tost,
                "difference_p_nadeau_bengio": diff.p_value, "difference_significant": significant,
                "cliffs_delta": float(cliffs_delta(a, b)), "hedges_g": float(hedges_g(a, b)),
                "verdict": verdict,
            }
        )

    # --- H4: are the discs alone insufficient?
    for species in SPECIES:
        dp, dc = _drops(df, species, "periphery"), _drops(df, species, "centre")
        common = dp.index.intersection(dc.index)
        if len(common) < 3:
            continue
        a, b = dp.loc[common].to_numpy(), dc.loc[common].to_numpy()
        n_train = float(df[df.species == species].n_train_imgs.dropna().mean() or 1)
        n_test = float(df[df.species == species].n_test_imgs.dropna().mean() or 1)
        tt = corrected_resampled_ttest(a - b, n_train=int(n_train), n_test=int(n_test))
        ci = bootstrap_ci(a - b, statistic=np.median)
        report["h4"].append(
            {
                "species": species, "n_folds": int(len(common)),
                "median_extra_cost_of_keeping_only_discs": float(np.median(a - b)),
                "ci_low": ci.ci_low, "ci_high": ci.ci_high,
                "p_value_nadeau_bengio": tt.p_value,
                "cliffs_delta": float(cliffs_delta(a, b)),
            }
        )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=pathlib.Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    rep = analyse(df)

    print(f"=== ROI occlusion ({METRIC}) ===")
    print(f"MDE for this design: {rep['mde_corrected_roc_auc']} — hence TOST for H3, not a null test.\n")
    print("  arm means (median ROC-AUC over folds)")
    for sp, arms in rep["arm_means"].items():
        cells = "  ".join(
            f"{a}={v['median']:.3f}(n={v['n_folds']})" for a, v in arms.items() if v
        )
        print(f"    {sp:11s} {cells}")

    print("\n  what each ablation costs (paired by fold, positive = it hurt)")
    print(f"    {'species':11s} {'arm':10s} {'drop':>7s} {'95% CI':>18s} {'q(BH)':>7s} {'hurt in':>8s}")
    for r in rep["ablation_cost"]:
        print(f"    {r['species']:11s} {r['arm']:10s} {r['median_drop']:+7.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] {r.get('p_bh', float('nan')):7.3f} "
              f"{r['folds_where_ablation_hurt']:>5d}/{r['n_folds']}")

    print("\n  positive control — destroying 76% must cost >= "
          f"{POSITIVE_CONTROL_MIN_DROP}")
    all_pass = True
    for r in rep["positive_control"]:
        all_pass &= r["passes"]
        print(f"    {r['species']:11s} drop={r['median_drop']:+.3f}  "
              f"{'PASSES' if r['passes'] else 'FAILS — claim nothing'}")

    print("\n  H3 (primary) — is losing the discs equivalent to losing a matched area?")
    for r in rep["primary_h3"]:
        print(f"    {r['species']:11s} centre={r['median_drop_centre']:+.3f} "
              f"matched={r['median_drop_matched']:+.3f} diff={r['median_difference']:+.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
        print(f"    {'':11s} TOST equivalent={r['tost_equivalent']} (p={r['tost_p']:.3f})  "
              f"difference p={r['difference_p_nadeau_bengio']:.3f}  g={r['hedges_g']:+.2f}")
        print(f"    {'':11s} -> {r['verdict']}")
    if not all_pass:
        print("\n  !! positive control failed — H3/H4 are not interpretable.")

    print("\n  H4 — extra cost of keeping ONLY the discs (periphery minus centre)")
    for r in rep["h4"]:
        print(f"    {r['species']:11s} {r['median_extra_cost_of_keeping_only_discs']:+.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  p={r['p_value_nadeau_bengio']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=float) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
