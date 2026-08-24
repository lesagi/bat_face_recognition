"""Arm-aware analysis for the Phase 3 sweeps.

Why this is separate from analyze_species_significance.py
--------------------------------------------------------
That script groups on ``(model, bg)`` and skips any cell where one species has
fewer than two runs. Phase 3's arms are distinguished BY their ``bg`` label
(``green_blur``, ``green_band``, ``original_bgonly``, ``green_isect``, ...), so:

* 3A is mauritius-only -- its rousettus comparator carries a different label, so
  the generic script never pairs them and silently emits nothing;
* 3D is rousettus-only, same problem;
* 3C's decisive comparison is *within* a species *across* backgrounds
  (inpainted-original vs the inpainted-random control), which that script does
  not do at all -- it compares models within a fixed background.

3B is the one arm the generic script handles natively (same label, both species).
It is recomputed here anyway so one report covers everything.

Pairing rules, which differ per comparison and matter
-----------------------------------------------------
*Paired* (Nadeau-Bengio corrected t-test on per-fold differences) is valid only
when the two arms draw the SAME identity partition on a given fold: same manifest
identity set + same fold seed. That holds for
degraded-vs-baseline-within-mauritius, and for bgonly-original-vs-random.

*Unpaired* (Mann-Whitney + Cliff's delta) is required for every
mauritius-vs-rousettus comparison: the species have different identity pools (16
vs 12, or 10 vs 10 after band restriction), so fold N means an unrelated
partition on each side. This matches what analyze_species_significance.py does.

Every comparison is reported with effect size, a bootstrap CI, and the minimum
detectable difference. Per docs/PHASE_SUMMARY.md the MDE for this design is 0.440
ROC-AUC against a largest observed gap of 0.207, so a bare p-value from the
species arms is meaningless and is never reported alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bat_stats import (
    bootstrap_ci,
    benjamini_hochberg,
    cliffs_delta,
    compare_groups,
    corrected_resampled_ttest,
    hedges_g,
    interpret_delta,
)

PHASE3 = Path("outputs/phase3_results.parquet")
PUBLISHED = Path("outputs/kfold_results.parquet")
POWER = Path("outputs/kfold_power_ceiling.json")
DEFAULT_OUT = Path("outputs/phase3_analysis.json")
METRIC = "roc_auc"  # pre-declared primary, per Phase D


def _load(phase3: Path, published: Path) -> pd.DataFrame:
    a = pd.read_parquet(phase3)
    a["arm"] = "phase3"
    b = pd.read_parquet(published)
    b["arm"] = "published"
    keep = ["species", "model", "bg", "fold_id", METRIC, "n_train_imgs", "n_test_imgs", "arm"]
    return pd.concat([a[keep], b[keep]], ignore_index=True)


def _series(df: pd.DataFrame, species: str, model: str, bg: str) -> pd.Series:
    s = df[(df.species == species) & (df.model == model) & (df.bg == bg)]
    return s.set_index("fold_id")[METRIC].dropna().sort_index()


def paired(df, species, model, bg_a, bg_b, label) -> dict[str, Any] | None:
    """Same identity pool + same fold seeds => per-fold differences are meaningful."""
    a, b = _series(df, species, model, bg_a), _series(df, species, model, bg_b)
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return None
    d = (a.loc[common] - b.loc[common]).to_numpy(dtype=float)
    n_train = float(df[(df.species == species) & (df.bg == bg_a)].n_train_imgs.mean() or 1)
    n_test = float(df[(df.species == species) & (df.bg == bg_a)].n_test_imgs.mean() or 1)
    tt = corrected_resampled_ttest(d, n_train=int(n_train), n_test=int(n_test))
    ci = bootstrap_ci(d, statistic=np.mean)
    return {
        "comparison": label, "kind": "paired", "species": species, "model": model,
        "arm_a": bg_a, "arm_b": bg_b, "n_folds": int(len(common)),
        "mean_a": float(a.loc[common].mean()), "mean_b": float(b.loc[common].mean()),
        "mean_difference": float(d.mean()),
        "ci_low": ci.ci_low, "ci_high": ci.ci_high,
        "p_value_nadeau_bengio": tt.p_value, "correction": tt.correction,
        "cliffs_delta": float(cliffs_delta(a.loc[common].to_numpy(), b.loc[common].to_numpy())),
        "hedges_g": float(hedges_g(a.loc[common].to_numpy(), b.loc[common].to_numpy())),
    }


def unpaired(df, model, bg_mau, bg_rou, label) -> dict[str, Any] | None:
    """Different identity pools => fold N is an unrelated partition on each side."""
    a = _series(df, "mauritius", model, bg_mau).to_numpy(dtype=float)
    b = _series(df, "rousettus", model, bg_rou).to_numpy(dtype=float)
    if a.size < 3 or b.size < 3:
        return None
    r = compare_groups(a, b)
    ca, cb = bootstrap_ci(a, statistic=np.median), bootstrap_ci(b, statistic=np.median)
    return {
        "comparison": label, "kind": "unpaired", "model": model,
        "arm_mauritius": bg_mau, "arm_rousettus": bg_rou,
        "n_mauritius": int(a.size), "n_rousettus": int(b.size),
        "median_mauritius": r.median_a, "median_rousettus": r.median_b,
        "gap": float(r.median_a - r.median_b),
        "ci_mauritius": [ca.ci_low, ca.ci_high], "ci_rousettus": [cb.ci_low, cb.ci_high],
        "p_value": r.p_value, "cliffs_delta": r.cliffs_delta,
        "delta_magnitude": r.delta_magnitude, "hedges_g": r.hedges_g,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase3", type=Path, default=PHASE3)
    ap.add_argument("--published", type=Path, default=PUBLISHED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = _load(args.phase3, args.published)
    mde = json.loads(POWER.read_text())["mde_corrected"] if POWER.exists() else None
    models = ["ArcFace", "AdaFace"]
    report: dict[str, Any] = {
        "metric": METRIC,
        "mde_corrected_roc_auc": mde,
        "interpretation_rule": (
            "The minimum detectable difference for this design is "
            f"{mde} ROC-AUC. Comparisons whose effect is smaller than that CANNOT "
            "return significance regardless of the truth, so effect size + CI are "
            "the result and a bare p-value is not reportable."
        ),
        "pairing_rationale": {
            "paired": "same identity pool and fold seed (degraded-vs-baseline, bgonly original-vs-random)",
            "unpaired": "mauritius-vs-rousettus: different identity pools, so folds do not correspond",
        },
        "sections": {},
    }

    # A. Did degradation actually lower mauritius' score? (paired, within species)
    A = []
    for variant in ("blur", "recrop"):
        for base in ("green", "original", "random"):
            for m in models:
                r = paired(df, "mauritius", m, f"{base}_{variant}", base,
                           f"3A {variant}: mauritius {base} degraded vs baseline")
                if r:
                    r["variant"] = variant
                    A.append(r)
    report["sections"]["A_degradation_effect_within_mauritius"] = A

    # B. Species gap with quality controlled (unpaired). rousettus comparator is
    # green_isect for green (the intersection arm) and original otherwise.
    # rousettus comparators: green uses the 1059-image intersection arm (3D),
    # original and random use the published arms.
    ROU = (("green", "green_isect"), ("original", "original"), ("random", "random"))
    B = []
    for variant in ("blur", "recrop"):
        for base, rou in ROU:
            for m in models:
                r = unpaired(df, m, f"{base}_{variant}", rou,
                             f"3A {variant}: species gap, {base}, quality-controlled")
                if r:
                    r["variant"] = variant
                    B.append(r)
    # Baseline (uncontrolled) gap for reference.
    for base, rou in ROU:
        for m in models:
            r = unpaired(df, m, base, rou, f"baseline: species gap, {base}, UNCONTROLLED")
            if r:
                r["variant"] = "none"
                B.append(r)
    report["sections"]["B_species_gap_3A"] = B

    # C. 3B band-restricted species gap (unpaired, same label both species)
    C = [r for base in ("green", "original", "random") for m in models
         if (r := unpaired(df, m, f"{base}_band", f"{base}_band",
                           f"3B: species gap, {base}, band-restricted real pixels"))]
    report["sections"]["C_species_gap_3B"] = C

    # D. 3C leak test: inpainted original vs the inpainted-random negative control.
    D = [r for sp in ("mauritius", "rousettus")
         if (r := paired(df, sp, "ArcFace", "original_bgonly", "random_bgonly",
                         f"3C: {sp} background-only, original vs random control"))]
    report["sections"]["D_background_only_leak"] = D

    # E. 3D: rousettus green on the intersection vs the published green arm.
    E = [r for m in ["Siamese", "ArcFace", "AdaFace"]
         if (r := paired(df, "rousettus", m, "green_isect", "green",
                         f"3D: rousettus green intersection vs published green"))]
    report["sections"]["E_intersection_vs_published"] = E

    # BH across each family of tests separately -- they answer different questions.
    for name, rows in report["sections"].items():
        ps = [r.get("p_value", r.get("p_value_nadeau_bengio")) for r in rows]
        ps = [p for p in ps if p is not None]
        if len(ps) > 1:
            q = benjamini_hochberg(ps)
            for r, qq in zip([r for r in rows if r.get("p_value", r.get("p_value_nadeau_bengio")) is not None], q):
                r["p_bh_within_section"] = float(qq)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    # ---- printed summary
    print(f"=== Phase 3 analysis ({METRIC}) ===")
    print(f"MDE for this design: {mde} ROC-AUC -- effect size + CI are the result.\n")
    for name, rows in report["sections"].items():
        if not rows:
            print(f"--- {name}: no data yet"); continue
        print(f"--- {name}")
        for r in rows:
            if r["kind"] == "paired":
                print(f"  {r['comparison']:58s} {r['model']:8s} "
                      f"diff={r['mean_difference']:+.3f} CI[{r['ci_low']:+.3f},{r['ci_high']:+.3f}] "
                      f"d={r['cliffs_delta']:+.2f} p={r['p_value_nadeau_bengio']:.3f}")
            else:
                flag = "" if mde is None else ("  [< MDE: undetectable by design]"
                                              if abs(r["gap"]) < mde else "  [> MDE]")
                print(f"  {r['comparison']:58s} {r['model']:8s} "
                      f"gap={r['gap']:+.3f} mau={r['median_mauritius']:.3f} "
                      f"rou={r['median_rousettus']:.3f} d={r['cliffs_delta']:+.2f} "
                      f"({r['delta_magnitude']}){flag}")
        print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
