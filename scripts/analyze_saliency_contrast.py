"""Test the pre-registered eye/nose hypotheses on the trained-minus-untrained contrast.

Reads the per-fold outputs of ``scripts/species_saliency_maps.py`` and executes
exactly the analysis fixed in ``docs/saliency_preregistration.md``. Nothing here
chooses a metric, a test or a direction — those were fixed before the data
existed, and this script is deliberately not general.

Why the contrast rather than the trained numbers
------------------------------------------------
Saliency maps look persuasive for networks that have learned nothing, which is
why a randomisation control exists at all (Adebayo 2018). So the quantity of
interest is not "how much attention do the eyes get" but "how much does training
CHANGE it". On the original background the trained-only reading gave mauritius
82.8% of images peaking on an eye — but an untrained network scored 64.1%, so
most of that is a property of the photographs.

What is paired and what is not
------------------------------
*Paired* within a species: the same identities, images and ROIs scored under a
trained model and its own control, so between-bat variation cancels. Pairing is
on ``(identity, path)`` because pose detection runs separately per arm and can
drop different images.

*Unpaired* across species: mauritius and rousettus are different animals with
different identity pools, so there is nothing to pair.

*Across models*: every statistic is recomputed per fold and the reported interval
is a bootstrap over the folds. A single model's identity-level interval answers
"is this bat different from that bat", not "would another training run show the
same thing" — and the latter is what a reviewer will ask.
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
    compare_groups,
    hedges_g,
    paired_wilcoxon,
)

FOLD_DIR = pathlib.Path("outputs/saliency/folds")
DEFAULT_OUT = pathlib.Path("outputs/saliency/contrast_analysis.json")
REGIONS = ("eyes", "nose", "periphery")
SPECIES = ("mauritius", "rousettus")
TRAINED, CONTROL = "trained", "randomised_weights"

# Pre-registered. H2 (nose) is primary; periphery is the declared negative control.
HYPOTHESES = {"eyes": "H1", "nose": "H2 (primary)", "periphery": "negative control"}


def _per_identity(arm_block: dict[str, Any], region: str) -> pd.Series:
    rows = arm_block.get("per_image") or []
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    col = f"{region}_density"
    if col not in df:
        return pd.Series(dtype=float)
    return df.groupby("identity")[col].mean()


def _paired_frame(entry: dict[str, Any], species: str, region: str) -> pd.DataFrame:
    """Per-identity trained/untrained densities, joined on identity.

    The join is what enforces the pairing. If the two arms scored different
    identity sets the mismatch shows up here as a shorter frame rather than as a
    silent misalignment, and the caller refuses the fold.
    """
    arms = entry.get("arms", {})
    tr = arms.get(TRAINED, {}).get("species", {}).get(species)
    ct = arms.get(CONTROL, {}).get("species", {}).get(species)
    if not tr or not ct:
        return pd.DataFrame()
    a, b = _per_identity(tr, region), _per_identity(ct, region)
    if a.empty or b.empty:
        return pd.DataFrame()
    return pd.DataFrame({"trained": a, "untrained": b}).dropna()


def load_folds(fold_dir: pathlib.Path, background: str) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {sp: {} for sp in SPECIES}
    for path in sorted(fold_dir.glob(f"*_{background}_f*.json")):
        stem = path.stem
        species = stem.split(f"_{background}_f")[0]
        if species not in SPECIES:
            continue
        fold = int(stem.rsplit("_f", 1)[1])
        out[species][fold] = json.loads(path.read_text())
    return out


def analyse(background: str, fold_dir: pathlib.Path) -> dict[str, Any]:
    folds = load_folds(fold_dir, background)
    n_folds = {sp: len(v) for sp, v in folds.items()}
    report: dict[str, Any] = {
        "background": background,
        "n_folds": n_folds,
        "metric": "per-identity mean ROI density (attribution share / area share)",
        "contrast": "trained minus untrained, within identity",
        "preregistration": "docs/saliency_preregistration.md",
        "within_species": {},
        "across_species": {},
    }

    # --- per fold, per species, per region: the paired contrast
    contrasts: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    paired_rows: list[dict[str, Any]] = []
    for species in SPECIES:
        for region in REGIONS:
            per_fold: dict[int, np.ndarray] = {}
            stats_per_fold = []
            for fold, entry in sorted(folds[species].items()):
                frame = _paired_frame(entry, species, region)
                if frame.empty or len(frame) < 3:
                    continue
                per_fold[fold] = (frame["trained"] - frame["untrained"]).to_numpy()
                res = paired_wilcoxon(
                    frame["untrained"].to_numpy(), frame["trained"].to_numpy(),
                    alternative="two-sided",
                )
                stats_per_fold.append(res)
            if not per_fold:
                continue
            contrasts[(species, region)] = per_fold
            med = np.array([float(np.median(v)) for v in per_fold.values()])
            ci = bootstrap_ci(med, statistic=np.median) if med.size >= 3 else None
            paired_rows.append(
                {
                    "species": species,
                    "region": region,
                    "hypothesis": HYPOTHESES[region],
                    "n_folds": len(per_fold),
                    "n_identities": int(len(next(iter(per_fold.values())))),
                    "median_contrast_across_folds": float(np.median(med)),
                    "ci_low": ci.ci_low if ci else None,
                    "ci_high": ci.ci_high if ci else None,
                    # Median across folds of the per-fold paired p; the folds are
                    # not independent (same images, overlapping training data), so
                    # combining them into one p would overstate the evidence.
                    "median_paired_p": float(np.median([r.p_value for r in stats_per_fold])),
                    "median_rank_biserial": float(
                        np.median([r.matched_pairs_rank_biserial for r in stats_per_fold])
                    ),
                    "folds_with_negative_contrast": int((med < 0).sum()),
                    "folds_with_positive_contrast": int((med > 0).sum()),
                }
            )

    if paired_rows:
        q = benjamini_hochberg([r["median_paired_p"] for r in paired_rows])
        for row, qq in zip(paired_rows, q, strict=True):
            row["p_bh"] = float(qq)
    report["within_species"] = paired_rows

    # --- the hypotheses themselves: is the contrast larger for mauritius?
    across = []
    for region in REGIONS:
        mau = contrasts.get(("mauritius", region))
        rou = contrasts.get(("rousettus", region))
        if not mau or not rou:
            continue
        shared = sorted(set(mau) & set(rou))
        deltas, gaps = [], []
        for fold in shared:
            a, b = mau[fold], rou[fold]
            deltas.append(cliffs_delta(a, b))
            gaps.append(float(np.median(a) - np.median(b)))
        if not deltas:
            continue
        pooled = compare_groups(
            np.concatenate([mau[f] for f in shared]),
            np.concatenate([rou[f] for f in shared]),
        )
        d_ci = bootstrap_ci(np.array(deltas), statistic=np.median) if len(deltas) >= 3 else None
        g_ci = bootstrap_ci(np.array(gaps), statistic=np.median) if len(gaps) >= 3 else None
        across.append(
            {
                "region": region,
                "hypothesis": HYPOTHESES[region],
                "n_fold_pairs": len(shared),
                "median_cliffs_delta_across_folds": float(np.median(deltas)),
                "delta_ci_low": d_ci.ci_low if d_ci else None,
                "delta_ci_high": d_ci.ci_high if d_ci else None,
                "median_contrast_gap": float(np.median(gaps)),
                "gap_ci_low": g_ci.ci_low if g_ci else None,
                "gap_ci_high": g_ci.ci_high if g_ci else None,
                "folds_favouring_mauritius": int(sum(1 for d in deltas if d > 0)),
                # Pooling identities across folds ignores that the same bats
                # recur, so this p is anticonservative and is reported as a
                # descriptive companion to the fold-level interval, not as the
                # test. The interval is the evidence.
                "pooled_p_anticonservative": pooled.p_value,
                "pooled_hedges_g": float(
                    hedges_g(
                        np.concatenate([mau[f] for f in shared]),
                        np.concatenate([rou[f] for f in shared]),
                    )
                ),
            }
        )
    report["across_species"] = across
    return report


def _verdict(across: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered reading, including the negative control."""
    by = {r["region"]: r for r in across}
    peri = by.get("periphery")
    out: dict[str, Any] = {}
    for region in ("eyes", "nose"):
        r = by.get(region)
        if r is None:
            continue
        ci_excludes_zero = (
            r["delta_ci_low"] is not None and r["delta_ci_low"] > 0
        ) or (r["delta_ci_high"] is not None and r["delta_ci_high"] < 0)
        out[region] = {
            "hypothesis": HYPOTHESES[region],
            "direction_as_predicted": r["median_cliffs_delta_across_folds"] > 0,
            "ci_excludes_zero": bool(ci_excludes_zero),
            "supported": bool(r["median_cliffs_delta_across_folds"] > 0 and ci_excludes_zero),
        }
    if peri is not None:
        # The three regions partition the frame, so periphery MUST move opposite
        # to eyes/nose. Same-direction movement of comparable size means the
        # measurement is broken and nothing may be claimed.
        eyes_d = by.get("eyes", {}).get("median_cliffs_delta_across_folds", 0.0)
        peri_d = peri["median_cliffs_delta_across_folds"]
        broken = eyes_d != 0 and np.sign(peri_d) == np.sign(eyes_d) and abs(peri_d) >= abs(eyes_d)
        out["negative_control"] = {
            "periphery_delta": peri_d,
            "expected": "opposite sign to eyes/nose (the regions partition the frame)",
            "passes": bool(not broken),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold-dir", type=pathlib.Path, default=FOLD_DIR)
    ap.add_argument("--backgrounds", nargs="+", default=["green", "original"])
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    report: dict[str, Any] = {"backgrounds": {}}
    for bg in args.backgrounds:
        res = analyse(bg, args.fold_dir)
        res["verdict"] = _verdict(res["across_species"])
        report["backgrounds"][bg] = res

        print(f"\n=== {bg} background — {res['n_folds']} folds ===")
        if not res["within_species"]:
            print("  no fold outputs yet")
            continue
        print("\n  within species: trained minus untrained (per-identity density)")
        print(f"    {'species':11s} {'region':10s} {'contrast':>9s} {'95% CI':>20s} "
              f"{'q(BH)':>7s}  {'folds +/-':>9s}")
        for r in res["within_species"]:
            ci = (f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                  if r["ci_low"] is not None else "n/a")
            print(f"    {r['species']:11s} {r['region']:10s} "
                  f"{r['median_contrast_across_folds']:+9.3f} {ci:>20s} "
                  f"{r.get('p_bh', float('nan')):7.3f}  "
                  f"{r['folds_with_positive_contrast']:>4d}/{r['folds_with_negative_contrast']:<4d}")

        print("\n  the hypotheses: is the contrast larger for mauritius?")
        for r in res["across_species"]:
            ci = (f"[{r['delta_ci_low']:+.2f}, {r['delta_ci_high']:+.2f}]"
                  if r["delta_ci_low"] is not None else "n/a")
            print(f"    {r['region']:10s} {r['hypothesis']:18s} "
                  f"delta={r['median_cliffs_delta_across_folds']:+.2f} {ci:>16s}  "
                  f"favours mau in {r['folds_favouring_mauritius']}/{r['n_fold_pairs']} folds")
        v = res["verdict"]
        for region in ("eyes", "nose"):
            if region in v:
                print(f"    -> {HYPOTHESES[region]:18s} "
                      f"{'SUPPORTED' if v[region]['supported'] else 'not supported'}")
        if "negative_control" in v:
            print(f"    -> negative control (periphery): "
                  f"{'PASSES' if v['negative_control']['passes'] else 'FAILS — claim nothing'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
