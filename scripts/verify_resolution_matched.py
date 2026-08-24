"""Acceptance check + review artifacts for the Phase 2A degraded-mauritius arms.

The build is only trustworthy if it can be checked, so this module measures the
degraded sets with the *same* battery used on the originals and reports how far
each metric moved toward rousettus.

Only ``gradient_energy`` is targeted by the blur solve. ``laplacian_var``,
``tenengrad`` and ``noise_sigma_immerkaer`` are therefore free consistency
checks: if they converge without being aimed at, the degradation is doing
something real. If they trade off, that is reported rather than patched.

Writes everything to ``outputs/phase2a_review/`` for later inspection.
"""

from __future__ import annotations

import json
import pathlib
from multiprocessing import Pool
from typing import Any

import cv2
import numpy as np
import pandas as pd

from bat_stats import cliffs_delta, hedges_g, interpret_delta, tost
from bat_stats.naming import build_filename

from bat_data.quality_metrics import compute_battery  # noqa: I001

REVIEW_DIR = pathlib.Path("outputs/phase2a_review")
FIG_DIR = REVIEW_DIR / "figures"
NATIVE_BOXES = pathlib.Path("outputs/quality/native_boxes.parquet")
METRICS = ("gradient_energy", "laplacian_var", "tenengrad", "noise_sigma_immerkaer")
TARGETED = "gradient_energy"
NEGLIGIBLE_DELTA = 0.147  # Romano 2006 threshold used throughout bat_stats
N_WORKERS = 20

SETS = {
    "mauritius_baseline": "data/manifests/mauritius_green_bg_manifest.csv",
    "mauritius_blur": "data/manifests/mauritius_blur_green_manifest.csv",
    "mauritius_recrop": "data/manifests/mauritius_recrop_green_manifest.csv",
    "rousettus_reference": "data/manifests/rousettus_green_bg_intersect_manifest.csv",
}


def _measure(a: tuple[str, str]) -> dict[str, Any] | None:
    path, identity = a
    m = compute_battery(path)
    if m is None:
        return None
    m.update(path=path, identity=identity, basename=pathlib.Path(path).name)
    return m


def measure_set(manifest_csv: str) -> pd.DataFrame | None:
    p = pathlib.Path(manifest_csv)
    if not p.exists():
        return None
    d = pd.read_csv(p)
    cv2.setNumThreads(1)
    with Pool(N_WORKERS) as pool:
        rows = [r for r in pool.map(_measure, list(zip(d.path, d.identity)), chunksize=16) if r]
    return pd.DataFrame(rows)


def compare(deg: pd.DataFrame, ref: pd.DataFrame) -> dict[str, Any]:
    """Per-identity-median comparison of one set against the rousettus reference.

    Identity medians, not per-image values: identities are the inference unit and
    the published parity analysis uses the same unit, so the numbers stay
    comparable to docs/quality_parity.md.
    """
    a = deg.groupby("identity").median(numeric_only=True)
    b = ref.groupby("identity").median(numeric_only=True)
    out: dict[str, Any] = {}
    for m in METRICS:
        x, y = a[m].to_numpy(), b[m].to_numpy()
        d = cliffs_delta(x, y)
        eq = tost(x, y, margin=0.5, margin_in_g_units=True)
        out[m] = {
            "median_degraded": float(np.median(x)),
            "median_reference": float(np.median(y)),
            "ratio": float(np.median(x) / np.median(y)) if np.median(y) else None,
            "cliffs_delta": float(d),
            "delta_magnitude": interpret_delta(d),
            "hedges_g": float(hedges_g(x, y)),
            "tost_equivalent": bool(eq.equivalent),
            "tost_p": float(eq.p_tost),
            "targeted": m == TARGETED,
            "negligible": bool(abs(d) < NEGLIGIBLE_DELTA),
        }
    return out


def montage(sets: dict[str, pd.DataFrame], per_identity: int = 1) -> list[str]:
    """One row per identity: baseline | each degraded variant | a reference crop.

    This is the artifact to actually look at -- the numbers can agree while the
    images are visibly wrong.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = [k for k in ("mauritius_baseline", "mauritius_blur", "mauritius_recrop") if k in sets]
    ref = sets.get("rousettus_reference")
    base = sets["mauritius_baseline"]
    written: list[str] = []
    rows = []
    for ident in sorted(base.identity.unique()):
        names = base[base.identity == ident].basename.sort_values().tolist()[:per_identity]
        for name in names:
            tiles = []
            for k in order:
                sub = sets[k][sets[k].basename == name]
                if sub.empty:
                    continue
                img = cv2.imread(sub.path.iloc[0])
                if img is not None:
                    tiles.append(cv2.resize(img, (224, 224)))
            if ref is not None and len(ref):
                r = cv2.imread(ref.path.iloc[len(rows) % len(ref)])
                if r is not None:
                    tiles.append(cv2.resize(r, (224, 224)))
            if tiles:
                rows.append(np.hstack(tiles))
    if rows:
        cfg = {"species": "mauritius", "source": "video", "background": "degraded",
               "model": "none", "loss": "none"}
        fn = FIG_DIR / build_filename("quality_match_montage", cfg)
        cv2.imwrite(str(fn), np.vstack(rows))
        written.append(str(fn))
        print(f"    montage: {fn}  ({len(rows)} rows, columns = {order + ['rousettus_reference']})")
    return written


def verify(variant: str) -> dict[str, Any]:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== Phase 2A acceptance check ===")
    sets = {}
    for name, man in SETS.items():
        df = measure_set(man)
        if df is not None:
            sets[name] = df
            print(f"  measured {name:22s} n={len(df):5d} ids={df.identity.nunique():3d}")
    if "rousettus_reference" not in sets:
        raise SystemExit("rousettus reference manifest missing")

    native = pd.read_parquet(NATIVE_BOXES)[["basename", "species", "native_side_px"]]
    report: dict[str, Any] = {
        "targeted_metric": TARGETED,
        "free_checks": [m for m in METRICS if m != TARGETED],
        "negligible_delta_threshold": NEGLIGIBLE_DELTA,
        "unit_of_analysis": "per-identity medians",
        "reference": SETS["rousettus_reference"],
        "arms": {},
    }
    ref = sets["rousettus_reference"]
    for name in ("mauritius_baseline", "mauritius_blur", "mauritius_recrop"):
        if name in sets:
            report["arms"][name] = compare(sets[name], ref)

    # native_side_px: unchanged by blur (same crops), changed by recrop.
    nat = {}
    for sp, key in (("mauritius", "mauritius_baseline"), ("rousettus", "rousettus_reference")):
        if key in sets:
            j = sets[key].merge(native[native.species == sp], on="basename", how="left")
            nat[sp] = float(j.groupby("identity").native_side_px.median().median())
    report["native_side_px_identity_median"] = nat

    print(f"\n  {'arm / metric':46s} {'ratio':>7s} {'delta':>7s} {'g':>7s} {'verdict':>12s}")
    for arm, res in report["arms"].items():
        for m, r in res.items():
            tag = "TARGETED" if r["targeted"] else "free"
            verdict = "negligible" if r["negligible"] else r["delta_magnitude"]
            print(f"  {arm+' / '+m:46s} {r['ratio']:7.2f} {r['cliffs_delta']:+7.2f} "
                  f"{r['hedges_g']:+7.2f} {verdict:>12s}  [{tag}]")

    # Acceptance verdict, stated rather than inferred.
    for arm in [a for a in report["arms"] if a != "mauritius_baseline"]:
        t = report["arms"][arm][TARGETED]
        others = [m for m in METRICS if m != TARGETED]
        conv = [m for m in others if report["arms"][arm][m]["negligible"]]
        report["arms"][arm]["_acceptance"] = {
            "targeted_negligible": t["negligible"],
            "free_checks_negligible": conv,
            "n_free_converged": len(conv),
        }
        print(f"\n  {arm}: targeted {TARGETED} "
              f"{'REACHED' if t['negligible'] else 'NOT reached'} (delta={t['cliffs_delta']:+.2f}); "
              f"{len(conv)}/{len(others)} free checks also negligible"
              + (f" ({', '.join(conv)})" if conv else ""))

    montage(sets)
    (REVIEW_DIR / "battery_comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    # factors.parquet: per-image record of what was done to each image.
    fac = sets["mauritius_baseline"][["basename", "identity", "path"] + list(METRICS)].copy()
    fac = fac.merge(native[native.species == "mauritius"][["basename", "native_side_px"]],
                    on="basename", how="left")
    solve = REVIEW_DIR / "blur_solve.parquet"
    if solve.exists():
        s = pd.read_parquet(solve)
        s["basename"] = s.path.map(lambda p: pathlib.Path(p).name)
        fac = fac.merge(s[["basename", "sigma", "target", "ge_before", "ge_after",
                           "reachable", "no_blur_needed"]], on="basename", how="left")
    rc = REVIEW_DIR / "recrop_factors.parquet"
    if rc.exists():
        fac = fac.merge(pd.read_parquet(rc), on="basename", how="left", suffixes=("", "_recrop"))
    fac.to_parquet(REVIEW_DIR / "factors.parquet")
    print(f"    factors: {REVIEW_DIR/'factors.parquet'}  ({len(fac)} rows, {len(fac.columns)} cols)")
    print(f"    report:  {REVIEW_DIR/'battery_comparison.json'}")
    return report


if __name__ == "__main__":
    import sys
    verify(sys.argv[1] if len(sys.argv) > 1 else "blur")
