"""How much identity leaks from colour alone, per identity-selection arm.

The point of selecting mauritius bats to match rousettus lighting is to shrink a
specific confound: each bat was filmed in exactly one video, so anything constant
within a clip -- exposure, white balance, lamp position -- is a free identity cue.
`docs/occlusion_results.md` measures that cue directly: six numbers (per-channel
mean and std inside the face matte, no spatial content at all) score 0.78 against
trained models at 0.81. This script asks whether a given choice of bats makes that
cue weaker.

**Why not one split.** Reading the probe off each arm's baked split gave 0.70 for
the paired arm and 0.9998 for a 12-bat random arm -- a difference that is mostly
noise, because a 12-identity arm holds out *two* bats and two bats either happen to
differ in colour or do not. The project already records this: a mauritius 2-identity
test split has a single-run ROC-AUC sd of about 0.12. So the comparison is run over
many seeded identity splits and reported as a distribution.

Descriptors are computed once per image and reused across splits; the cost is the
image read, not the resampling. Standardisation is fitted on each split's **train**
identities, never on test, so the number is not transductive.

    uv run python scripts/compare_arm_colour_leakage.py --arms paired lit12 lit all
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from bat_data import manifest_from_csv
from bat_stats import bootstrap_ci

MANIFESTS = pathlib.Path("data/manifests")
OUT_JSON = pathlib.Path("outputs/quality/arm_colour_leakage.json")


def descriptors(manifest_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Six-number face-colour descriptor per image, plus the identity of each."""
    import cv2

    from bat_data.quality_metrics import face_mask_from_green

    feats, ids = [], []
    for r in manifest_from_csv(manifest_path).records:
        img = cv2.imread(str(r.path))
        if img is None:
            continue
        m = face_mask_from_green(img)
        if m is None or not m.any():
            continue
        sel = m.astype(bool)
        feats.append([f(img[:, :, c][sel]) for c in range(3) for f in (np.mean, np.std)])
        ids.append(r.identity)
    return np.asarray(feats, dtype=float), np.asarray(ids)


def verification_auc(feat: np.ndarray, ids: np.ndarray) -> float | None:
    """ROC-AUC over all same/different identity pairs, by negative distance.

    The same task the models are scored on: given two images, same bat or not.
    """
    if len(np.unique(ids)) < 2 or len(feat) < 4:
        return None
    d = np.linalg.norm(feat[:, None, :] - feat[None, :, :], axis=2)
    iu = np.triu_indices(len(feat), k=1)
    dist, same = d[iu], (ids[:, None] == ids[None, :])[iu]
    if same.all() or not same.any():
        return None
    # Rank-based AUC: P(same-pair distance < different-pair distance).
    order = np.argsort(dist)
    ranks = np.empty(len(dist), dtype=float)
    ranks[order] = np.arange(1, len(dist) + 1)
    n_pos, n_neg = int(same.sum()), int((~same).sum())
    # The U statistic over ascending DISTANCE ranks gives P(same-pair distance >
    # different-pair distance). Verification ROC-AUC is the other orientation --
    # score = similarity, not distance -- so invert. Checked against the published
    # probe: rousettus comes out 0.816 here and 0.8168 from probe_face_colour.
    u = (ranks[same].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(1.0 - u)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", default="mauritius")
    ap.add_argument("--prefix", default="cull2")
    ap.add_argument("--arms", nargs="+", default=["paired", "lit12", "lit", "all"])
    ap.add_argument("--background", default="green_bg")
    ap.add_argument("--crop", default="head")
    ap.add_argument("--edge", type=int, default=224)
    ap.add_argument("--reference", default="data/manifests/rousettus_green_bg_manifest.csv")
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--test-ids", type=int, default=2, help="Identities held out per split.")
    args = ap.parse_args()

    jobs = {
        a: MANIFESTS
        / f"{args.species}_{args.prefix}_{a}_{args.background}_{args.crop}_{args.edge}_manifest.csv"
        for a in args.arms
    }
    jobs["rousettus"] = pathlib.Path(args.reference)

    results = {}
    for name, path in jobs.items():
        if not path.exists():
            print(f"  {name}: missing {path}")
            continue
        feat, ids = descriptors(path)
        uniq = np.unique(ids)
        if len(uniq) <= args.test_ids:
            print(f"  {name}: only {len(uniq)} identities")
            continue
        rng = np.random.default_rng(42)
        aucs = []
        for _ in range(args.n_splits):
            test = rng.choice(uniq, size=args.test_ids, replace=False)
            sel = np.isin(ids, test)
            # Standardise on train, apply to test: not transductive.
            mu, sd = feat[~sel].mean(0), feat[~sel].std(0)
            sd[sd < 1e-9] = 1.0
            a = verification_auc((feat[sel] - mu) / sd, ids[sel])
            if a is not None:
                aucs.append(a)
        if not aucs:
            continue
        v = np.asarray(aucs)
        boot = bootstrap_ci(v, statistic=np.median, n_boot=2000, seed=42)
        lo, hi = boot.ci_low, boot.ci_high
        results[name] = {
            "manifest": str(path),
            "n_identities": int(len(uniq)),
            "n_images": int(len(feat)),
            "n_splits": int(len(v)),
            "median": round(float(np.median(v)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p25": round(float(np.percentile(v, 25)), 4),
            "p75": round(float(np.percentile(v, 75)), 4),
            "frac_above_0_9": round(float((v > 0.9).mean()), 3),
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"test_ids": args.test_ids, "arms": results}, indent=1))

    print(
        f"\n  identity leakage from 6 colour numbers, {args.n_splits} seeded splits, "
        f"{args.test_ids} bats held out each time"
    )
    print("  0.5 = colour says nothing; higher = colour alone identifies the bat.")
    print(
        "  The rousettus value is the target: that is the leakage the species\n  comparison already lives with.\n"
    )
    print(
        f"  {'arm':<12} {'ids':>4} {'imgs':>6} {'median':>8} {'95% CI':>18} "
        f"{'IQR':>16} {'>0.9':>6}"
    )
    for n, r in results.items():
        print(
            f"  {n:<12} {r['n_identities']:>4} {r['n_images']:>6} {r['median']:>8.3f} "
            f"  [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] "
            f"  [{r['p25']:.3f}, {r['p75']:.3f}] {r['frac_above_0_9']:>6.0%}"
        )
    print(f"\n  -> {OUT_JSON}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
