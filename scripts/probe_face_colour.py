"""How much of the `green` result is reachable from six global colour numbers?

Sibling of ``scripts/probe_background_leakage.py`` and deliberately built the same
way: identical per-fold split resolution, identical pair construction, identical
metric — only the descriptor differs. That one swaps the model for a colour
histogram of the *background*; this one swaps it for the mean and standard
deviation of the *face*.

Why this exists
---------------
The ROI-occlusion arm (`docs/occlusion_results.md`) failed its positive control:
destroying 76% of every mauritius image did not reduce recognition. The diagnosis
was that the signal is not localised — six numbers with no spatial content
identify a bat at ~0.78, against trained models at ~0.81. This script turns that
observation into a per-fold, paired measurement with an interval, which is the
form the claim has to take before it can be used.

The descriptor, fixed now
-------------------------
Per-channel **mean and standard deviation** of the pixels inside the green-screen
face matte. Six numbers. No spatial information of any kind, so no localised
ablation can remove it, and no crop, rotation or reflection changes it.

Stated plainly: this descriptor was chosen while diagnosing the occlusion result,
not in advance. Variants (mean only, with white balance, whole crop rather than
face) are **not** explored here — picking the best of several would be the
best-of-N defect this project has already corrected once. Grey-world white
balancing was tried during the diagnosis and left the cue intact
(0.777 -> 0.778), which is recorded in `docs/occlusion_results.md`.

The reading, declared before running
------------------------------------
The quantity of interest is the **paired per-fold margin**, model minus
descriptor, on the same test pairs.

* margin CI **excludes zero and positive** -> the models use something beyond
  global face colour, and the size of the margin is how much.
* margin CI **contains zero** -> on this data the models are not demonstrably
  better than six numbers, and no claim of the form "the network recognises
  facial features" is supportable without further evidence.

Either way the margin is reported with its interval, never a bare p-value, and
against the design's minimum detectable difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from bat_cli.runtime import (  # noqa: E402
    compose_config,
    find_project_root,
    load_manifest,
    resplit_manifest,
)
from bat_data.quality_metrics import face_mask_from_green  # noqa: E402
from bat_stats import bootstrap_ci, corrected_resampled_ttest  # noqa: E402

SPECIES = ("mauritius", "rousettus")
# The published green arm, so the comparison is against the project's headline
# numbers rather than against something rebuilt for this script.
EXPERIMENT = "arcface_{sp}_green_bg_video_tuned"
MODEL_RESULTS = Path("outputs/kfold_results.parquet")
POWER = Path("outputs/kfold_power_ceiling.json")
DEFAULT_OUT = Path("outputs/leakage/face_colour_probe.json")
MIN_FACE_PIXELS = 200


def face_colour_descriptor(image_path: Path) -> np.ndarray | None:
    """Six numbers: per-channel mean and std inside the face matte."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    face = face_mask_from_green(img)
    if face is None or int(face.sum()) < MIN_FACE_PIXELS:
        return None
    px = img[face].astype(np.float64)
    return np.concatenate([px.mean(axis=0) / 255.0, px.std(axis=0) / 255.0])


def probe_fold(species: str, fold: int, root: Path) -> dict[str, Any] | None:
    """Descriptor ROC-AUC over the same test pairs the model was scored on."""
    cfg = compose_config(
        experiment=EXPERIMENT.format(sp=species),
        overrides=(
            f"data.fold_id={fold}",
            "data.resplit=true",
            "data.split_size_mode=seeded",
        ),
    )
    manifest = load_manifest(cfg, root=root)
    manifest, info = resplit_manifest(manifest, cfg)
    if info is None:
        return None

    def collect(split: str) -> tuple[np.ndarray, list[str], int]:
        vals, ids, skipped = [], [], 0
        for record in manifest.filter_split(split):
            d = face_colour_descriptor(Path(record.path))
            if d is None:
                skipped += 1
                continue
            vals.append(d)
            ids.append(record.identity)
        return (np.stack(vals) if vals else np.empty((0, 6))), ids, skipped

    F, identities, n_skipped = collect("test")
    train_F, _, _ = collect("train")
    if len(F) < 4:
        return None

    # Standardise so no channel dominates the distance purely by having a larger
    # range. Fitted on the TRAIN split, not the test split: the model sees the
    # train split too, so this keeps the comparison fair. (Fitting on test is
    # transductive and would hand the descriptor information the model never got.
    # Measured both ways -- train-fitted is actually slightly *better*, 0.849 vs
    # 0.831 on mauritius, so the fair version is not the flattering one.)
    ref = train_F if len(train_F) >= 2 else F
    F = (F - ref.mean(axis=0)) / (ref.std(axis=0) + 1e-12)

    # Identical pair construction to predictions_from_embedding.
    i, j = np.triu_indices(len(identities), k=1)
    ids = np.asarray(identities, dtype=object)
    labels = (ids[i] == ids[j]).astype(int)
    if labels.sum() in (0, labels.size):
        return None
    scores = -np.linalg.norm(F[i] - F[j], axis=1)

    return {
        "species": species,
        "fold": fold,
        "n_images": int(F.shape[0]),
        "n_identities": int(len(set(identities))),
        "n_pairs": int(labels.size),
        "n_skipped": n_skipped,
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--species", choices=(*SPECIES, "all"), default="all")
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(20)))
    ap.add_argument("--model-results", type=Path, default=MODEL_RESULTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    root = find_project_root(None)
    species = SPECIES if args.species == "all" else (args.species,)
    mde = json.loads(POWER.read_text())["mde_corrected"] if POWER.exists() else None

    print("=== six-number face-colour baseline, per fold, green background ===")
    print("descriptor: per-channel mean + std inside the face matte. No spatial content.\n")

    per_fold = [
        r for sp in species for f in args.folds if (r := probe_fold(sp, f, root)) is not None
    ]
    probe = pd.DataFrame(per_fold)

    models = pd.read_parquet(args.model_results)
    models = models[(models.bg == "green")]

    report: dict[str, Any] = {
        "descriptor": "per-channel mean+std inside the green face matte (6 numbers)",
        "standardisation": "fitted on the train split (not transductive)",
        "background": "green",
        "model_results": str(args.model_results),
        "mde_corrected_roc_auc": mde,
        "per_fold": per_fold,
        "descriptor_summary": {},
        "paired_margins": [],
    }

    print("  descriptor alone (median over folds)")
    for sp in species:
        s = probe[probe.species == sp].roc_auc
        if s.empty:
            continue
        report["descriptor_summary"][sp] = {
            "n_folds": int(s.size), "median": float(s.median()),
            "mean": float(s.mean()), "sd": float(s.std(ddof=1)),
        }
        print(f"    {sp:11s} n={s.size:2d}  {s.median():.3f}  (mean {s.mean():.3f} ± {s.std(ddof=1):.3f})")

    print("\n  paired margin: trained model minus descriptor, same folds and pairs")
    print(f"    {'species':11s} {'model':9s} {'model':>7s} {'probe':>7s} {'margin':>8s} "
          f"{'95% CI':>18s} {'p':>7s} {'beats':>7s}")
    rows = []
    for sp in species:
        p = probe[probe.species == sp].set_index("fold").roc_auc
        for model in ("Siamese", "ArcFace", "AdaFace"):
            m = models[(models.species == sp) & (models.model == model)]
            if m.empty:
                continue
            m = m.set_index("fold_id").roc_auc.dropna()
            common = sorted(set(p.index) & set(m.index))
            if len(common) < 3:
                continue
            margin = (m.loc[common] - p.loc[common]).to_numpy()
            n_tr = float(models[(models.species == sp)].n_train_imgs.dropna().mean() or 1)
            n_te = float(models[(models.species == sp)].n_test_imgs.dropna().mean() or 1)
            tt = corrected_resampled_ttest(margin, n_train=int(n_tr), n_test=int(n_te))
            ci = bootstrap_ci(margin, statistic=np.median)
            row = {
                "species": sp, "model": model, "n_folds": len(common),
                "model_median": float(m.loc[common].median()),
                "probe_median": float(p.loc[common].median()),
                "median_margin": float(np.median(margin)),
                "ci_low": ci.ci_low, "ci_high": ci.ci_high,
                "p_value_nadeau_bengio": tt.p_value,
                "folds_model_wins": int((margin > 0).sum()),
                "ci_excludes_zero": bool(ci.ci_low > 0 or ci.ci_high < 0),
            }
            rows.append(row)
            print(f"    {sp:11s} {model:9s} {row['model_median']:7.3f} {row['probe_median']:7.3f} "
                  f"{row['median_margin']:+8.3f} [{ci.ci_low:+.3f}, {ci.ci_high:+.3f}] "
                  f"{tt.p_value:7.3f} {row['folds_model_wins']:>4d}/{len(common)}")
    report["paired_margins"] = rows

    print("\n  reading (declared before running)")
    for r in rows:
        if r["ci_excludes_zero"] and r["median_margin"] > 0:
            verdict = "model adds something beyond global colour"
        elif r["ci_excludes_zero"] and r["median_margin"] < 0:
            verdict = "model is WORSE than six numbers"
        else:
            verdict = "model NOT demonstrably better than six numbers"
        r["verdict"] = verdict
        print(f"    {r['species']:11s} {r['model']:9s} -> {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
