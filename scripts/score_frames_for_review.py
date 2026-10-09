"""Order review frames by how likely you are to keep them.

Not to decide them. Measured against 5,655 frames already labelled by hand, every
prefill strategy is worse than a blank page:

    prefill                marked keep   corrections per 100 frames
    blank                           0%                         14.6
    model, top 10%                 10%                         16.8
    model, top 15%                 15%                         18.2
    the automatic frontal gate     30%                         31.7

The keep rate is 14.6%, so marking everything "drop" and clicking only the keeps is
already near-optimal; anything that guesses keeps creates more corrections than it
saves. The gate is worse still -- of the frames it passes, 78% get dropped by hand
(it agrees with the reviewer on only 68% overall, and its held-out-by-bat ROC-AUC
against the hand labels is 0.589, barely above chance).

So the model is used for **ordering**, where it cannot introduce a wrong decision --
only change which frames you see first. A gradient-boosted tree on the recorded pose
and mask geometry reaches held-out-by-bat ROC-AUC 0.794, so sorting by its score puts
most keeps near the top of the page and the reviewer can stop scanning once they dry
up.

Validation is **identity-disjoint** (GroupKFold on bat), as everywhere else in this
project: frames from one bat are near-duplicates of each other, so a random split
would report a number that has nothing to do with performance on a new bat.

Writes `outputs/quality/review_scores.json`, which `make_review_sheets.py` reads if
present. Re-run it after each batch of reviewing to fold in the new labels.

    uv run python scripts/score_frames_for_review.py
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

CURATED = pathlib.Path("data/curated")
OUT = pathlib.Path("outputs/quality/review_scores.json")
# Everything `build_frontal_dataset.py --keep-all` records per frame. Pose geometry
# plus one mask statistic -- the same evidence a human has at a glance.
FEATURES = [
    "eye_angle", "interocular", "nose_x_off", "nose_drop",
    "le_c", "re_c", "no_c", "mask_area_pct",
]
# Only decisions a human actually made. `seed:` rows are inherited from the legacy
# cull and were never ruled on in this round.
HUMAN = ("html:", "browser:")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", default="mauritius")
    ap.add_argument("--min-labelled", type=int, default=300)
    args = ap.parse_args()

    root = CURATED / args.species
    dec = json.loads((root / "decisions.json").read_text())
    met = json.loads((root / "metrics.json").read_text())

    def feats(key: str) -> list[float] | None:
        m = met.get(key)
        if not m or any(m.get(f) is None for f in FEATURES):
            return None
        return [float(m[f]) for f in FEATURES]

    X, y, groups, gate = [], [], [], []
    for key, rec in dec["frames"].items():
        if not rec.get("source", "").startswith(HUMAN) or rec["decision"] not in ("keep", "drop"):
            continue
        f = feats(key)
        if f is None:
            continue
        X.append(f)
        y.append(1 if rec["decision"] == "keep" else 0)
        groups.append(rec["identity"])
        gate.append(1 if met[key].get("auto_keep") else 0)
    X, y, groups, gate = np.array(X), np.array(y), np.array(groups), np.array(gate)
    if len(y) < args.min_labelled or len(set(groups)) < 3:
        raise SystemExit(f"only {len(y)} labelled frames over {len(set(groups))} bats; too few")

    n_splits = min(5, len(set(groups)))
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-9] = 1.0
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit((X[tr] - mu) / sd, y[tr])
        oof[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    auc_model = float(roc_auc_score(y, oof))
    auc_gate = float(roc_auc_score(y, gate))

    # Final model on everything, to score the unreviewed frames.
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit((X - mu) / sd, y)

    scores = {}
    for key in dec["frames"]:
        f = feats(key)
        if f is not None:
            scores[key] = round(float(clf.predict_proba(((np.array(f) - mu) / sd)[None])[0, 1]), 4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "species": args.species,
        "n_labelled": int(len(y)),
        "n_bats_labelled": int(len(set(groups))),
        "keep_rate": round(float(y.mean()), 4),
        "features": FEATURES,
        "roc_auc_identity_disjoint": round(auc_model, 4),
        "roc_auc_gate": round(auc_gate, 4),
        "note": "scores order the review page; they never decide a frame",
        "scores": scores,
    }, indent=1))

    print(f"\n  trained on {len(y)} hand-labelled frames over {len(set(groups))} bats "
          f"(keep rate {y.mean():.1%})")
    print(f"  held-out-by-bat ROC-AUC:  model {auc_model:.3f}   vs the gate {auc_gate:.3f}")
    print(f"  scored {len(scores)} frames -> {OUT}")
    print("  the sheets will now show most-likely-keep first\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
