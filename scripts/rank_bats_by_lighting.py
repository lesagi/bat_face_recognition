"""Score each curated bat on how closely its lighting matches the rousettus set.

Why this exists. The two species' footage was shot under different light, and that
is a confound the project has already been bitten by: one video per bat means
anything constant within a clip is a free identity cue, and six global colour
numbers already score 0.78 against trained models at 0.81. Choosing mauritius bats
whose lighting resembles rousettus narrows that gap at the source, before any
model sees the data.

The intuitive rule -- "pick the ones filmed after dark" -- is **backwards**, which
is the finding that prompted this script. Mauritius night footage was shot with a
lamp, so the face is blown out while the background stays dim:

    rousettus          face 137  background 127  contrast  +10   (flat, even)
    mauritius night    face 184  background 126  contrast  +59   (lamp on the face)
    mauritius day      face 119  background  95  contrast  +24

Backgrounds are nearly identical (126 vs 127); it is the *face* that is over-lit.
So daylight mauritius is the closer match, and time of day is the wrong selector --
measured contrast is the right one.

Method: sample frames per bat, segment the face, and record mean grey inside and
outside the matte. Lighting is a property of the clip (one video per bat, constant
exposure), so a handful of frames estimates it well; ``--per-bat`` trades runtime
for precision. Frames are sampled across keep/unreviewed/dropped alike, since the
cull says nothing about exposure.

Ranking is by |contrast - rousettus median contrast|. Contrast rather than absolute
brightness because it is what differs: a face/background difference of +59 against
+10 is a different *lighting setup*, while overall level is partly gain.

    uv run python scripts/rank_bats_by_lighting.py --species mauritius
"""

from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

from bat_preprocessing import YOLOSegmenter

CURATED = pathlib.Path("data/curated")
OUT_JSON = pathlib.Path("outputs/quality/bat_lighting.json")
SEG_WEIGHTS = {
    "mauritius": "models/preprocessing/face_seg_mauritius_v2.pt",
    "rousettus": "legacy/rousesttus_segmentation/training_results/runs/segment/bat_face_seg/weights/best.pt",
}
STATE_DIRS = ("keep", "unreviewed", "dropped")
# Reference: the rousettus set this project already trains on. Measured on its
# published 320px crops with the same face matte.
ROUSETTUS_CONTRAST = 10.4
ROUSETTUS_FACE = 137.4
ROUSETTUS_BG = 127.0


def frames_for(species: str, identity: str, per_bat: int) -> list[pathlib.Path]:
    """Evenly spaced frames for one bat, drawn from every decision state."""
    found: list[pathlib.Path] = []
    for state in STATE_DIRS:
        d = CURATED / species / state / identity
        if d.is_dir():
            found.extend(sorted(d.glob("*.jpg")))
    if len(found) <= per_bat:
        return found
    idx = np.linspace(0, len(found) - 1, per_bat).round().astype(int)
    return [found[i] for i in sorted(set(idx))]


def measure(species: str, per_bat: int, device: str) -> list[dict]:
    root = CURATED / species
    identities = sorted(
        {
            d.name
            for s in STATE_DIRS
            if (root / s).is_dir()
            for d in (root / s).iterdir()
            if d.is_dir()
        }
    )
    if not identities:
        raise SystemExit(f"no curated identities under {root}")
    seg = YOLOSegmenter({"weights": SEG_WEIGHTS[species], "device": device})

    rows = []
    for identity in identities:
        faces, bgs = [], []
        for p in frames_for(species, identity, per_bat):
            img = cv2.imread(str(p))
            if img is None:
                continue
            s = seg.predict(img)
            if s is None or s.mask is None:
                continue
            h, w = img.shape[:2]
            m = s.mask
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            sel = m > 0
            if not sel.any() or sel.all():
                continue
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces.append(float(grey[sel].mean()))
            bgs.append(float(grey[~sel].mean()))
        if not faces:
            rows.append({"identity": identity, "n_frames": 0, "usable": False})
            continue
        face, bg = float(np.mean(faces)), float(np.mean(bgs))
        rows.append(
            {
                "identity": identity,
                "n_frames": len(faces),
                "usable": True,
                "face": round(face, 1),
                "background": round(bg, 1),
                "contrast": round(face - bg, 1),
                "distance_to_rousettus": round(abs(face - bg - ROUSETTUS_CONTRAST), 1),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", default="mauritius")
    ap.add_argument("--per-bat", type=int, default=10, help="Frames sampled per bat.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--good-threshold",
        type=float,
        default=20.0,
        help="A bat is a 'good' lighting match when |contrast - rousettus| is at most this.",
    )
    args = ap.parse_args()

    rows = measure(args.species, args.per_bat, args.device)
    ok = [r for r in rows if r.get("usable")]
    ok.sort(key=lambda r: r["distance_to_rousettus"])
    for r in ok:
        r["lighting_match"] = (
            "good" if r["distance_to_rousettus"] <= args.good_threshold else "poor"
        )
    for r in rows:
        if not r.get("usable"):
            r["lighting_match"] = "unknown"

    out = {
        "species": args.species,
        "reference": {
            "source": "rousettus published 320px crops",
            "face": ROUSETTUS_FACE,
            "background": ROUSETTUS_BG,
            "contrast": ROUSETTUS_CONTRAST,
        },
        "good_threshold": args.good_threshold,
        "frames_per_bat": args.per_bat,
        "n_good": sum(1 for r in ok if r["lighting_match"] == "good"),
        "n_poor": sum(1 for r in ok if r["lighting_match"] == "poor"),
        "bats": ok + [r for r in rows if not r.get("usable")],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))

    print(
        f"\n  reference: rousettus contrast {ROUSETTUS_CONTRAST:+.1f} "
        f"(face {ROUSETTUS_FACE}, background {ROUSETTUS_BG})"
    )
    print(
        f"  {out['n_good']} good / {out['n_poor']} poor "
        f"(threshold: within {args.good_threshold:.0f} of the reference)\n"
    )
    print(f"  {'bat':<18} {'face':>6} {'bg':>6} {'contrast':>9} {'dist':>6}  match")
    for r in ok:
        print(
            f"  {r['identity']:<18} {r['face']:>6.1f} {r['background']:>6.1f} "
            f"{r['contrast']:>+9.1f} {r['distance_to_rousettus']:>6.1f}  {r['lighting_match']}"
        )
    print(f"\n  -> {OUT_JSON}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
