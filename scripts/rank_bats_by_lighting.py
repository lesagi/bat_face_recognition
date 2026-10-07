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
# Crop margin, matching `build_variants_from_frames.py --margin`.
MARGIN = 0.03

# Reference: the 12 rousettus bats this project already trains on, each measured on
# its published 320px crops with the green face matte.
#
# The criterion is the observed RANGE, not a distance from the mean. Rousettus is
# not tightly lit -- its per-bat contrast runs -7.5 to +31.9 (median +11.1, sd 14.4)
# -- so a mauritius bat at +24 sits comfortably inside the reference population even
# though it is twice the rousettus median. Judging by distance-from-mean would call
# that a poor match while calling a bat at -8 a good one, which is backwards. A bat
# is a good match when its lighting is one the rousettus set itself exhibits.
ROUSETTUS_CONTRASTS = [-7.5, -3.2, -1.8, -0.2, 3.3, 6.1, 16.1, 18.8, 25.5, 25.9, 31.9, 31.9]
ROUSETTUS_LO = min(ROUSETTUS_CONTRASTS)
ROUSETTUS_HI = max(ROUSETTUS_CONTRASTS)
ROUSETTUS_MEDIAN = 11.1


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
            # Measure inside the HEAD CROP, not the whole frame. The model never sees
            # the frame; it sees the square the aligner cuts around the mask. Measured
            # on the same bats the two bases disagree wildly -- 20230805_132834 reads
            # -22.5 on the frame and +16.0 on the crop -- because the frame drags in
            # metres of unrelated scene. Same square as the aligner: the mask bbox's
            # larger side plus MARGIN on each edge.
            ys, xs = np.nonzero(sel)
            side = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * (1.0 + 2.0 * MARGIN)
            cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
            x0, x1 = int(max(0, cx - side / 2)), int(min(w, cx + side / 2))
            y0, y1 = int(max(0, cy - side / 2)), int(min(h, cy + side / 2))
            sub_sel = sel[y0:y1, x0:x1]
            if sub_sel.size == 0 or not sub_sel.any() or sub_sel.all():
                continue
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
            faces.append(float(grey[sub_sel].mean()))
            bgs.append(float(grey[~sub_sel].mean()))
        if not faces:
            rows.append({"identity": identity, "n_frames": 0, "usable": False})
            continue
        face, bg = float(np.mean(faces)), float(np.mean(bgs))
        contrast = face - bg
        # Distance outside the rousettus range; 0 means inside it.
        outside = max(0.0, ROUSETTUS_LO - contrast, contrast - ROUSETTUS_HI)
        rows.append(
            {
                "identity": identity,
                "n_frames": len(faces),
                "usable": True,
                "face": round(face, 1),
                "background": round(bg, 1),
                "contrast": round(contrast, 1),
                "outside_rousettus_range": round(outside, 1),
                "in_range": outside == 0.0,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", default="mauritius")
    ap.add_argument("--per-bat", type=int, default=10, help="Frames sampled per bat.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Slack beyond the rousettus range, in grey levels. 0 means a bat must "
        "land inside the range rousettus itself spans.",
    )
    args = ap.parse_args()

    rows = measure(args.species, args.per_bat, args.device)
    ok = [r for r in rows if r.get("usable")]
    ok.sort(key=lambda r: r["outside_rousettus_range"])
    for r in ok:
        r["lighting_match"] = "good" if r["outside_rousettus_range"] <= args.tolerance else "poor"
    for r in rows:
        if not r.get("usable"):
            r["lighting_match"] = "unknown"

    out = {
        "species": args.species,
        "reference": {
            "source": "12 rousettus bats, published 320px crops, green face matte",
            "contrast_range": [ROUSETTUS_LO, ROUSETTUS_HI],
            "contrast_median": ROUSETTUS_MEDIAN,
            "criterion": "good = contrast inside the range rousettus itself spans",
        },
        "tolerance": args.tolerance,
        "frames_per_bat": args.per_bat,
        "n_good": sum(1 for r in ok if r["lighting_match"] == "good"),
        "n_poor": sum(1 for r in ok if r["lighting_match"] == "poor"),
        "bats": ok + [r for r in rows if not r.get("usable")],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))

    print(
        f"\n  reference: rousettus spans {ROUSETTUS_LO:+.1f} to {ROUSETTUS_HI:+.1f} "
        f"(median {ROUSETTUS_MEDIAN:+.1f}) -- good = inside that range"
    )
    print("  measured inside the head crop, not the whole frame")
    print(f"  {out['n_good']} good / {out['n_poor']} poor\n")
    print(f"  {'bat':<18} {'face':>6} {'bg':>6} {'contrast':>9} {'outside':>8}  match")
    for r in ok:
        print(
            f"  {r['identity']:<18} {r['face']:>6.1f} {r['background']:>6.1f} "
            f"{r['contrast']:>+9.1f} {r['outside_rousettus_range']:>8.1f}  {r['lighting_match']}"
        )
    print(f"\n  -> {OUT_JSON}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
