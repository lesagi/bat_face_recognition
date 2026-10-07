"""Pair each rousettus bat with the mauritius bat most like it.

A species comparison over unmatched sets measures the datasets as much as the
animals. The project has already been bitten by exactly this: each bat was filmed
in one video, so anything constant within a clip is a free identity cue, and six
global colour numbers score 0.78 against trained models at 0.81. Pairing bat to bat
on capture conditions removes that at the level the comparison is actually made.

What can and cannot be matched by selection
-------------------------------------------
**Lighting: yes.** Face-minus-background contrast overlaps between the species --
rousettus spans -7.5..+31.9, mauritius +6.5..+85.5 -- so a mauritius bat can be
found for most rousettus bats.

**Resolution: no.** Native crop sides are rousettus 242..407 px and mauritius
441..738 px. Those ranges do not overlap *at all*, which is the same fact
`docs/quality_parity.md` records as Cliff's delta = 1.00. No choice of bats can
match resolution, so it must be handled by degrading mauritius -- which is what the
existing `blur` and `recrop` arms already do. This script therefore matches on
lighting and *reports* the residual resolution ratio per pair, so the downsampling
factor is known per pair rather than applied as one species-wide constant.

Method: optimal 1:1 assignment (Hungarian, `scipy.optimize.linear_sum_assignment`)
minimising total |contrast difference|, rather than greedy nearest-neighbour. Greedy
lets an early pair take the only good partner for a later one, and the pairing is
only as good as its worst member -- the whole point is that no pair is badly matched.

Candidates are restricted to bats with enough kept frames to be usable
(``--min-keeps``); a perfectly-lit bat with four frames is not a usable partner.

    uv run python scripts/pair_species_bats.py
    uv run python scripts/pair_species_bats.py --max-contrast-gap 8 --min-keeps 20
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

LIGHTING = pathlib.Path("outputs/quality/bat_lighting.json")
NATIVE_BOXES = pathlib.Path("outputs/quality/native_boxes.parquet")
DECISIONS = "data/curated/{sp}/decisions.json"
CROPS = "data/processed/{sp}/video/not_augmented/base_320/{bg}"
OUT_JSON = pathlib.Path("outputs/quality/species_pairs.json")


def crop_contrast(species: str, per_bat: int = 10) -> dict[str, float]:
    """Per-bat face-minus-background contrast, from the published 320px crops.

    Uses the green matte rather than re-running segmentation: the green crop *is*
    the project's definition of the face region, and it is what `probe_face_colour`
    measures, so the two agree by construction.
    """
    import cv2

    from bat_data.quality_metrics import face_mask_from_green

    orig = pathlib.Path(CROPS.format(sp=species, bg="original_bg"))
    grn = pathlib.Path(CROPS.format(sp=species, bg="green_bg"))
    if not orig.is_dir():
        return {}
    out: dict[str, float] = {}
    for idd in sorted(p for p in orig.iterdir() if p.is_dir()):
        ps = sorted(idd.glob("*.jpg"))
        if not ps:
            continue
        pick = [ps[i] for i in np.linspace(0, len(ps) - 1, min(per_bat, len(ps))).astype(int)]
        fs, bs = [], []
        for p in pick:
            im, gi = cv2.imread(str(p)), cv2.imread(str(grn / idd.name / p.name))
            if im is None or gi is None:
                continue
            m = face_mask_from_green(gi)
            if m is None or not m.any():
                continue
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            sel = m.astype(bool)
            fs.append(float(g[sel].mean()))
            bs.append(float(g[~sel].mean()))
        if fs:
            out[idd.name] = float(np.mean(fs) - np.mean(bs))
    return out


def keeps(species: str) -> dict[str, int]:
    """Kept frames per bat, or an empty dict when the species has no cull."""
    p = pathlib.Path(DECISIONS.format(sp=species))
    if not p.exists():
        return {}
    c: collections.Counter = collections.Counter()
    for v in json.loads(p.read_text())["frames"].values():
        if v["decision"] == "keep":
            c[v["identity"]] += 1
    return dict(c)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--min-keeps",
        type=int,
        default=15,
        help="Skip mauritius bats with fewer kept frames than this.",
    )
    ap.add_argument(
        "--max-contrast-gap",
        type=float,
        default=12.0,
        help="Reject a pair whose contrast differs by more than this.",
    )
    ap.add_argument("--per-bat", type=int, default=10)
    args = ap.parse_args()

    rc = crop_contrast("rousettus", args.per_bat)
    # Mauritius lighting comes from the curated tree when it exists (it covers the new
    # bats, which have no published crops yet), else from the crops.
    mc: dict[str, float] = {}
    if LIGHTING.exists():
        for b in json.loads(LIGHTING.read_text())["bats"]:
            if b.get("usable"):
                mc[b["identity"]] = float(b["contrast"])
    mc = mc or crop_contrast("mauritius", args.per_bat)
    if not rc or not mc:
        raise SystemExit("need contrast for both species; run rank_bats_by_lighting.py first")

    nb = pd.read_parquet(NATIVE_BOXES)
    res = {
        sp: nb[nb.species == sp].groupby("identity").native_side_px.median().to_dict()
        for sp in ("mauritius", "rousettus")
    }
    mk = keeps("mauritius")

    cand = [m for m in mc if mk.get(m, 0) >= args.min_keeps] if mk else list(mc)
    if not cand:
        raise SystemExit(f"no mauritius bat has >= {args.min_keeps} keeps yet")
    rous = sorted(rc)

    cost = np.array([[abs(rc[r] - mc[m]) for m in cand] for r in rous], dtype=float)
    ri, ci = linear_sum_assignment(cost)

    pairs, rejected = [], []
    for a, b in zip(ri, ci, strict=True):
        r, m = rous[a], cand[b]
        gap = float(cost[a, b])
        rr, mr = res["rousettus"].get(r), res["mauritius"].get(m)
        entry = {
            "rousettus": r,
            "mauritius": m,
            "contrast_rousettus": round(rc[r], 1),
            "contrast_mauritius": round(mc[m], 1),
            "contrast_gap": round(gap, 1),
            "keeps_mauritius": mk.get(m),
            "native_rousettus_px": None if rr is None else round(float(rr)),
            "native_mauritius_px": None if mr is None else round(float(mr)),
            "resolution_ratio": None if not (rr and mr) else round(float(mr / rr), 2),
        }
        (pairs if gap <= args.max_contrast_gap else rejected).append(entry)

    pairs.sort(key=lambda e: e["contrast_gap"])
    gaps = [p["contrast_gap"] for p in pairs]
    ratios = [p["resolution_ratio"] for p in pairs if p["resolution_ratio"]]
    out = {
        "method": "Hungarian 1:1 assignment minimising total |contrast difference|",
        "matched_on": "face-minus-background contrast",
        "not_matched_on": (
            "native resolution -- rousettus 242-407px and mauritius 441-738px do not "
            "overlap, so no selection can match it; use the blur/recrop arms"
        ),
        "min_keeps": args.min_keeps,
        "max_contrast_gap": args.max_contrast_gap,
        "n_pairs": len(pairs),
        "n_rejected": len(rejected),
        "contrast_gap": {
            "mean": round(float(np.mean(gaps)), 2) if gaps else None,
            "max": round(float(np.max(gaps)), 2) if gaps else None,
        },
        "resolution_ratio": {
            "mean": round(float(np.mean(ratios)), 2) if ratios else None,
            "min": round(float(np.min(ratios)), 2) if ratios else None,
            "max": round(float(np.max(ratios)), 2) if ratios else None,
        },
        "pairs": pairs,
        "rejected": rejected,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))

    print(
        f"\n  {len(pairs)} pair(s) within a {args.max_contrast_gap:.0f}-level contrast gap"
        f"   ({len(rejected)} rejected, {len(cand)} mauritius candidates "
        f"with >= {args.min_keeps} keeps)\n"
    )
    print(
        f"  {'rousettus':<14} {'mauritius':<18} {'contrast':>18} {'gap':>5} "
        f"{'native px':>14} {'ratio':>6}"
    )
    for p in pairs:
        # "unmeasured", not 0: native_boxes.parquet covers only the published bats, so
        # a freshly curated mauritius bat has no entry. Printing 0.00 would read as
        # "identical resolution", the opposite of the truth.
        mpx = "    -" if p["native_mauritius_px"] is None else f"{p['native_mauritius_px']:>5}"
        rat = "     -" if p["resolution_ratio"] is None else f"{p['resolution_ratio']:>6.2f}"
        print(
            f"  {p['rousettus']:<14} {p['mauritius']:<18} "
            f"{p['contrast_rousettus']:>+8.1f} vs {p['contrast_mauritius']:>+6.1f} "
            f"{p['contrast_gap']:>5.1f} "
            f"{p['native_rousettus_px'] or 0:>6} vs {mpx} {rat}"
        )
    if rejected:
        print(f"\n  rejected (gap > {args.max_contrast_gap:.0f}):")
        for p in rejected:
            print(
                f"    {p['rousettus']:<14} best was {p['mauritius']:<18} gap {p['contrast_gap']:.1f}"
            )
    n_unmeasured = sum(1 for p in pairs if p["resolution_ratio"] is None)
    if ratios:
        print(
            f"\n  residual resolution ratio (mauritius / rousettus), over the "
            f"{len(ratios)} pair(s) with both measured: "
            f"{min(ratios):.2f}-{max(ratios):.2f}, mean {np.mean(ratios):.2f}"
        )
        print("  -> not fixable by selection; the blur / recrop arms exist for this.")
    if n_unmeasured:
        print(
            f"  {n_unmeasured} pair(s) have no mauritius native size yet -- "
            "native_boxes.parquet covers only the published bats."
        )
        print("  Run scripts/measure_native_boxes.py once the arm is built.")
    print(f"\n  -> {OUT_JSON}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
