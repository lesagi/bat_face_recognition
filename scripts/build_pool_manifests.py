"""Manifests for the expanded mauritius pool and its session-matched subset.

Two arms, one image set (`pool40`), so nothing is duplicated on disk:

``pool40``       all 40 identities. The point is the identity count. The design's
                 minimum detectable difference (0.440 ROC-AUC) is driven by
                 having only 12-16 individuals, and this is the only route in
                 this project to improving it.
``session0831``  the 14 identities filmed on 2023-08-31, a subset of the same
                 images. Mauritius's answer to rousettus, which is 12 bats in a
                 single 9-minute window: it tests whether the six-number colour
                 baseline survives when session variation is largely removed.

Splits are drawn from the identity set, so the three background variants of an
arm share one partition and differ only in what sits behind the bat.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import pandas as pd

from bat_core.types import ImageRecord, Manifest
from bat_data import build_manifest, manifest_to_csv
from bat_data.splitter import IdentitySplitter

ROOT = pathlib.Path("data/processed/mauritius/video/not_augmented/pool40")
BACKGROUNDS = ("green", "original", "random")
SESSION_DAY = "20230831"
OUT_JSON = pathlib.Path("outputs/quality/pool40_manifests.json")

# 40 identities allows a much smaller held-out fraction than 16 did while still
# leaving plenty of test pairs -- which is exactly where the power comes from.
BOUNDS = {
    "pool40": dict(val_fraction=0.15, test_fraction=0.15, min_val=3, max_val=6,
                   min_test=4, max_test=10, min_train=20),
    "session0831": dict(val_fraction=0.20, test_fraction=0.20, min_val=2, max_val=3,
                        min_test=2, max_test=5, min_train=7),
}


def _records(bg: str) -> list[ImageRecord]:
    src = ROOT / f"{bg}_bg"
    m = build_manifest(src, species="mauritius", default_split="train")
    return list(m.records)


def _write(records: list[ImageRecord], arm: str, bg: str, seed: int) -> dict[str, Any]:
    b = BOUNDS[arm]
    split = IdentitySplitter(
        val_fraction=b["val_fraction"], test_fraction=b["test_fraction"],
        seed=seed, size_mode="exact", min_per_split=2,
    ).split(Manifest.from_records(records))
    split.assert_identity_disjoint()
    out = pathlib.Path(f"data/manifests/mauritius_{arm}_{bg}_manifest.csv")
    manifest_to_csv(split, out)
    per_id = pd.Series([r.identity for r in records]).value_counts()
    return {
        "arm": arm, "background": bg, "manifest": str(out),
        "manifest_hash": split.manifest_hash,
        "n_images": len(records), "n_identities": int(per_id.size),
        "images_per_identity": {"min": int(per_id.min()), "median": int(per_id.median()),
                                "max": int(per_id.max())},
        "split_counts": {s: len(split.filter_split(s)) for s in ("train", "val", "test")},
        "split_identities": {s: len(split.identities(s)) for s in ("train", "val", "test")},
        "fold_bounds": b,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=pathlib.Path, default=OUT_JSON)
    args = ap.parse_args()

    summaries = []
    for bg in BACKGROUNDS:
        recs = _records(bg)
        summaries.append(_write(recs, "pool40", bg, args.seed))
        day = [r for r in recs if r.identity.startswith(SESSION_DAY)]
        summaries.append(_write(day, "session0831", bg, args.seed))

    print("=== manifests ===")
    print(f"  {'arm':13s} {'bg':9s} {'images':>7s} {'ids':>4s} {'img/bat (min/med/max)':>22s}  split ids")
    for s in summaries:
        i = s["images_per_identity"]
        print(f"  {s['arm']:13s} {s['background']:9s} {s['n_images']:7d} {s['n_identities']:4d} "
              f"{i['min']:7d}/{i['median']:3d}/{i['max']:<8d} "
              f"{s['split_identities']['train']}/{s['split_identities']['val']}/{s['split_identities']['test']}")
    # sanity: the three backgrounds of an arm must share one identity partition
    for arm in ("pool40", "session0831"):
        sets = [tuple(sorted(pd.read_csv(s["manifest"]).query("split=='test'").identity.unique()))
                for s in summaries if s["arm"] == arm]
        print(f"  {arm}: identical test identities across backgrounds: {len(set(sets)) == 1}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
