"""Cut identity-restricted arms out of a built manifest, for the selection comparison.

Two ways to choose which mauritius bats enter a species comparison, and this builds
both so they can be run against each other:

``paired``   the 12 bats the Hungarian assignment matched 1:1 to the 12 rousettus
             bats on lighting (`pair_species_bats.py`). Tightest match, fewest bats.
``lit``      every bat whose lighting falls inside the range rousettus itself spans
             (`rank_bats_by_lighting.py`). Looser match, more bats.
``all``      every curated bat, as the uncontrolled baseline.

Why not just re-run `bat-cli build-manifest` on a filtered directory: the split must
be drawn **over the arm's own identities**, not inherited from a larger manifest.
Inheriting would give the arms different val/test sizes depending on which bats
happened to land where, and the comparison would partly measure that. So each arm is
split from scratch with the same seed and fractions, over its own identity list.

The arms deliberately differ in identity count (12 vs ~24), which is itself a
confound: more training identities is easier. ``--subsample N`` draws N identities
from an arm at random, which is how `lit` can be cut to 12 to isolate "matched
selection" from "more bats".

    uv run python scripts/build_arm_manifests.py --species mauritius --prefix cull2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

from bat_core.types import Manifest
from bat_data import manifest_from_csv, manifest_to_csv
from bat_data.splitter import IdentitySplitter

LIGHTING = pathlib.Path("outputs/quality/bat_lighting.json")
PAIRS = pathlib.Path("outputs/quality/species_pairs.json")
MANIFESTS = pathlib.Path("data/manifests")


def arm_identities(name: str, available: set[str]) -> list[str]:
    if name == "all":
        return sorted(available)
    if name == "paired":
        ids = [p["mauritius"] for p in json.loads(PAIRS.read_text())["pairs"]]
    elif name == "lit":
        ids = [
            b["identity"]
            for b in json.loads(LIGHTING.read_text())["bats"]
            if b.get("lighting_match") == "good"
        ]
    else:
        raise SystemExit(f"unknown arm {name!r}")
    # A bat with no kept frames never reaches the built manifest, so intersect rather
    # than fail: `lit` nominally has 25 bats but one was culled to zero frames.
    return sorted(set(ids) & available)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", default="mauritius")
    ap.add_argument("--prefix", default="cull2")
    ap.add_argument("--crop", default="head")
    ap.add_argument("--edge", type=int, default=224)
    ap.add_argument("--backgrounds", nargs="+", default=["green_bg", "original_bg", "random_bg"])
    ap.add_argument("--arms", nargs="+", default=["paired", "lit", "all"])
    ap.add_argument(
        "--min-keeps",
        type=int,
        default=10,
        help="Exclude bats with fewer kept frames than this. A bat with 3 frames "
        "contributes almost no same-identity pairs, and lands in a 2-identity "
        "test split often enough to add real variance for no information.",
    )
    ap.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="Draw this many identities from each arm (0 = keep all).",
    )
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    prefix = f"{args.prefix}_" if args.prefix and not args.prefix.endswith("_") else args.prefix
    written = []
    for bg in args.backgrounds:
        src = MANIFESTS / f"{args.species}_{prefix}{bg}_{args.crop}_{args.edge}_manifest.csv"
        if not src.exists():
            raise SystemExit(f"missing source manifest: {src}")
        base = manifest_from_csv(src)
        available = {r.identity for r in base.records}

        for arm in args.arms:
            ids = arm_identities(arm, available)
            if args.min_keeps:
                counts = {i: sum(1 for r in base.records if r.identity == i) for i in ids}
                thin = sorted(i for i in ids if counts[i] < args.min_keeps)
                if thin:
                    print(
                        f"  {arm}/{bg}: dropping {len(thin)} thin bat(s) "
                        f"(<{args.min_keeps} frames): "
                        + ", ".join(f"{i}={counts[i]}" for i in thin)
                    )
                ids = [i for i in ids if counts[i] >= args.min_keeps]
            tag = arm
            if args.subsample and len(ids) > args.subsample:
                ids = sorted(random.Random(args.seed).sample(ids, args.subsample))
                tag = f"{arm}{args.subsample}"
            recs = [r for r in base.records if r.identity in ids]
            if not recs:
                print(f"  {arm}/{bg}: no records, skipped")
                continue

            # Split over THIS arm's identities. `IdentitySplitter.split` takes a
            # manifest, partitions its identities and rehashes -- so the arm gets a
            # split drawn over its own bats rather than inherited from the larger
            # build, which would hand the arms different val/test sizes by accident.
            splitter = IdentitySplitter(
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                seed=args.seed,
            )
            man = splitter.split(Manifest.from_records(recs))
            man.assert_identity_disjoint()
            recs = man.records
            out = MANIFESTS / (
                f"{args.species}_{prefix}{tag}_{bg}_{args.crop}_{args.edge}_manifest.csv"
            )
            counts = {
                s: len({r.identity for r in recs if r.split == s}) for s in ("train", "val", "test")
            }
            print(
                f"  {tag:10s} {bg:12s} {len(ids):2d} ids  {len(recs):5d} imgs  "
                f"train/val/test ids = {counts['train']}/{counts['val']}/{counts['test']}"
                + ("" if args.execute else "   (dry run)")
            )
            if args.execute:
                manifest_to_csv(man, out)
                written.append(out)

    if written:
        print(f"\n  wrote {len(written)} manifest(s)")
        for w in written:
            print(f"    {w}")
    elif not args.execute:
        print("\n  DRY RUN -- nothing written. Re-run with --execute.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
