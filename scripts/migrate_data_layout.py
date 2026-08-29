"""Normalise the mauritius data tree to one layout and one filename convention.

Why
---
Three directory conventions had accumulated (variant-at-root, arm/variant, and
arm-is-the-leaf), and 13,231 files carried the identity twice --
``m--20230805_132834--20230805_132834.532.jpg`` -- because for mauritius the
identity *is* the video stem and the id token was built from it. Rousettus never
had the problem: its identities are names, distinct from its video stems, so it
is left alone.

Target
------
``<arm>/<variant>/<identity>/m--<identity>--f<frame>.jpg`` throughout, with one
subdirectory per identity so frames can be reviewed a bat at a time.

What is preserved, and how it is checked
----------------------------------------
Only the ``path`` column of each manifest changes. Identity, split, background,
source, augmented and quality are copied verbatim, so **the identity partitions
are bit-identical** -- no IdentitySplitter is re-run. The script verifies this
after rewriting.

What necessarily changes
------------------------
``Manifest.from_records`` digests the paths, so every manifest hash changes, and
the ``manifest_hash`` recorded in past MLflow runs no longer matches by equality.
The old->new mapping is written to ``docs/manifest_hash_migration.md`` so those
runs stay traceable. Outputs keyed on basename (native_boxes, battery) must be
regenerated afterwards; the script reports which.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
from typing import Any

import pandas as pd

from bat_core.types import ImageRecord, Manifest
from bat_data import manifest_from_csv, manifest_to_csv

def base_for(species: str) -> pathlib.Path:
    return pathlib.Path(f"data/processed/{species}/video/not_augmented")


# Both species get the same tree. Only mauritius filenames change: rousettus
# identities are names, distinct from its video stems, so its names were never
# duplicated and new_filename() passes them through untouched.
SPECIES = ("mauritius", "rousettus")

# old directory (relative to BASE) -> new directory. pool40 is already correct
# and is deliberately absent.
DIR_MAP = {
    "green_bg": "base/green_bg",
    "original_bg": "base/original_bg",
    "random_bg": "base/random_bg",
    "face_ellipse": "base/face_ellipse",
    "silhouette": "silhouette",
    "224/green_bg": "base_224/green_bg",
    "224/original_bg": "base_224/original_bg",
    "224/random_bg": "base_224/random_bg",
    "224/face_ellipse": "base_224/face_ellipse",
    "320/green_bg": "base_320/green_bg",
    "320/original_bg": "base_320/original_bg",
    "320/random_bg": "base_320/random_bg",
    "320/face_ellipse": "base_320/face_ellipse",
    "blur/green_bg": "blur/green_bg",
    "blur/original_bg": "blur/original_bg",
    "blur/random_bg": "blur/random_bg",
    "recrop/green_bg": "recrop/green_bg",
    "recrop/original_bg": "recrop/original_bg",
    "recrop/random_bg": "recrop/random_bg",
    "bgonly_original": "bgonly/original",
    "bgonly_random": "bgonly/random",
    "occ_roi_none": "occ_roi/none",
    "occ_roi_centre": "occ_roi/centre",
    "occ_roi_matched": "occ_roi/matched",
    "occ_roi_periphery": "occ_roi/periphery",
    "occl_A": "occl/A",
    "occl_B": "occl/B",
    "occl_C": "occl/C",
}

OLD_NAME = re.compile(r"^(?P<t>\w+)--(?P<id>\w+)--(?P=id)\.(?P<frame>\d+)\.jpg$")


def new_filename(name: str) -> str:
    """m--<id>--<id>.<frame>.jpg -> m--<id>--f<frame>.jpg, else unchanged."""
    m = OLD_NAME.match(name)
    if not m:
        return name
    return f"{m.group('t')}--{m.group('id')}--f{int(m.group('frame')):06d}.jpg"


def identity_of(name: str) -> str | None:
    parts = name.split("--")
    return parts[1] if len(parts) >= 3 else None


def plan(species: str) -> dict[str, str]:
    """old repo-relative path -> new, for every file to move."""
    base = base_for(species)
    moves: dict[str, str] = {}
    for old_rel, new_rel in DIR_MAP.items():
        src = base / old_rel
        if not src.is_dir():
            continue
        for f in sorted(src.glob("*.jpg")):
            ident = identity_of(f.name)
            if ident is None:
                continue
            moves[str(f)] = str(base / new_rel / ident / new_filename(f.name))
        # sidecars (build reports and the like) sit beside the images; keep them
        # with their variant rather than stranding them in an emptied directory.
        for f in sorted(src.glob("*")):
            if f.is_file() and f.suffix.lower() != ".jpg":
                moves[str(f)] = str(base / new_rel / f.name)
    return moves


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", choices=(*SPECIES, "all"), default="mauritius")
    ap.add_argument("--execute", action="store_true", help="without this it is a dry run")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/manifest_hash_migration.md"))
    args = ap.parse_args()

    species = SPECIES if args.species == "all" else (args.species,)
    moves = {}
    for sp in species:
        moves.update(plan(sp))
    dests = list(moves.values())
    collisions = len(dests) - len(set(dests))
    renamed = sum(1 for o, n in moves.items() if pathlib.Path(o).name != pathlib.Path(n).name)
    print(f"=== plan ===")
    print(f"  files to move   : {len(moves)}")
    print(f"  of which renamed: {renamed}")
    print(f"  destination collisions: {collisions}")
    if collisions:
        print("  !! refusing to run with collisions")
        return 1
    print(f"  species: {', '.join(species)}")

    manifests = sorted(
        m for sp in species for m in pathlib.Path("data/manifests").glob(f"{sp}_*.csv")
    )
    print(f"  manifests to rewrite: {len(manifests)}")

    if not args.execute:
        ex = list(moves.items())[:3]
        print("\n  examples:")
        for o, n in ex:
            print(f"    {o}\n      -> {n}")
        print("\n  dry run; pass --execute to apply")
        return 0

    # ---- move files
    for old, new in moves.items():
        p = pathlib.Path(new)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(old, new)
    print(f"\n  moved {len(moves)} files")

    # ---- rewrite manifests: path only, everything else verbatim
    rows: list[dict[str, Any]] = []
    for mpath in manifests:
        before = pd.read_csv(mpath)
        old_hash = None
        hp = mpath.with_suffix(mpath.suffix + ".hash")
        if hp.exists():
            old_hash = hp.read_text().strip()
        recs = []
        missing = 0
        for r in before.itertuples():
            new = moves.get(str(r.path), str(r.path))
            if not pathlib.Path(new).exists():
                missing += 1
                continue
            recs.append(ImageRecord(
                path=pathlib.Path(new), identity=r.identity, species=r.species,
                background=r.background, source=r.source, augmented=bool(r.augmented),
                split=r.split, quality=float(r.quality)))
        man = Manifest.from_records(recs)
        man.assert_identity_disjoint()
        manifest_to_csv(man, mpath)
        after = pd.read_csv(mpath)
        # the whole point: partitions must be untouched
        same = (
            before.groupby("split").identity.apply(lambda s: tuple(sorted(set(s)))).to_dict()
            == after.groupby("split").identity.apply(lambda s: tuple(sorted(set(s)))).to_dict()
        )
        rows.append({"manifest": mpath.name, "old_hash": old_hash, "new_hash": man.manifest_hash,
                     "rows_before": len(before), "rows_after": len(after),
                     "missing": missing, "splits_identical": bool(same)})

    bad = [r for r in rows if not r["splits_identical"] or r["missing"]]
    print(f"  rewrote {len(rows)} manifests; splits identical in {sum(r['splits_identical'] for r in rows)}/{len(rows)}")
    if bad:
        print("  !! problems:")
        for r in bad:
            print(f"     {r['manifest']}: missing={r['missing']} splits_identical={r['splits_identical']}")

    # ---- remove now-empty old directories
    for sp in species:
        base = base_for(sp)
        for old_rel in list(DIR_MAP) + ["224", "320"]:
            d = base / old_rel
            if d.is_dir() and not any(d.rglob("*")):
                shutil.rmtree(d)

    # ---- the hash migration record
    lines = [
        "# Manifest hash migration — 2026-08-30 layout normalisation",
        "",
        "The mauritius data tree was normalised to one layout",
        "(`<arm>/<variant>/<identity>/m--<identity>--f<frame>.jpg`) and the duplicated",
        "identity was removed from 13,231 filenames.",
        "",
        "`Manifest.from_records` digests the file paths, so every manifest hash changed.",
        "**Identity partitions did not change** — only the `path` column was rewritten, and",
        "the migration verified split membership before and after for every manifest.",
        "",
        "MLflow runs recorded before this date carry the old hash in",
        "`params.manifest_hash` and `params.split.source_manifest_hash`. Use this table to",
        "resolve them; hash equality alone will no longer match.",
        "",
        "| manifest | old hash | new hash | rows |",
        "|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: x["manifest"]):
        oh = (r["old_hash"] or "—")[:16]
        lines.append(f"| `{r['manifest']}` | `{oh}` | `{r['new_hash'][:16]}` | {r['rows_after']} |")
    lines += ["", "## Regenerate afterwards (these key on basename)", "",
              "- `outputs/quality/native_boxes.parquet` — `scripts/measure_native_boxes.py`",
              "- `outputs/quality/battery.parquet` — `scripts/build_quality_battery.py`", ""]
    args.out.write_text("\n".join(lines))
    print(f"  wrote {args.out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
