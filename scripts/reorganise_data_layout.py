"""Reorganise ``data/`` so each top-level directory means exactly one thing.

Why
---
Three problems had accumulated. (1) ``raw/`` used two conflicting conventions --
``raw/video/mauritius/`` (91 .mp4) alongside ``raw/mauritius/video/`` (1392
extracted .jpg) -- and most of what it held was derived frames, not source
media. (2) ``processed/`` mixed the two canonical species trees with a 1.9G
curation workspace, a pre-migration flat dump, a legacy flat arm, and a YOLO
annotation set. (3) The training data behind ``models/preprocessing/face_seg.pt``
was scattered across six locations under ``interim/`` and ``processed/``, filed
as "intermediate" -- which it is not.

Target
------
``raw/`` source media only, ``interim/`` derived intermediates only,
``processed/`` the two canonical trees only, ``annotations/`` YOLO training data
grouped by the model it trains, ``work/`` curation workspaces.

What is preserved, and how it is checked
----------------------------------------
**Nothing that any manifest points at moves.** Every ``data/manifests/*.csv``
and every ``.hash`` sidecar is byte-unchanged, so no manifest hash changes and
past MLflow runs stay comparable -- unlike ``migrate_data_layout.py``, which had
to rewrite paths and log the old->new hash mapping. ``--execute`` runs all moves
first, re-verifies every manifest via ``manifest_from_csv`` (which raises on
hash mismatch), and only deletes once that passes.

Idempotent: every op is skipped when its source is gone and its destination is
already in place, so a partial run can simply be re-run.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

DATA = pathlib.Path("data")

# ---------------------------------------------------------------------------
# Directory moves, in dependency order.
#
# raw/mauritius/video must be vacated before the .mp4 files can move into it,
# so the frames leave first (step 1) and the videos arrive after (step 3).
# ---------------------------------------------------------------------------
# ``guard`` names a glob the source must still contain for the move to apply.
# Only the first move needs one: ``raw/mauritius/video`` is vacated of frames and
# then *refilled* with the .mp4 files by move 3, so on a second run both source
# and destination exist and the plain collision check would abort. Requiring a
# .jpg in the source distinguishes "not yet migrated" from "already migrated".
MOVES: list[tuple[str, str] | tuple[str, str, str]] = [
    # 1-2. mauritius: extracted frames + contactsheets are derived -> interim/
    ("raw/mauritius/video", "interim/mauritius/video/frames", "*.jpg"),
    ("interim/mauritius/video/frames/contactsheets", "interim/mauritius/video/contactsheets"),
    # 3. the 91 source .mp4 land in the now-vacant slot, under <species>/<source>
    ("raw/video/mauritius", "raw/mauritius/video"),
    # 5. rousettus: these are extracted frames too, despite living under raw/
    (
        "raw/rousettus/video/frontal_video_frames/rousettus-09_09_2025",
        "interim/rousettus/video/frames",
    ),
    # 6. YOLO segmentation training data -> annotations/, grouped by model
    ("interim/mauritius/segmentation/general", "annotations/face_seg/mauritius/general"),
    ("interim/mauritius/segmentation/frontal_only", "annotations/face_seg/mauritius/frontal_only"),
    ("interim/rousettus/still/segmentation", "annotations/face_seg/rousettus/still"),
    (
        "processed/siamese_input_aligned_picsum/rous_images+labels",
        "annotations/face_seg/rousettus/still_alt",
    ),
    (
        "interim/rousettus/video/segmentation/yolo_dataset",
        "annotations/face_seg/rousettus/video_yolo",
    ),
    ("interim/rousettus/video/segmentation/images", "annotations/face_seg/rousettus/video/images"),
    ("interim/rousettus/video/segmentation/labels", "annotations/face_seg/rousettus/video/labels"),
    (
        "interim/rousettus/video/segmentation/annotations.xml",
        "annotations/face_seg/rousettus/video/annotations.xml",
    ),
    # 7-8. curation workspace + legacy flat arm leave processed/
    ("processed/mauritius_frontal", "work/mauritius_frontal"),
]

# Deleted only after the moves land and the manifest check passes.
DELETES: list[tuple[str, str]] = [
    ("interim/rousettus/video/augmented", "augmented output in interim/; regenerable"),
    ("processed/mauritius/archived", "pre-migration flat dump; re-downloadable"),
    (
        "processed/siamese_input_aligned_picsum",
        "98 of 99 images + all 99 labels byte-identical to annotations/face_seg/mauritius/general",
    ),
]

# Emptied by the moves above; removed only if genuinely empty.
PRUNE: list[str] = [
    "raw/still",
    "raw/video",
    "raw/rousettus/video/frontal_video_frames",
    "raw/rousettus/video",
    "interim/rousettus/video/augmented",
    "interim/rousettus/video/not_augmented",
    "interim/rousettus/video/segmentation",
    "interim/rousettus/still",
    "interim/mauritius/segmentation",
]


class Runner:
    """Applies (or, by default, simulates) the migration.

    In dry-run the moves are not performed, so later ops would see stale state
    and false-alarm -- the chained ``raw/mauritius/video -> interim/.../frames
    -> .../contactsheets`` sequence in particular. So dry-run keeps a virtual
    filesystem: ``moved`` maps a destination back to where its bytes still
    physically live, and every existence question is answered through it.
    """

    def __init__(self, execute: bool) -> None:
        self.execute = execute
        self.done = 0
        self.skipped = 0
        self.moved: dict[pathlib.Path, pathlib.Path] = {}
        self.gone: set[pathlib.Path] = set()

    # -- virtual filesystem -------------------------------------------------

    def real(self, p: pathlib.Path) -> pathlib.Path:
        """Where p's bytes live right now, following the chain of pending moves."""
        for _ in range(10):
            for dst, src in self.moved.items():
                if p == dst:
                    p = src
                    break
                if dst in p.parents:
                    p = src / p.relative_to(dst)
                    break
            else:
                return p
        return p

    def exists(self, p: pathlib.Path) -> bool:
        target = self.real(p)
        if target != p:
            return target.exists()
        if any(p == g or g in p.parents for g in self.gone):
            return False
        return p.exists()

    def record(self, src: pathlib.Path, dst: pathlib.Path) -> None:
        if self.execute:
            return
        self.moved[dst] = src
        self.gone.add(src)

    # -- operations ---------------------------------------------------------

    def say(self, verb: str, detail: str) -> None:
        prefix = "" if self.execute else "[dry-run] "
        print(f"{prefix}{verb:<8} {detail}")

    def move(self, src_rel: str, dst_rel: str, guard: str = "") -> None:
        src, dst = DATA / src_rel, DATA / dst_rel
        if guard and self.exists(src) and not list(self.real(src).glob(guard)):
            self.say("skip", f"{src_rel} (no {guard}; already migrated)")
            self.skipped += 1
            return
        if not self.exists(src):
            if self.exists(dst):
                self.say("skip", f"{src_rel} (already at {dst_rel})")
                self.skipped += 1
            else:
                self.say("MISSING", f"{src_rel} -- and {dst_rel} absent; nothing to do")
            return
        if self.exists(dst):
            print(f"  !! refusing: both {src_rel} and {dst_rel} exist", file=sys.stderr)
            raise SystemExit(1)
        self.say("move", f"{src_rel} -> {dst_rel}  ({self.du(src)})")
        if self.execute:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        self.record(src, dst)
        self.done += 1

    def delete(self, rel: str, why: str) -> None:
        path = DATA / rel
        if not self.exists(path):
            self.say("skip", f"{rel} (already gone)")
            self.skipped += 1
            return
        self.say("DELETE", f"{rel}  ({self.du(path)})  -- {why}")
        if self.execute:
            target = self.real(path)
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        else:
            self.gone.add(path)
        self.done += 1

    def prune(self, rel: str) -> None:
        path = DATA / rel
        if not self.exists(path):
            return
        target = self.real(path)
        if not target.is_dir():
            return
        # .DS_Store must not count as content: it is deleted in the same run
        leftovers = [
            q
            for q in target.rglob("*")
            if q.name != ".DS_Store" and not self.gone_virtually(target, q)
        ]
        if leftovers:
            self.say("keep", f"{rel} (not empty: {len(leftovers)} entries)")
            return
        self.say("rmdir", rel)
        if self.execute:
            shutil.rmtree(target)
        else:
            self.gone.add(path)
        self.done += 1

    def gone_virtually(self, root: pathlib.Path, q: pathlib.Path) -> bool:
        """True if q is inside something this run already moved or deleted."""
        return any(g == q or g in q.parents for g in self.gone)

    def du(self, path: pathlib.Path) -> str:
        return du(self.real(path))


def du(path: pathlib.Path) -> str:
    total = (
        sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        if path.is_dir()
        else path.stat().st_size
    )
    for unit in ("B", "K", "M", "G"):
        if total < 1024 or unit == "G":
            return f"{total:.0f}{unit}" if unit == "B" else f"{total:.1f}{unit}"
        total /= 1024.0
    return f"{total:.1f}G"


def relink_pool40(runner: Runner) -> None:
    """Rewrite the 40 pool40 symlinks from absolute to relative targets.

    They pointed at ``data/processed/mauritius_frontal/...`` by absolute path, so
    the workspace move breaks them. Relative targets survive any future move.
    """
    frames = runner.real(DATA / "work/mauritius_frontal/_pool40/frames")
    if not frames.is_dir():
        runner.say("skip", "_pool40/frames (not present)")
        return
    exp = runner.real(DATA / "work/mauritius_frontal/_experiment_16/frames")
    rev = runner.real(DATA / "work/mauritius_frontal/_manual_review")
    fixed = already = 0
    for link in sorted(frames.iterdir()):
        if not link.is_symlink():
            continue
        ident = link.name
        if (exp / ident).is_dir():
            target = f"../../_experiment_16/frames/{ident}"
        elif (rev / ident / "frames").is_dir():
            target = f"../../_manual_review/{ident}/frames"
        else:
            print(f"  !! no target found for {ident}", file=sys.stderr)
            raise SystemExit(1)
        if os.readlink(link) == target:
            already += 1
            continue
        runner.say("relink", f"_pool40/frames/{ident} -> {target}")
        if runner.execute:
            link.unlink()
            link.symlink_to(target)
        fixed += 1
    print(f"  relinked {fixed}, already relative {already}")


def drop_ds_store(runner: Runner) -> None:
    found = sorted(DATA.rglob(".DS_Store"))
    if not found:
        return
    runner.say("DELETE", f"{len(found)} x .DS_Store")
    if runner.execute:
        for p in found:
            p.unlink()


def verify_manifests() -> None:
    """Every manifest must still parse and match its .hash sidecar.

    manifest_from_csv raises InvalidManifestError on mismatch, so a clean pass
    proves nothing under processed/ moved.
    """
    from bat_data import manifest_from_csv

    csvs = sorted((DATA / "manifests").glob("*.csv"))
    for p in csvs:
        manifest_from_csv(p)
    print(f"  {len(csvs)} manifests parse, all hashes verified")


def verify_images() -> None:
    import pandas as pd

    missing = 0
    for p in sorted((DATA / "manifests").glob("*.csv")):
        bad = [q for q in pd.read_csv(p)["path"] if not pathlib.Path(q).exists()]
        if bad:
            missing += len(bad)
            print(f"  {p.name}: {len(bad)} missing, e.g. {bad[0]}", file=sys.stderr)
    if missing:
        raise SystemExit(f"{missing} manifest images missing -- refusing to delete")
    print("  every manifest image resolves on disk")


def verify_symlinks() -> None:
    dangling = [p for p in DATA.rglob("*") if p.is_symlink() and not p.exists()]
    if dangling:
        for p in dangling:
            print(f"  dangling: {p} -> {os.readlink(p)}", file=sys.stderr)
        raise SystemExit("dangling symlinks -- refusing to delete")
    print("  no dangling symlinks")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--execute", action="store_true", help="actually move/delete (default: dry-run)"
    )
    ap.add_argument(
        "--no-delete", action="store_true", help="do the moves, skip the ~1.5G of deletions"
    )
    args = ap.parse_args()

    if not DATA.is_dir():
        raise SystemExit("run from the repository root (data/ not found)")

    runner = Runner(args.execute)

    print("\n=== moves ===")
    for entry in MOVES:
        runner.move(*entry)

    print("\n=== relink pool40 symlinks (absolute -> relative) ===")
    relink_pool40(runner)

    if args.no_delete:
        print("\n=== deletions skipped (--no-delete) ===")
    else:
        print("\n=== verify before deleting ===")
        if args.execute:
            verify_manifests()
            verify_images()
            verify_symlinks()
        else:
            print("  (dry-run: checks run for real under --execute)")

        print("\n=== deletions ===")
        for rel, why in DELETES:
            runner.delete(rel, why)
        drop_ds_store(runner)

    print("\n=== prune emptied directories ===")
    for rel in PRUNE:
        runner.prune(rel)

    verb = "applied" if args.execute else "planned"
    print(f"\n{verb} {runner.done} operations, skipped {runner.skipped}")
    if not args.execute:
        print("re-run with --execute to apply")


if __name__ == "__main__":
    main()
