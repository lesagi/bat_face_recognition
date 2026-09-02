"""Stage 2 of dataset generation: turn a cull into a trainable dataset arm.

Stage 1 (``curate_frames.py``) decides which full-resolution frames the model may
see. This turns that decision into images and manifests, in dependency order, with
one command.

    curated keep/  ->  <arm>/ and <arm>_320/  ->  manifests for <arm>

**A cull produces a new arm; it never mutates a published one.** That is a
deliberate constraint, not caution for its own sake. ``manifest_hash`` is what
ties an MLflow run to the exact data it saw, and ``Manifest.from_records`` folds
the split column into that digest -- so regenerating an existing manifest
re-splits it, changes its hash, and quietly severs every published run that cites
it. Worse, a plain directory walk cannot reproduce the manifests that are not
directory walks: ``*_band_*`` are 10-identity band-restricted subsets,
``*_intersect_*`` is an intersection with another background, and ``occl_*`` are
occlusion arms. Rebuilding those from a directory listing would flatten them into
something that merely looks right.

So each cull gets a name (``--arm``, default ``curated``) and writes:

    data/processed/<species>/video/not_augmented/<arm>/{original_bg,green_bg,random_bg,face_ellipse}/<identity>/
    data/processed/<species>/video/not_augmented/<arm>_320/...
    data/manifests/<species>_<arm>_{original,green,random,face_ellipse}_bg_manifest.csv

Nothing already on disk is touched. To train on a cull, point an experiment config
at that arm's manifests; the published arms stay frozen and comparable.

The derived experiment arms -- ``blur``, ``recrop``, ``bgonly``, ``occ_roi``,
``occl``, ``silhouette`` -- are deliberately NOT wired in here. Each reads a
specific published manifest by hardcoded name and belongs to a concluded
experiment, so regenerating them against a fresh cull would produce arms whose
names no longer mean what the write-ups say they mean. Rebuild those individually,
on purpose. ``--list-derived`` prints them and what each consumes.

Usage::

    python scripts/rebuild_dataset.py --species mauritius                    # dry run
    python scripts/rebuild_dataset.py --species mauritius --execute
    python scripts/rebuild_dataset.py --species mauritius --arm day31 --execute
    python scripts/rebuild_dataset.py --species mauritius --only images --execute
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

CURATED = pathlib.Path("data/curated")
PROCESSED = pathlib.Path("data/processed")
MANIFESTS = pathlib.Path("data/manifests")
BACKGROUNDS = ("original_bg", "green_bg", "random_bg", "face_ellipse")
# The clean arms the occlusion step runs on. green and random only: `original` is
# 93% solvable with the face deleted, and `face_ellipse` already removes its own
# periphery, so an occlusion result on either would not mean what it says.
OCCLUSION_BACKGROUNDS = ("green", "random")

# Derived arms, for --list-derived. Each is tied to a published manifest and a
# concluded experiment, which is why none of them is rebuilt automatically.
DERIVED = {
    "blur": ("build_resolution_matched.py --variant blur", "<species>_{green,original,random}_bg"),
    "recrop": ("build_resolution_recrop.py", "<species>_{green,original,random}_bg"),
    "bgonly": ("build_background_only.py --all", "<species>_{original,random}_bg"),
    "occ_roi": ("build_roi_occlusion.py", "<species>_green_bg"),
    "occl": ("occlusion_probe.py --gen-only", "<species>_original_bg"),
    "silhouette": ("generate_silhouette.py", "<species>_green_bg"),
}


def variants_cmd(species: str, crop: str, edge: int, prefix: str, device: str) -> list[str]:
    """One pass emits all four backgrounds for a given (crop, edge)."""
    return [
        sys.executable,
        "scripts/build_variants_from_frames.py",
        "--species",
        species,
        "--frames-root",
        str(CURATED / species / "keep"),
        "--out-root",
        str(PROCESSED / species / "video/not_augmented"),
        "--crop",
        crop,
        "--edge",
        str(edge),
        "--out-layout",
        "bg-crop-edge",
        "--name-style",
        "compact",
        "--device",
        device,
        *(["--variant-prefix", prefix] if prefix else []),
    ]


def manifest_cmds(
    species: str, crop: str, edge: int, prefix: str, val: float, test: float, seed: int
) -> list[list[str]]:
    cmds = []
    for bg in BACKGROUNDS:
        src = PROCESSED / species / "video/not_augmented" / f"{prefix}{bg}" / crop / str(edge)
        out = MANIFESTS / f"{species}_{prefix}{bg}_{crop}_{edge}_manifest.csv"
        cmds.append(
            [
                "uv",
                "run",
                "bat-cli",
                "build-manifest",
                "--input-dir",
                str(src),
                "--output",
                str(out),
                "--species",
                species,
                "--val-fraction",
                str(val),
                "--test-fraction",
                str(test),
                "--seed",
                str(seed),
            ]
        )
    return cmds


def occlusion_cmds(species: str, crop: str, edge: int, prefix: str, device: str) -> list[list[str]]:
    return [
        [
            sys.executable,
            "scripts/build_occlusion_arms.py",
            "--species",
            species,
            "--crop",
            crop,
            "--edge",
            str(edge),
            "--prefix",
            prefix,
            "--device",
            device,
            "--execute",
        ]
    ]


def check_cull(species: str) -> tuple[int, int, int]:
    dec_path = CURATED / species / "decisions.json"
    if not dec_path.exists():
        raise SystemExit(
            f"no cull to build from: {dec_path} is missing (run curate_frames.py --init)"
        )
    dec = json.loads(dec_path.read_text())
    n_keep = sum(1 for v in dec["frames"].values() if v["decision"] == "keep")
    keep = CURATED / species / "keep"
    on_disk = len(list(keep.rglob("*.jpg"))) if keep.is_dir() else 0
    idents = len([p for p in keep.iterdir() if p.is_dir()]) if keep.is_dir() else 0
    return n_keep, on_disk, idents


def collides(species: str, prefix: str, crops: list[str], edges: list[int]) -> list[pathlib.Path]:
    """Anything this run would overwrite. A prefix should be new, or reused knowingly."""
    hits = []
    na = PROCESSED / species / "video/not_augmented"
    for bg in BACKGROUNDS:
        for crop in crops:
            for edge in edges:
                d = na / f"{prefix}{bg}" / crop / str(edge)
                if d.exists():
                    hits.append(d)
    hits += sorted(MANIFESTS.glob(f"{species}_{prefix}*_manifest.csv"))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", default="mauritius")
    ap.add_argument(
        "--prefix",
        default="",
        help="Names this cull's dataset, as a prefix on the background directory and "
        "the manifest (e.g. 'day31' -> day31_green_bg/). Empty is the canonical path.",
    )
    ap.add_argument("--crops", default="head,eyes", help="Comma list: head and/or eyes.")
    ap.add_argument(
        "--edges",
        default="192",
        help="Comma list of output edges. 192 is the measured value at which neither "
        "species upscales under the eyes alignment (0%% / 0.4%%).",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default="", help="Comma list from: images, manifests, occlusion.")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--execute", action="store_true", help="Actually run; default is a dry run.")
    ap.add_argument("--force", action="store_true", help="Proceed even though the target exists.")
    ap.add_argument("--list-derived", action="store_true", help="Print the derived arms and exit.")
    args = ap.parse_args()

    if args.list_derived:
        print("\n  Derived arms are rebuilt individually, on purpose:")
        for name, (cmd, reads) in DERIVED.items():
            print(f"    {name:11s} scripts/{cmd}\n                reads {reads}")
        print()
        return

    prefix = (
        "" if not args.prefix else (args.prefix if args.prefix.endswith("_") else f"{args.prefix}_")
    )
    crops = [c.strip() for c in args.crops.split(",") if c.strip()]
    edges = [int(e) for e in args.edges.split(",") if e.strip()]
    bad = set(crops) - {"head", "eyes"}
    if bad:
        raise SystemExit(f"unknown crop(s) {sorted(bad)}; known: head, eyes")

    steps: dict[str, list[list[str]]] = {
        "images": [
            variants_cmd(args.species, c, e, prefix, args.device) for c in crops for e in edges
        ],
        "manifests": [
            cmd
            for c in crops
            for e in edges
            for cmd in manifest_cmds(
                args.species, c, e, prefix, args.val_fraction, args.test_fraction, args.seed
            )
        ],
        # Occlusion runs on `head` only: inside an eyes-aligned crop the discs occupy a
        # different area fraction, so `span` - `periphery` would not mean there what it
        # means on head, and the two could not be compared.
        "occlusion": [
            cmd
            for e in edges
            for cmd in occlusion_cmds(args.species, "head", e, prefix, args.device)
            if "head" in crops
        ],
    }
    order = ["images", "manifests", "occlusion"]
    want = [s.strip() for s in args.only.split(",") if s.strip()] or order
    unknown = set(want) - set(order)
    if unknown:
        raise SystemExit(f"unknown step(s) {sorted(unknown)}; known: {order}")
    want = [s for s in order if s in want]

    n_keep, on_disk, idents = check_cull(args.species)
    print(
        f"\n  cull      {n_keep} frames marked keep; {on_disk} present under "
        f"{CURATED / args.species / 'keep'} ({idents} identities)"
    )
    if n_keep != on_disk:
        raise SystemExit(
            "  !! decisions.json and keep/ disagree -- run `curate_frames.py --execute` "
            "first,\n     otherwise this would build from a stale frame set."
        )
    if on_disk == 0:
        raise SystemExit("  !! nothing marked keep; there is no dataset to build.")

    print(f"  dataset   {prefix or '(canonical, no prefix)'}   crops={crops}  edges={edges}")
    existing = collides(args.species, prefix, crops, edges)
    if existing:
        print(f"  !! target already exists ({len(existing)} path(s)); it would be overwritten:")
        for e in existing[:6]:
            print(f"       {e}")
        if not args.force:
            raise SystemExit("     pick a new --prefix, or pass --force to overwrite.")

    print(f"  plan      {len(want)} step(s)")
    for s in want:
        print(f"    {s:11s} {len(steps[s])} command(s)")
        for c in steps[s]:
            print("       $ " + " ".join(c))
    if not args.execute:
        print("\n  DRY RUN -- nothing built. Re-run with --execute.\n")
        return

    for s in want:
        print(f"\n  == {s} ==")
        t0 = time.time()
        for cmd in steps[s]:
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                raise SystemExit(f"\n  step {s} failed (exit {rc}): {' '.join(cmd)}")
        print(f"     done in {time.time() - t0:.0f}s")

    print("\n  built. Manifests:")
    for m in sorted(MANIFESTS.glob(f"{args.species}_{prefix}*_manifest.csv")):
        print(f"    {m}")
    print()


if __name__ == "__main__":
    main()
