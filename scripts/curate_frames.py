"""Stage 1 of dataset generation: decide which full-size frames the model may use.

The problem this solves. Every transformed variant -- ``base``, ``base_224``,
``base_320``, ``blur``, ``recrop``, ``bgonly``, ``occ_roi``, ``occl``,
``silhouette`` -- is downstream of one set of culled full-resolution frames. But
the cull used to be expressed *only* as which files happened to exist in a
frames directory, and the rejects were deleted on the floor. So "this frame is
bad" meant editing ten directories by hand, and "actually put that one back"
meant re-extracting from video. Neither is a filter you can iterate on.

Here the cull is data, not directory state:

    data/curated/<species>/
        decisions.json          <- the ONLY thing you edit; the source of truth
        metrics.json            <- per-frame geometry + the auto gate's verdict
        keep/<identity>/*.jpg        decision = keep       (feeds every variant)
        unreviewed/<identity>/*.jpg  decision = undecided  (never seen by a human)
        dropped/<identity>/*.jpg     decision = drop
        thumbs/<identity>/*.jpg      320px, mask outlined; never relocated
        review/<identity>.html       contact sheet that edits the decisions

Nothing is deleted, so every decision is reversible. Three states, not two:
re-extraction surfaces frames a human never ruled on, and calling those "drop"
would falsely imply they were rejected while calling them "keep" would silently
change the dataset. ``undecided`` says what is true.

Two review surfaces, and you pick which one wins:

    --authority html --import <export.json>   the contact sheet's checkboxes win
    --authority dirs                          where the files currently sit wins
    --authority json                          decisions.json wins (default)

Whichever you choose, the run ends the same way: decisions.json is updated, then
files are **relocated** so the tree matches it. That is the point -- the folders
and the JSON can never drift, because one of them is always reconciled onto the
other.

Frames are keyed ``<identity>/f<frame:06d>``, on the frame number rather than the
filename, because the filename carries a mask-area suffix that shifts if
segmentation is ever re-run. The key survives re-extraction; the filename does not.

Typical use::

    # once, after build_frontal_dataset.py --keep-all
    python scripts/curate_frames.py --species mauritius --init --execute

    # emit contact sheets, review them in a browser, apply the export
    python scripts/curate_frames.py --species mauritius --sheets
    python scripts/curate_frames.py --species mauritius \
        --authority html --import ~/Downloads/decisions_export.json --execute

    # or just drag rejects into dropped/ in a file manager, then
    python scripts/curate_frames.py --species mauritius --authority dirs --execute

Dry-run is the default everywhere; nothing moves without ``--execute``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field

CURATED = pathlib.Path("data/curated")
STATE_DIR = {"keep": "keep", "undecided": "unreviewed", "drop": "dropped"}
DIR_STATE = {v: k for k, v in STATE_DIR.items()}
STATES = tuple(STATE_DIR)
FRAME_RE = re.compile(r"__f(\d+)__")

# Where the pre-existing, human-curated keeps live. Frames matching one of these
# are seeded `keep` so --init reproduces today's dataset exactly rather than
# quietly re-opening 40 bats' worth of settled decisions.
LEGACY_KEEPS = {
    "mauritius": pathlib.Path("data/work/mauritius_frontal/_pool40/frames"),
}


def frame_no(name: str) -> int | None:
    m = FRAME_RE.search(name)
    return int(m.group(1)) if m else None


def key_of(identity: str, name: str) -> str | None:
    n = frame_no(name)
    return None if n is None else f"{identity}/f{n:06d}"


@dataclass
class Plan:
    """What a run intends to do, so a dry-run can print it and stop."""

    moves: list[tuple[pathlib.Path, pathlib.Path]] = field(default_factory=list)
    changed: dict[str, tuple[str, str]] = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)
    problems: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- io


def root(species: str) -> pathlib.Path:
    return CURATED / species


def load_decisions(species: str) -> dict:
    p = root(species) / "decisions.json"
    if not p.exists():
        return {"species": species, "frames": {}}
    return json.loads(p.read_text())


def save_decisions(species: str, dec: dict) -> None:
    p = root(species) / "decisions.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    dec["n_keep"] = sum(1 for v in dec["frames"].values() if v["decision"] == "keep")
    dec["n_drop"] = sum(1 for v in dec["frames"].values() if v["decision"] == "drop")
    dec["n_undecided"] = sum(1 for v in dec["frames"].values() if v["decision"] == "undecided")
    p.write_text(json.dumps(dec, indent=1, sort_keys=True))


def locate(species: str) -> dict[str, tuple[str, pathlib.Path]]:
    """Map frame key -> (current state, path) by looking at where files actually sit."""
    found: dict[str, tuple[str, pathlib.Path]] = {}
    for dirname, state in DIR_STATE.items():
        base = root(species) / dirname
        if not base.is_dir():
            continue
        for ident_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for f in sorted(ident_dir.glob("*.jpg")):
                k = key_of(ident_dir.name, f.name)
                if k:
                    found[k] = (state, f)
    return found


# ------------------------------------------------------------------------ init


def init(species: str, plan: Plan) -> dict:
    """Ingest the extractor's staging tree into the curated layout.

    Seeds each frame `keep` if a human already kept that frame number for that
    bat, else `undecided`. The staged full-resolution frame is moved (not copied)
    so the superset exists in exactly one place.
    """
    stage = root(species) / "_staging"
    if not stage.is_dir():
        plan.problems.append(f"no staging dir at {stage}")
        return load_decisions(species)

    legacy: dict[str, str] = {}
    lk = LEGACY_KEEPS.get(species)
    if lk and lk.is_dir():
        for ident_dir in lk.iterdir():
            real = ident_dir.resolve()
            if not real.is_dir():
                continue
            for f in real.glob("*.jpg"):
                k = key_of(ident_dir.name, f.name)
                if k:
                    legacy[k] = "seed:curated-keep"
    plan.counts["legacy_keeps_found"] = len(legacy)

    dec = load_decisions(species)
    frames = dec.setdefault("frames", {})
    metrics: dict[str, dict] = {}

    for ident_dir in sorted(p for p in stage.iterdir() if p.is_dir()):
        identity = ident_dir.name
        mj = ident_dir / "metrics.json"
        rows = json.loads(mj.read_text())["frames"] if mj.exists() else []
        by_name = {r["file"]: r for r in rows}

        for f in sorted((ident_dir / "frames").glob("*.jpg")):
            k = key_of(identity, f.name)
            if k is None:
                plan.problems.append(f"unparseable frame name: {f}")
                continue
            row = by_name.get(f.name, {})
            if k not in frames:
                state = "keep" if k in legacy else "undecided"
                frames[k] = {
                    "file": f.name,
                    "identity": identity,
                    "decision": state,
                    "source": legacy.get(k, "seed:unreviewed"),
                }
                plan.counts[f"seeded_{state}"] += 1
            metrics[k] = {
                "auto_keep": row.get("auto_keep"),
                "gate_reason": row.get("reason"),
                "mask_area_pct": row.get("mask_area_pct"),
                **{
                    c: row.get(c)
                    for c in (
                        "eye_angle",
                        "interocular",
                        "nose_x_off",
                        "nose_drop",
                        "le_c",
                        "re_c",
                        "no_c",
                    )
                },
            }
            dest = root(species) / STATE_DIR[frames[k]["decision"]] / identity / f.name
            if f.resolve() != dest.resolve():
                plan.moves.append((f, dest))

        for t in sorted((ident_dir / "thumbs").glob("*.jpg")):
            dest = root(species) / "thumbs" / identity / t.name
            if t.resolve() != dest.resolve():
                plan.moves.append((t, dest))

    if metrics:
        (root(species) / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
        plan.counts["metrics_rows"] = len(metrics)
        dec["_metrics_pending"] = metrics
    return dec


# ------------------------------------------------------------------- authority


def apply_dirs(species: str, dec: dict, plan: Plan) -> dict:
    """Where a file currently sits becomes its decision."""
    for k, (state, _path) in locate(species).items():
        rec = dec["frames"].get(k)
        if rec is None:
            plan.problems.append(f"file present but absent from decisions.json: {k}")
            continue
        if rec["decision"] != state:
            plan.changed[k] = (rec["decision"], state)
            rec["decision"], rec["source"] = state, "dirs"
    return dec


def apply_html(species: str, dec: dict, export: pathlib.Path, plan: Plan) -> dict:
    """A contact-sheet export becomes the decision for every frame it names."""
    payload = json.loads(export.read_text())
    items = payload.get("frames", payload)
    if isinstance(items, list):
        items = {i["key"]: i["decision"] for i in items}
    for k, state in items.items():
        if state not in STATES:
            plan.problems.append(f"unknown state {state!r} for {k}")
            continue
        rec = dec["frames"].get(k)
        if rec is None:
            plan.problems.append(f"export names an unknown frame: {k}")
            continue
        if rec["decision"] != state:
            plan.changed[k] = (rec["decision"], state)
            rec["decision"], rec["source"] = state, f"html:{export.name}"
    return dec


# -------------------------------------------------------------------- relocate


def relocate(species: str, dec: dict, plan: Plan) -> None:
    """Queue the moves that make the tree match decisions.json."""
    here = locate(species)
    for k, rec in dec["frames"].items():
        want = STATE_DIR[rec["decision"]]
        cur = here.get(k)
        if cur is None:
            continue  # not yet ingested, or --init will place it
        state, path = cur
        if state == rec["decision"]:
            continue
        plan.moves.append((path, root(species) / want / rec["identity"] / path.name))


def execute(plan: Plan) -> int:
    done = 0
    for src, dst in plan.moves:
        if not src.exists():
            plan.problems.append(f"vanished before move: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        done += 1
    return done


# ------------------------------------------------------------------------ main


def report(species: str, dec: dict, plan: Plan, executed: int | None) -> None:
    print(f"\n  curated root : {root(species)}")
    for label in ("legacy_keeps_found", "seeded_keep", "seeded_undecided", "metrics_rows"):
        if plan.counts.get(label):
            print(f"  {label:20s} {plan.counts[label]}")
    if plan.changed:
        print(f"  decisions changed    {len(plan.changed)}")
        flips = Counter(f"{a} -> {b}" for a, b in plan.changed.values())
        for f, n in flips.most_common():
            print(f"      {f:24s} {n}")
    tally = Counter(v["decision"] for v in dec["frames"].values())
    print("  decisions.json       " + "  ".join(f"{s}={tally.get(s, 0)}" for s in STATES))
    print(
        f"  file moves           {len(plan.moves)}"
        + ("" if executed is None else f" ({executed} applied)")
    )
    if plan.problems:
        print(f"  problems             {len(plan.problems)}")
        for p in plan.problems[:8]:
            print(f"      {p}")
        if len(plan.problems) > 8:
            print(f"      ... and {len(plan.problems) - 8} more")
    if executed is None:
        print("\n  DRY RUN -- nothing moved. Re-run with --execute.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", default="mauritius")
    ap.add_argument("--init", action="store_true", help="Ingest _staging/ into the curated layout.")
    ap.add_argument(
        "--authority",
        choices=("json", "dirs", "html"),
        default="json",
        help="Which surface wins when it disagrees with decisions.json.",
    )
    ap.add_argument(
        "--import", dest="import_path", type=pathlib.Path, help="Contact-sheet export JSON."
    )
    ap.add_argument("--sheets", action="store_true", help="(Re)generate the review HTML and exit.")
    ap.add_argument(
        "--no-sheets",
        action="store_true",
        help="Skip the automatic contact-sheet refresh after --execute.",
    )
    ap.add_argument(
        "--execute", action="store_true", help="Actually move files and write decisions.json."
    )
    args = ap.parse_args()

    if args.sheets:
        from make_review_sheets import build_sheets  # local import: optional dependency

        n = build_sheets(args.species)
        print(f"  wrote {n} contact sheet(s) -> {root(args.species) / 'review'}")
        return

    plan = Plan()
    dec = init(args.species, plan) if args.init else load_decisions(args.species)
    if not dec["frames"] and not args.init:
        raise SystemExit(f"no decisions.json under {root(args.species)} -- run --init first")

    if args.authority == "dirs":
        dec = apply_dirs(args.species, dec, plan)
    elif args.authority == "html":
        if not args.import_path:
            raise SystemExit("--authority html needs --import <export.json>")
        dec = apply_html(args.species, dec, args.import_path, plan)

    if not args.init:
        relocate(args.species, dec, plan)

    executed = None
    if args.execute:
        executed = execute(plan)
        metrics = dec.pop("_metrics_pending", None)
        if metrics:
            # Merge, never replace. A second --init (a new batch of videos staged
            # alongside an existing cull) only measures the frames it staged, so
            # writing wholesale would drop the gate reason and geometry for every
            # frame curated earlier -- which the contact sheets read.
            mp = root(args.species) / "metrics.json"
            merged = json.loads(mp.read_text()) if mp.exists() else {}
            merged.update(metrics)
            mp.write_text(json.dumps(merged, indent=1, sort_keys=True))
        save_decisions(args.species, dec)
        # The contact sheets are a snapshot of decisions.json, so an import that
        # is not followed by a regeneration leaves the index contradicting the
        # data it describes -- still listing a bat as unreviewed after you
        # reviewed it. Refresh them here rather than relying on the operator to
        # remember a second command.
        if not args.no_sheets:
            from make_review_sheets import build_sheets

            n = build_sheets(args.species)
            print(f"  refreshed {n} contact sheet(s)")
    else:
        dec.pop("_metrics_pending", None)
    report(args.species, dec, plan, executed)


if __name__ == "__main__":
    main()
