"""Serve the contact sheets and accept decisions straight back from the browser.

Without this the review loop is: click through a bat, Export, find the file in
Downloads, switch to a terminal, run ``curate_frames.py --authority html --import``,
reload. Per bat. The decisions are already in the page; the round trip through the
filesystem is the only reason a terminal is involved at all.

``python -m http.server`` cannot help because it answers GET and nothing else. This
subclasses it and adds one POST route, ``/_save``, carrying exactly the payload the
Export button builds.

A save applies through the **same functions the CLI uses** --
``curate_frames.apply_html`` -> ``relocate`` -> ``execute`` -> ``save_decisions`` --
rather than reimplementing the merge. So a decision made in the browser and one made
with ``--import`` are the same operation, and the invariant the whole design rests on
still holds: decisions.json is the source of truth, and files are relocated to match
it. The sheets are then regenerated so the index's counts and lighting-ranked order
reflect the save, and the page reloads onto a fresh seed stamp (which clears the now
superseded localStorage for that bat).

Scope, deliberately: bound to 127.0.0.1, reachable over the SSH tunnel you already
use, no auth. It writes only to one species' curated tree, and only decisions it can
resolve -- every frame key must already exist in decisions.json, and every state must
be one of keep/drop/undecided. Unknown keys are reported, not invented.

    uv run python scripts/serve_review.py --species mauritius --port 8811
    # then, locally:  ssh -N -L 8811:127.0.0.1:8811 <user>@<host>
    # and open:       http://127.0.0.1:8811/review/index.html
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import curate_frames as cf  # noqa: E402
from make_review_sheets import build_sheets  # noqa: E402

MAX_BODY = 4 * 1024 * 1024  # a 250-frame bat's payload is ~20 KB; this is slack, not a target
_LOCK = threading.Lock()


def apply_payload(species: str, payload: dict) -> dict:
    """Fold one sheet's decisions into decisions.json and move the files to match."""
    items = payload.get("frames")
    if not isinstance(items, dict) or not items:
        return {"ok": False, "error": "payload has no 'frames' object"}

    dec = cf.load_decisions(species)
    if not dec.get("frames"):
        return {"ok": False, "error": f"no decisions.json for {species}; run --init first"}

    plan = cf.Plan()
    for key, state in items.items():
        if state not in cf.STATES:
            plan.problems.append(f"unknown state {state!r} for {key}")
            continue
        rec = dec["frames"].get(key)
        if rec is None:
            plan.problems.append(f"unknown frame: {key}")
            continue
        if rec["decision"] != state:
            plan.changed[key] = (rec["decision"], state)
            rec["decision"] = state
            rec["source"] = f"browser:{payload.get('identity', '?')}"

    cf.relocate(species, dec, plan)
    moved = cf.execute(plan)
    cf.save_decisions(species, dec)
    n_sheets = build_sheets(species)

    tally = {s: sum(1 for v in dec["frames"].values() if v["decision"] == s) for s in cf.STATES}
    return {
        "ok": True,
        "identity": payload.get("identity"),
        "changed": len(plan.changed),
        "moved": moved,
        "problems": plan.problems[:10],
        "n_problems": len(plan.problems),
        "sheets": n_sheets,
        "totals": tally,
    }


class Handler(http.server.SimpleHTTPRequestHandler):
    species = "mauritius"

    def _json(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:  # noqa: N802  (http.server's required spelling)
        if self.path.rstrip("/") != "/_save":
            self._json(404, {"ok": False, "error": "unknown endpoint"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0 or n > MAX_BODY:
            self._json(413, {"ok": False, "error": f"body must be 1..{MAX_BODY} bytes"})
            return
        try:
            payload = json.loads(self.rfile.read(n))
        except (ValueError, UnicodeDecodeError) as exc:
            self._json(400, {"ok": False, "error": f"bad JSON: {exc}"})
            return
        # One save at a time: two browser tabs finishing together would otherwise
        # interleave a read-modify-write of decisions.json and lose one of them.
        with _LOCK:
            try:
                result = apply_payload(self.species, payload)
            except Exception as exc:  # surface it in the page rather than only the log
                self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                return
        self._json(200 if result.get("ok") else 400, result)

    def log_message(self, fmt: str, *args) -> None:
        # Quiet the per-thumbnail GET spam; a sheet is 250 images.
        if "POST" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", default="mauritius")
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--host", default="127.0.0.1", help="Keep this on loopback.")
    args = ap.parse_args()

    root = cf.CURATED / args.species
    if not (root / "decisions.json").exists():
        raise SystemExit(f"no cull at {root} -- run curate_frames.py --init --execute first")

    Handler.species = args.species
    handler = functools.partial(Handler, directory=str(root))
    srv = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    print(f"  serving {root} on http://{args.host}:{args.port}")
    print(f"  open    http://{args.host}:{args.port}/review/index.html")
    print("  the Save button posts to /_save; Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
