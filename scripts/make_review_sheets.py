"""Contact sheets for the stage-1 cull: one offline HTML page per bat.

Each page shows every frame in that bat's superset as a 320px thumbnail with the
segmentation mask outlined, plus what the automatic frontal gate decided and why.
Click a card to cycle keep -> drop -> undecided, or use the keyboard. "Export"
downloads a decisions file that ``curate_frames.py --authority html --import``
folds back into ``decisions.json``.

Design constraints worth stating, because they rule out the obvious choices:

* **Offline.** These pages are opened from the filesystem on a workstation, so
  there are no webfonts, no CDN, and no server. Thumbnails are referenced by
  relative path rather than inlined as data URIs -- a 250-frame bat would
  otherwise produce a ~5 MB single-file page that no browser enjoys.
* **Interrupted reviews must survive.** 32 bats x ~250 frames is not one sitting,
  so every click is written to ``localStorage`` immediately, keyed by identity.
  Reloading resumes exactly where you left off; Export is only needed when you
  want the decisions back in the JSON.
* **The gate's verdict is shown, not applied.** Frames the gate rejected are the
  interesting ones -- ``asymmetric`` and low ``eye_conf`` dominate, and some of
  those are perfectly usable frames. Sorting puts them first for that reason.
"""

from __future__ import annotations

import json
import pathlib

CURATED = pathlib.Path("data/curated")
STATES = ("keep", "drop", "undecided")

_CSS = """
:root{
  --bg:#f7f6f3; --panel:#fffffe; --ink:#1b1a17; --muted:#6f6a60; --line:#e2ded5;
  --keep:#2f7d4f; --keep-bg:#e6f2ea; --drop:#a8322d; --drop-bg:#f7e7e5;
  --und:#8a6d1f; --und-bg:#f6efdb; --accent:#2f5d7d;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme=light]){
    --bg:#16181a; --panel:#1f2225; --ink:#e8e6e1; --muted:#9a958c; --line:#313539;
    --keep:#6fce97; --keep-bg:#1d2f25; --drop:#e8867f; --drop-bg:#33211f;
    --und:#d9bd6a; --und-bg:#302a1c; --accent:#8fbcd9;
  }
}
:root[data-theme=dark]{
  --bg:#16181a; --panel:#1f2225; --ink:#e8e6e1; --muted:#9a958c; --line:#313539;
  --keep:#6fce97; --keep-bg:#1d2f25; --drop:#e8867f; --drop-bg:#33211f;
  --und:#d9bd6a; --und-bg:#302a1c; --accent:#8fbcd9;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{position:sticky;top:0;z-index:9;background:var(--panel);border-bottom:1px solid var(--line);
  padding:14px 20px;display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
h1{margin:0;font-size:19px;letter-spacing:-.01em;font-variant-numeric:tabular-nums}
.sub{color:var(--muted);font-size:13px}
.tallies{margin-left:auto;display:flex;gap:8px;font-size:13px;font-variant-numeric:tabular-nums}
.t{padding:3px 9px;border-radius:999px;border:1px solid var(--line)}
.t.keep{color:var(--keep);background:var(--keep-bg)}
.t.drop{color:var(--drop);background:var(--drop-bg)}
.t.undecided{color:var(--und);background:var(--und-bg)}
button{font:inherit;padding:6px 12px;border-radius:7px;border:1px solid var(--line);
  background:var(--panel);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
main{padding:18px 20px 60px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px}
figure{margin:0;background:var(--panel);border:2px solid var(--line);border-radius:10px;
  overflow:hidden;cursor:pointer;transition:border-color .12s}
figure:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
figure[data-d=keep]{border-color:var(--keep)}
figure[data-d=drop]{border-color:var(--drop);opacity:.55}
figure[data-d=undecided]{border-color:var(--und);border-style:dashed}
img{display:block;width:100%;height:auto;background:#000}
figcaption{padding:7px 9px;font-size:12px;display:flex;justify-content:space-between;
  gap:8px;align-items:center;font-variant-numeric:tabular-nums}
.reason{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.mark{font-weight:650}
figure[data-d=keep] .mark{color:var(--keep)}
figure[data-d=drop] .mark{color:var(--drop)}
figure[data-d=undecided] .mark{color:var(--und)}
.hint{color:var(--muted);font-size:13px;margin:0 0 14px}
kbd{font:inherit;font-size:12px;border:1px solid var(--line);border-bottom-width:2px;
  border-radius:4px;padding:1px 5px;background:var(--bg)}
a{color:var(--accent)}
"""

_JS = """
const S=['keep','drop','undecided'];
const KEY='batcuration:'+ID;
let store={};
try{store=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){store={}}
const cards=[...document.querySelectorAll('figure')];
function paint(){
  const t={keep:0,drop:0,undecided:0};
  for(const c of cards){
    const d=store[c.dataset.key]||c.dataset.seed;
    c.dataset.d=d; t[d]++;
    c.querySelector('.mark').textContent={keep:'KEEP',drop:'DROP',undecided:'?'}[d];
  }
  for(const s of S)document.querySelector('.t.'+s+' b').textContent=t[s];
}
function persist(){try{localStorage.setItem(KEY,JSON.stringify(store))}catch(e){}}
function set(c,d){store[c.dataset.key]=d;persist();paint()}
function cycle(c){
  const cur=store[c.dataset.key]||c.dataset.seed;
  set(c,S[(S.indexOf(cur)+1)%S.length]);
}
cards.forEach(c=>{
  c.tabIndex=0;
  c.addEventListener('click',()=>cycle(c));
  c.addEventListener('keydown',e=>{
    const m={k:'keep',d:'drop',u:'undecided'}[e.key.toLowerCase()];
    if(m){set(c,m);e.preventDefault();
      (c.nextElementSibling||c).focus?.();}
    else if(e.key==='Enter'||e.key===' '){cycle(c);e.preventDefault()}
  });
});
document.getElementById('exp').addEventListener('click',()=>{
  const frames={};
  for(const c of cards)frames[c.dataset.key]=store[c.dataset.key]||c.dataset.seed;
  const blob=new Blob([JSON.stringify({identity:ID,frames},null,1)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='decisions_'+ID+'.json';a.click();
  URL.revokeObjectURL(a.href);
});
document.getElementById('allk').addEventListener('click',()=>{
  if(confirm('Mark every frame on this page KEEP?'))
    {for(const c of cards)store[c.dataset.key]='keep';persist();paint()}
});
document.getElementById('reset').addEventListener('click',()=>{
  if(confirm('Discard your edits on this page and return to the saved decisions?'))
    {store={};persist();paint()}
});
paint();
"""


def _page(identity: str, rows: list[dict]) -> str:
    cards = []
    for r in rows:
        gate = "gate: keep" if r["auto_keep"] else (r["gate_reason"] or "gate: reject")
        cards.append(
            f'<figure data-key="{r["key"]}" data-seed="{r["decision"]}">'
            f'<img loading="lazy" src="../thumbs/{identity}/{r["file"]}" alt="frame {r["frame"]}">'
            f'<figcaption><span>f{r["frame"]:06d}</span>'
            f'<span class="reason">{gate}</span>'
            f'<span class="mark"></span></figcaption></figure>'
        )
    n_auto = sum(1 for r in rows if r["auto_keep"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{identity} · frame review</title><style>{_CSS}</style></head><body>
<header>
  <h1>{identity}</h1>
  <span class="sub">{len(rows)} frames in superset · {n_auto} pass the automatic frontal gate</span>
  <span class="tallies">
    <span class="t keep">keep <b>0</b></span>
    <span class="t drop">drop <b>0</b></span>
    <span class="t undecided">undecided <b>0</b></span>
  </span>
  <button id="allk">All keep</button>
  <button id="reset">Reset</button>
  <button id="exp">Export…</button>
</header>
<main>
<p class="hint">Click a card to cycle, or focus one and press
<kbd>K</kbd> keep · <kbd>D</kbd> drop · <kbd>U</kbd> undecided.
Every click is saved in this browser straight away, so you can stop and come back.
Frames the gate rejected are listed first — those are the ones worth a human look.
<strong>Export…</strong> writes the file you feed back to
<code>curate_frames.py --authority html</code>.</p>
<div class="grid">
{chr(10).join(cards)}
</div></main>
<script>const ID={json.dumps(identity)};{_JS}</script>
</body></html>"""


def _index(species: str, per: list[tuple[str, int, int, int]]) -> str:
    rows = "".join(
        f'<tr><td><a href="{i}.html">{i}</a></td><td class="n">{n}</td>'
        f'<td class="n">{k}</td><td class="n">{u}</td></tr>'
        for i, n, k, u in per
    )
    tot, totk, totu = sum(p[1] for p in per), sum(p[2] for p in per), sum(p[3] for p in per)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{species} · stage-1 review</title><style>{_CSS}
table{{border-collapse:collapse;background:var(--panel);border:1px solid var(--line);
  border-radius:10px;overflow:hidden;width:100%;max-width:640px}}
th,td{{padding:8px 14px;text-align:left;border-bottom:1px solid var(--line)}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
tr:last-child td{{border-bottom:none}}
</style></head><body>
<header><h1>{species} · stage-1 frame review</h1>
<span class="sub">{len(per)} identities · {tot} frames · {totk} keep · {totu} undecided</span></header>
<main><p class="hint">One page per bat. Decisions are stored per page in this browser;
export each page you finish and fold them in with
<code>curate_frames.py --authority html --import &lt;file&gt;</code>.</p>
<table><thead><tr><th>identity</th><th>frames</th><th>keep</th><th>undecided</th></tr></thead>
<tbody>{rows}</tbody></table></main></body></html>"""


def build_sheets(species: str) -> int:
    root = CURATED / species
    dec = json.loads((root / "decisions.json").read_text())
    met = (
        json.loads((root / "metrics.json").read_text()) if (root / "metrics.json").exists() else {}
    )
    out = root / "review"
    out.mkdir(parents=True, exist_ok=True)

    by_ident: dict[str, list[dict]] = {}
    for key, rec in dec["frames"].items():
        m = met.get(key, {})
        by_ident.setdefault(rec["identity"], []).append(
            {
                "key": key,
                "file": rec["file"],
                "frame": int(key.rsplit("/f", 1)[1]),
                "decision": rec["decision"],
                "auto_keep": bool(m.get("auto_keep")),
                "gate_reason": m.get("gate_reason"),
            }
        )

    per = []
    for identity, rows in sorted(by_ident.items()):
        # Gate-rejected frames first: they are the ones a human decision changes.
        rows.sort(key=lambda r: (r["auto_keep"], r["frame"]))
        (out / f"{identity}.html").write_text(_page(identity, rows))
        per.append(
            (
                identity,
                len(rows),
                sum(1 for r in rows if r["decision"] == "keep"),
                sum(1 for r in rows if r["decision"] == "undecided"),
            )
        )
    (out / "index.html").write_text(_index(species, per))
    return len(per)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", default="mauritius")
    a = ap.parse_args()
    n = build_sheets(a.species)
    print(f"  wrote {n} sheet(s) -> {CURATED / a.species / 'review'}/index.html")
