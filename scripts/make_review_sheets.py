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

import hashlib
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
button.primary{background:var(--keep-bg);border-color:var(--keep);color:var(--keep);font-weight:600}
button.primary:hover{border-color:var(--keep);filter:brightness(.97)}
button[disabled]{opacity:.5;cursor:default}
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
const STAMPKEY=KEY+':seed';
// Stored edits are only meaningful against the seed they were made on. Once an
// export has been imported, decisions.json IS those edits and the page reseeds
// from it -- but localStorage would otherwise keep overriding the fresh seed
// forever, so a page could show decisions the JSON no longer holds. Drop the
// cache whenever this identity's seed has moved. Other bats' in-progress work
// is untouched: the stamp covers one identity only.
let store={};
try{
  if(localStorage.getItem(STAMPKEY)!==SEED_STAMP){
    localStorage.removeItem(KEY);
    localStorage.setItem(STAMPKEY,SEED_STAMP);
  }
  store=JSON.parse(localStorage.getItem(KEY)||'{}');
}catch(e){store={}}
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
function persist(){try{localStorage.setItem(KEY,JSON.stringify(store));localStorage.setItem(STAMPKEY,SEED_STAMP)}catch(e){}}
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
const saveBtn=document.getElementById('save'),status=document.getElementById('status');
// Only reachable when served by scripts/serve_review.py. Opened straight off the
// filesystem there is nothing to POST to, so the button would be a dead control.
if(location.protocol==='file:'){saveBtn.hidden=true}
else saveBtn.addEventListener('click',async()=>{
  const frames={};
  for(const c of cards)frames[c.dataset.key]=store[c.dataset.key]||c.dataset.seed;
  saveBtn.disabled=true;status.textContent='saving…';
  try{
    const r=await fetch('/_save',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({identity:ID,frames})});
    const j=await r.json();
    if(!j.ok){status.textContent='failed: '+(j.error||'unknown');saveBtn.disabled=false;return}
    status.textContent=j.changed+' changed, '+j.moved+' file(s) moved — reloading';
    // The seed has moved, so this page is stale by construction. Reloading picks up
    // the new stamp, which clears this bat's localStorage and reseeds from the JSON.
    setTimeout(()=>location.reload(),600);
  }catch(e){status.textContent='failed: '+e;saveBtn.disabled=false}
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
    # Fingerprint of the seeded decisions for this bat, so the page can tell when
    # decisions.json has moved past whatever the browser cached.
    stamp = hashlib.sha1(
        "".join(
            f'{r["key"]}={r["decision"]};' for r in sorted(rows, key=lambda x: x["key"])
        ).encode()
    ).hexdigest()[:12]
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
  <button id="save" class="primary">Save to server</button>
  <button id="exp">Export…</button>
  <span class="sub" id="status"></span>
</header>
<main>
<p class="hint">Click a card to cycle, or focus one and press
<kbd>K</kbd> keep · <kbd>D</kbd> drop · <kbd>U</kbd> undecided.
Every click is saved in this browser straight away, so you can stop and come back.
Frames the gate rejected are listed first — those are the ones worth a human look.
<strong>Save to server</strong> applies them straight to
<code>decisions.json</code> and moves the files, no terminal needed.
<strong>Export…</strong> still downloads a file for
<code>curate_frames.py --authority html</code> if you prefer that route.</p>
<div class="grid">
{chr(10).join(cards)}
</div></main>
<script>const ID={json.dumps(identity)};const SEED_STAMP={json.dumps(stamp)};{_JS}</script>
</body></html>"""


def _lighting(species: str) -> dict[str, dict]:
    """Per-bat lighting scores, if `rank_bats_by_lighting.py` has been run."""
    p = pathlib.Path("outputs/quality/bat_lighting.json")
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {b["identity"]: b for b in d.get("bats", [])}


def _index(species: str, per: list[tuple[str, int, int, int]]) -> str:
    """The worklist.

    Two orderings compete here, and lighting wins. Reviewing a bat whose lighting
    does not match the rousettus set is wasted effort no matter how many frames it
    has -- it would be excluded from the matched arm anyway -- so bats are ranked by
    |contrast - rousettus| first and by review state second. The filter hides the
    poor matches outright, which is the point: the reviewer should not have to know
    which bats are worth their time.
    """
    light = _lighting(species)

    def key(r):
        lit = light.get(r[0], {})
        return (
            0 if lit.get("lighting_match") == "good" else 1,
            lit.get("distance_to_rousettus", 1e9),
            r[2] > 0,
            r[0],
        )

    per = sorted(per, key=key)
    cells = []
    for i, n, k, u in per:
        lit = light.get(i, {})
        match = lit.get("lighting_match", "unknown")
        cls = []
        if not k:
            cls.append("todo")
        if match == "poor":
            cls.append("poor")
        cells.append(
            "<tr{cls} data-match='{m}'><td><a href='{i}.html'>{i}</a></td>"
            "<td class='n'>{n}</td><td class='n'>{k}</td><td class='n'>{u}</td>"
            "<td class='n'>{c}</td><td class='lm {m}'>{m}</td>"
            "<td class='st'>{st}</td></tr>".format(
                cls=f" class='{' '.join(cls)}'" if cls else "",
                m=match,
                i=i,
                n=n,
                k=k,
                u=u,
                c=f"{lit['contrast']:+.0f}" if "contrast" in lit else "–",
                st="reviewed" if k else "first pass needed",
            )
        )
    rows = "".join(cells)
    n_good = sum(1 for i, _, _, _ in per if light.get(i, {}).get("lighting_match") == "good")
    n_todo_good = sum(
        1 for i, _, k, _ in per if not k and light.get(i, {}).get("lighting_match") == "good"
    )
    tot, totk, totu = sum(p[1] for p in per), sum(p[2] for p in per), sum(p[3] for p in per)
    ref = ""
    if light:
        d = json.loads(pathlib.Path("outputs/quality/bat_lighting.json").read_text())
        ref = (
            f"Lighting match is |face−background contrast − rousettus "
            f"({d['reference']['contrast']:+.0f})|, "
            f"within {d['good_threshold']:.0f} = good."
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{species} · stage-1 review</title><style>{_CSS}
table{{border-collapse:collapse;background:var(--panel);border:1px solid var(--line);
  border-radius:10px;overflow:hidden;width:100%;max-width:780px}}
th,td{{padding:8px 14px;text-align:left;border-bottom:1px solid var(--line)}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
tr:last-child td{{border-bottom:none}}
tr.todo{{background:var(--und-bg)}}
tr.todo td:first-child{{box-shadow:inset 3px 0 0 var(--und)}}
td.st{{font-size:12px;color:var(--muted)}}
tr.todo td.st{{color:var(--und);font-weight:600}}
td.lm{{font-size:12px;font-weight:600}}
td.lm.good{{color:var(--keep)}}
td.lm.poor{{color:var(--drop)}}
td.lm.unknown{{color:var(--muted);font-weight:400}}
tr.poor{{opacity:.5}}
.bar{{display:flex;gap:10px;align-items:center;margin:0 0 14px;flex-wrap:wrap}}
label{{font-size:13px;color:var(--muted);display:flex;gap:6px;align-items:center;cursor:pointer}}
</style></head><body>
<header><h1>{species} · stage-1 frame review</h1>
<span class="sub">{len(per)} identities · {tot} frames · {totk} keep · {totu} undecided</span>
<span class="tallies">
  <span class="t keep">{n_good} good lighting</span>
  <span class="t undecided">{n_todo_good} of those need a first pass</span>
</span></header>
<main>
<p class="hint">Sorted by lighting match, best first — review from the top and stop
when you have enough bats. {ref} Poor matches would be dropped from the
species-matched arm anyway, so tagging them is wasted effort.</p>
<div class="bar">
  <label><input type="checkbox" id="hide" checked> Hide poor lighting matches</label>
  <label><input type="checkbox" id="todo"> Only bats needing a first pass</label>
  <span class="sub" id="count"></span>
</div>
<table><thead><tr><th>identity</th><th>frames</th><th>keep</th><th>undecided</th>
<th>contrast</th><th>lighting</th><th>status</th></tr></thead>
<tbody>{rows}</tbody></table></main>
<script>
const rows=[...document.querySelectorAll('tbody tr')];
const hide=document.getElementById('hide'),todo=document.getElementById('todo');
function apply(){{
  let shown=0;
  for(const r of rows){{
    const poor=r.dataset.match==='poor';
    const needs=r.classList.contains('todo');
    const ok=(!hide.checked||!poor)&&(!todo.checked||needs);
    r.hidden=!ok; if(ok)shown++;
  }}
  document.getElementById('count').textContent=shown+' of '+rows.length+' shown';
}}
hide.addEventListener('change',apply);todo.addEventListener('change',apply);apply();
</script>
</body></html>"""


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
