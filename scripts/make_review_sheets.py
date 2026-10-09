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
import re

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
figure:focus{outline:none}
/* Which card the keys will act on. Needed because a mouse click does not
   trigger :focus-visible, so after clicking there was no cursor to see. */
figure.current{box-shadow:0 0 0 3px var(--accent);position:relative}
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
// Keyboard handling lives on the DOCUMENT, not on each card.
// Per-card listeners only fire when that card holds focus, and on load nothing does
// -- so every keypress went to <body> and was silently dropped. A tracked cursor also
// means the keys work the moment the page opens, with no click or Tab first.
let cur=0;
function markCurrent(scroll){
  cards.forEach((c,i)=>c.classList.toggle('current',i===cur));
  const c=cards[cur];
  if(c){c.focus({preventScroll:true});
    if(scroll)c.scrollIntoView({block:'nearest',behavior:'smooth'});}
}
function move(step){
  // Skip cards the filter has hidden, so the cursor never lands out of sight.
  let i=cur, n=cards.length;
  for(let k=0;k<n;k++){
    i=(i+step+n)%n;
    if(!cards[i].hidden){cur=i;break}
  }
  markCurrent(true);
}
cards.forEach((c,i)=>{
  c.tabIndex=-1;
  c.addEventListener('click',()=>{cur=i;markCurrent(false);cycle(c)});
});
document.addEventListener('keydown',e=>{
  // Let the buttons and checkboxes keep their own keys.
  const tag=(e.target.tagName||'').toUpperCase();
  if(tag==='INPUT'||tag==='BUTTON'||tag==='TEXTAREA'||tag==='SELECT')return;
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  const c=cards[cur]; if(!c)return;
  const m={k:'keep',d:'drop',u:'undecided'}[e.key.toLowerCase()];
  if(m){set(c,m);move(1);e.preventDefault();return}
  if(e.key==='ArrowRight'||e.key==='ArrowDown'){move(1);e.preventDefault();return}
  if(e.key==='ArrowLeft'||e.key==='ArrowUp'){move(-1);e.preventDefault();return}
  if(e.key==='Enter'||e.key===' '){cycle(c);e.preventDefault()}
});
markCurrent(false);
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
document.getElementById('gate').addEventListener('click',()=>{
  // A starting point, not a decision. The automatic frontal filter already ruled on
  // every frame; this writes that ruling into the page so the job becomes "correct
  // the gate" rather than "judge 200 frames from scratch". Nothing is saved until
  // you press Save, and every card is still one click from changing.
  const n=cards.length;
  if(!confirm('Set all '+n+' frames to the automatic gate verdict? This only fills the page - review and correct it, then Save.'))return;
  for(const c of cards)store[c.dataset.key]=c.dataset.gate==='keep'?'keep':'drop';
  persist();paint();
  status.textContent='filled from the gate \u2014 review, then Save';
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


def seed_stamp(pairs: dict[str, str]) -> str:
    """Fingerprint of one bat's decisions: {frame key -> decision}.

    Shared with `serve_review.py`, which computes it straight from decisions.json to
    answer /_status. Both sides must agree, or the index would report unsaved edits
    that are not there (or miss ones that are), so there is exactly one definition.
    """
    body = "".join(f"{k}={pairs[k]};" for k in sorted(pairs))
    return hashlib.sha1(body.encode()).hexdigest()[:12]


def stamps_from_decisions(species: str) -> dict[str, str]:
    """Per-identity seed stamps, read from decisions.json."""
    dec = json.loads((CURATED / species / "decisions.json").read_text())
    per: dict[str, dict[str, str]] = {}
    for key, rec in dec["frames"].items():
        per.setdefault(rec["identity"], {})[key] = rec["decision"]
    return {i: seed_stamp(p) for i, p in per.items()}


def _extract_js(html: str) -> str:
    """Everything inside <script> tags, concatenated."""
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))


def _check_js(js: str, where: str) -> None:
    """Reject a script with an unterminated string literal.

    A literal newline inside a JS string is a syntax error, and the browser then
    runs NONE of the script -- every button and key on the page goes dead at once,
    with nothing visible until someone opens the console. That is exactly what
    shipped: a `\n` escape collapsed into a real newline while generating the
    "Apply gate" confirm text, and the page looked completely normal.

    This is not a JS parser. It walks the source tracking quote state and comments,
    and raises if a quote is still open at a line end -- which is the one failure
    mode this generator can actually produce.
    """
    quote = None
    in_block = False
    for n, line in enumerate(js.split("\n"), 1):
        i = 0
        while i < len(line):
            c = line[i]
            nxt = line[i + 1] if i + 1 < len(line) else ""
            if in_block:
                if c == "*" and nxt == "/":
                    in_block = False
                    i += 1
            elif quote:
                if c == "\\":
                    i += 1
                elif c == quote:
                    quote = None
            elif c == "/" and nxt == "/":
                break
            elif c == "/" and nxt == "*":
                in_block = True
                i += 1
            elif c in "\"'`":
                quote = c
            i += 1
        if quote and quote != "`":  # backticks legally span lines
            raise AssertionError(
                f"{where}: unterminated {quote} string at line {n}: {line.strip()[:70]}"
            )


def _page(identity: str, rows: list[dict]) -> str:
    stamp = seed_stamp({r["key"]: r["decision"] for r in rows})
    cards = []
    for r in rows:
        gate = "gate: keep" if r["auto_keep"] else (r["gate_reason"] or "gate: reject")
        cards.append(
            f'<figure data-key="{r["key"]}" data-seed="{r["decision"]}" '
            f'data-gate="{"keep" if r["auto_keep"] else "reject"}">'
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
  <button id="gate">Apply gate</button>
  <button id="allk">All keep</button>
  <button id="reset">Reset</button>
  <button id="save" class="primary">Save to server</button>
  <button id="exp">Export…</button>
  <span class="sub" id="status"></span>
</header>
<main>
<p class="hint">The blue ring shows which card the keys act on — it starts on the
first card, so you can type straight away without clicking.
<kbd>K</kbd> keep · <kbd>D</kbd> drop · <kbd>U</kbd> undecided (each advances to the
next), arrow keys move without deciding, <kbd>Enter</kbd> cycles. Clicking a card also
moves the ring there.
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
            lit.get("outside_rousettus_range", 1e9),
            r[2] > 0,
            r[0],
        )

    per = sorted(per, key=key)
    cells = []
    for i, n, k, u, dr in per:
        lit = light.get(i, {})
        match = lit.get("lighting_match", "unknown")
        cls = []
        if not k:
            cls.append("todo")
        if match == "poor":
            cls.append("poor")
        cells.append(
            "<tr{cls} data-match='{m}' data-id='{i}'><td><a href='{i}.html'>{i}</a></td>"
            "<td class='n' data-f='n'>{n}</td><td class='n keep' data-f='keep'>{k}</td>"
            "<td class='n drop' data-f='drop'>{d}</td>"
            "<td class='n und' data-f='undecided'>{u}</td>"
            "<td class='n'>{c}</td><td class='lm {m}'>{m}</td>"
            "<td class='sv'></td><td class='st'>{st}</td></tr>".format(
                cls=f" class='{' '.join(cls)}'" if cls else "",
                m=match,
                i=i,
                n=n,
                k=k,
                d=dr,
                u=u,
                c=f"{lit['contrast']:+.0f}" if "contrast" in lit else "–",
                st="reviewed" if k else "first pass needed",
            )
        )
    rows = "".join(cells)
    n_all = len(per)
    n_poor = sum(1 for i, _, _, _, _ in per if light.get(i, {}).get("lighting_match") == "poor")
    n_good = sum(1 for i, _, _, _, _ in per if light.get(i, {}).get("lighting_match") == "good")
    n_todo_good = sum(
        1 for i, _, k, _, _ in per if not k and light.get(i, {}).get("lighting_match") == "good"
    )
    tot = sum(p[1] for p in per)
    ref = ""
    if light:
        d = json.loads(pathlib.Path("outputs/quality/bat_lighting.json").read_text())
        ref = (
            f"Good = face−background contrast inside the range the rousettus set "
            f"itself spans ({d['reference']['contrast_range'][0]:+.0f} to "
            f"{d['reference']['contrast_range'][1]:+.0f}, median "
            f"{d['reference']['contrast_median']:+.0f}), measured inside the head crop."
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
tr.dirty{{outline:2px solid var(--drop);outline-offset:-2px;opacity:1}}
#unsaved a{{color:inherit}}
td.sv{{font-size:12px;font-weight:600}}
td.sv.ok{{color:var(--muted);font-weight:400}}
td.sv.pending{{color:var(--drop)}}
#live.stale{{color:var(--drop)}}
td.keep{{color:var(--keep)}} td.drop{{color:var(--drop)}} td.und{{color:var(--und)}}
.bar{{display:flex;gap:10px;align-items:center;margin:0 0 14px;flex-wrap:wrap}}
label{{font-size:13px;color:var(--muted);display:flex;gap:6px;align-items:center;cursor:pointer}}
</style></head><body>
<header><h1>{species} · stage-1 frame review</h1>
<span class="sub">{len(per)} identities · {tot} frames · <span id="live"></span></span>
<span class="tallies">
  <span class="t keep" id="tk">keep <b>–</b></span>
  <span class="t drop" id="td">drop <b>–</b></span>
  <span class="t undecided" id="tu">undecided <b>–</b></span>
  <span class="t" id="unsaved" hidden></span>
</span></header>
<main>
<p class="hint">Sorted by lighting match, best first — review from the top and stop
when you have enough bats. <strong>{n_good} bats match well, {n_todo_good} of those
still need a first pass.</strong> {ref} Poor matches would be dropped from the
species-matched arm anyway, so tagging them is wasted effort.
The counts below come live from the server; <em>saved</em> means this browser has no
edits the server has not seen.</p>
<div class="bar">
  <label><input type="checkbox" id="hide" checked> Hide poor lighting matches <span class="sub">({n_poor} of {n_all})</span></label>
  <label><input type="checkbox" id="todo"> Only bats needing a first pass</label>
  <span class="sub" id="count"></span>
</div>
<table><thead><tr><th>identity</th><th>frames</th><th>keep</th><th>drop</th><th>undecided</th>
<th>contrast</th><th>lighting</th><th>saved</th><th>status</th></tr></thead>
<tbody>{rows}</tbody></table></main>
<script>
const rows=[...document.querySelectorAll('tbody tr')];
const hide=document.getElementById('hide'),todo=document.getElementById('todo');
function apply(){{
  let shown=0,hiddenPoor=0;
  for(const r of rows){{
    const poor=r.dataset.match==='poor';
    const needs=r.classList.contains('todo');
    // Unsaved work always stays visible. Hiding it is how the banner ended up
    // reporting a bat the table no longer contained.
    const dirty=r.classList.contains('dirty');
    const ok=dirty||((!hide.checked||!poor)&&(!todo.checked||needs));
    r.hidden=!ok;
    if(ok)shown++; else if(poor)hiddenPoor++;
  }}
  document.getElementById('count').textContent=
    shown+' of '+rows.length+' shown'+(hiddenPoor?' · '+hiddenPoor+' poor-lighting bats hidden':'');
}}
hide.addEventListener('change',apply);todo.addEventListener('change',apply);apply();

// This page is static HTML, regenerated on save -- so a tab left open goes stale the
// moment another tab saves. Poll the server for the real counts instead of trusting
// what was baked in at render time.
const live=document.getElementById('live');
function pendingFor(id,stamp){{
  // A successful save reloads the sheet onto a new stamp, which clears that bat's
  // cache. So a non-empty cache still matching the server's stamp = edits not sent.
  try{{
    if(localStorage.getItem('batcuration:'+id+':seed')!==stamp)return 0;
    const s=JSON.parse(localStorage.getItem('batcuration:'+id)||'{{}}');
    return Object.keys(s).length;
  }}catch(e){{return 0}}
}}
async function refresh(){{
  let d;
  try{{d=await (await fetch('/_status',{{cache:'no-store'}})).json()}}
  catch(e){{live.textContent='offline — counts below are from the last render';
           live.className='stale';return}}
  document.querySelector('#tk b').textContent=d.totals.keep;
  document.querySelector('#td b').textContent=d.totals.drop;
  document.querySelector('#tu b').textContent=d.totals.undecided;
  const dirtyIds=[];
  for(const r of rows){{
    const b=d.bats[r.dataset.id]; if(!b)continue;
    for(const f of ['n','keep','drop','undecided']){{
      const td=r.querySelector("[data-f='"+f+"']"); if(td)td.textContent=b[f];
    }}
    const pend=pendingFor(r.dataset.id,b.stamp);
    const sv=r.querySelector('.sv');
    sv.textContent=pend?('● '+pend+' unsaved'):'saved';
    sv.className='sv '+(pend?'pending':'ok');
    r.classList.toggle('dirty',!!pend);
    if(pend)dirtyIds.push(r.dataset.id);
    r.classList.toggle('todo',b.keep===0);
  }}
  const u=document.getElementById('unsaved');
  u.hidden=!dirtyIds.length; u.className='t drop';
  // Name them. "1 bat with unsaved edits" with no way to find it is not an
  // indicator, it is a puzzle.
  u.innerHTML='unsaved: '+dirtyIds.map(
    id=>"<a href='"+id+".html'>"+id+"</a>").join(', ');
  live.textContent='live · updated '+new Date().toLocaleTimeString();
  live.className='';
  apply();
}}
refresh(); setInterval(refresh,10000);
addEventListener('focus',refresh);           // returning from a sheet tab
addEventListener('storage',refresh);         // another tab edited localStorage
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
        html = _page(identity, rows)
        _check_js(_extract_js(html), f"{identity}.html")
        (out / f"{identity}.html").write_text(html)
        per.append(
            (
                identity,
                len(rows),
                sum(1 for r in rows if r["decision"] == "keep"),
                sum(1 for r in rows if r["decision"] == "undecided"),
                sum(1 for r in rows if r["decision"] == "drop"),
            )
        )
    idx = _index(species, per)
    _check_js(_extract_js(idx), "index.html")
    (out / "index.html").write_text(idx)
    return len(per)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", default="mauritius")
    a = ap.parse_args()
    n = build_sheets(a.species)
    print(f"  wrote {n} sheet(s) -> {CURATED / a.species / 'review'}/index.html")
