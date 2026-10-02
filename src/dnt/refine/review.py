"""Static review page for HUMAN_PENDING events (spec 8.1)."""

from __future__ import annotations

import contextlib
import html
import json
import logging
import os
from pathlib import Path

from .events import Decision, Event, EventKind

log = logging.getLogger(__name__)

_CSS = """
body{font:14px/1.4 system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}
.card{background:#fff;border:1px solid #ccc;border-radius:6px;margin:12px 0;padding:10px}
.card img{max-width:100%;border:1px solid #ddd}
.meta span{margin-right:12px}.sig{color:#555}pre{background:#f0f0f0;padding:6px;overflow:auto}
.bar{position:sticky;top:0;background:#fafafa;padding:6px 0}
"""

_JS_PURE = """
function storageKey(run) { return "dnt-refine-review:" + run; }
function restorable(saved, id, proposalKey) {
  var s = saved && saved[id];
  if (!s || s.key !== proposalKey) { return null; }
  return (s.choice === "accept" || s.choice === "reject") ? s : null;
}
function exportDecisions(rows) {
  var out = {};
  rows.forEach(function (r) {
    if (r.choice === "accept") {
      out[r.id] = r.cls ? {accept: true, new_cls: parseInt(r.cls, 10)} : "accept";
    } else if (r.choice === "reject") { out[r.id] = "reject"; }
  });
  return out;
}
"""

_JS_DOM = """
(function () {
  var KEY = storageKey(document.body.dataset.run);
  var cards = [].slice.call(document.querySelectorAll(".card"));
  var saved = {};
  try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
  function choice(c) {
    var r = c.querySelector("input[type=radio]:checked");
    var sel = c.querySelector("select.cls");
    if (!r) { return null; }
    return {id: c.dataset.id, key: c.dataset.key, choice: r.value, cls: sel ? sel.value : ""};
  }
  cards.forEach(function (c) {
    var s = restorable(saved, c.dataset.id, c.dataset.key);
    if (!s) { return; }
    var r = c.querySelector('input[value="' + s.choice + '"]');
    if (r) { r.checked = true; }
    var sel = c.querySelector("select.cls");
    if (sel && s.cls) { sel.value = s.cls; }
  });
  document.addEventListener("change", function () {
    var s = {};
    cards.forEach(function (c) { var v = choice(c); if (v) { s[c.dataset.id] = v; } });
    try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) {}
  });
  document.getElementById("export").onclick = function () {
    var rows = cards.map(choice).filter(Boolean);
    var blob = new Blob([JSON.stringify(exportDecisions(rows), null, 2)],
                        {type: "application/json"});
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "decisions.json"; a.click();
  };
  function refresh() {
    var stage = document.getElementById("stage-filter").value;
    var key = document.getElementById("sort").value;
    var box = document.getElementById("cards");
    cards.sort(function (a, b) {
      return key === "score-asc" ? a.dataset.score - b.dataset.score
           : key === "score-desc" ? b.dataset.score - a.dataset.score : 0;
    }).forEach(function (c) {
      box.appendChild(c);
      c.style.display = (!stage || c.dataset.stage === stage) ? "" : "none";
    });
  }
  document.getElementById("stage-filter").onchange = refresh;
  document.getElementById("sort").onchange = refresh;
  [].forEach.call(document.querySelectorAll("button.copy"), function (b) {
    b.onclick = function () { navigator.clipboard.writeText(b.nextElementSibling.textContent); };
  });
})();
"""
_JS = _JS_PURE + _JS_DOM


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _top_signals(signals: dict, n: int = 6) -> list[str]:
    nums = [
        (k, v)
        for k, v in signals.items()
        if isinstance(v, int | float) and not isinstance(v, bool) and v == v
    ]
    nums.sort(key=lambda kv: -abs(kv[1]))
    return [f"{k}={v:.3f}" for k, v in nums[:n]]


def _reason(ev: Event) -> str:
    return str(ev.params.get("reason") or ev.signals.get("hypothesis") or ev.kind)


def _snippet(ev: Event, id_map: dict, fps: float, video_file, track_file) -> str:
    ids = [i for i in (id_map.get(t, id_map.get(str(t))) for t in ev.tracks) if i is not None]
    margin = round(2.0 * fps)
    # the clip spans the selected tracks' first to last frame, plus a 2 s margin on each side
    return (
        "labeler.draw_track_clips(\n"
        f"    input_video={str(video_file)!r}, output_path='clips/{ev.id}',\n"
        f"    track_file={str(track_file)!r}, method='specify', track_ids={ids},\n"
        f"    start_frame_offset={margin}, end_frame_offset={margin},\n"
        ")"
    )


def _extras(ev: Event) -> str:
    out = []
    for alt in ev.signals.get("alternatives") or []:
        out.append(f"<div>next best: {_e(alt['i'])} &rarr; {_e(alt['j'])} {alt['score']:.2f}</div>")
    segs = ev.signals.get("segments")
    if segs:
        parts = ", ".join(
            f"{int(a)}-{int(b)} " + ("-" if c is None else f"{c:.2f}") for a, b, c in segs
        )
        out.append(f"<div>segments (score): {_e(parts)}</div>")
    return "".join(out)


def _vlm_block(ev: Event) -> str:
    v = ev.vlm
    if not v:
        return ""
    ans = "-" if v.get("answer") is None else _e(v["answer"])
    err = f" error: {_e(v['error'])}" if v.get("error") else ""
    return (
        f'<div class="vlm">VLM {_e(v.get("backend"))}/{_e(v.get("model"))}: {ans} '
        f"({float(v.get('confidence') or 0):.2f}) {_e(v.get('reason', ''))}{err}</div>"
    )


def _picker(ev: Event, reclass_map: dict) -> str:
    rider = (ev.vlm or {}).get("answer") in ("cyclist", "motorcycle_rider", "scooter_rider")
    if ev.kind is not EventKind.RECLASS and not (ev.stage == "screen" and rider):
        return ""
    opts = '<option value="">(keep)</option>' + "".join(
        f'<option value="{_e(c)}">{_e(name)} ({_e(c)})</option>' for name, c in reclass_map.items()
    )
    return f'<label>class <select class="cls">{opts}</select></label>'


def _card(ev: Event, img_rel: str | None, snippet: str, reclass_map: dict) -> str:
    image = f'<img src="{_e(img_rel)}" alt="evidence">' if img_rel else "<div>no image</div>"
    name = _e(ev.id)
    return (
        f'<div class="card" data-id="{name}" data-key="{_e(ev.proposal_key)}" '
        f'data-stage="{_e(ev.stage)}" data-score="{ev.algo_score:.3f}">'
        f'{image}<div class="meta"><span><b>{_e(ev.kind)}</b> {_e(_reason(ev))}</span>'
        f"<span>tracks {_e(ev.tracks)}</span><span>frames {_e(ev.frames[0])}-{_e(ev.frames[1])}"
        f"</span><span>score {ev.algo_score:.3f}</span></div>"
        f'<div class="sig">{_e(" ".join(_top_signals(ev.signals)))}</div>'
        f"{_vlm_block(ev)}{_extras(ev)}"
        f'<div><label><input type="radio" name="{name}" value="accept"> accept</label> '
        f'<label><input type="radio" name="{name}" value="reject"> reject</label> '
        f"{_picker(ev, reclass_map)}</div>"
        f'<button class="copy" type="button">copy clip snippet</button><pre>{_e(snippet)}</pre>'
        "</div>"
    )


_MANIFEST = ".dnt-review.json"  # the report's own name: a user's manifest.json is never touched


def _listed_images(img_dir: Path) -> list[str]:
    """Return the image names the previous run wrote here (plain ``*.jpg`` names), or ``[]``.

    Anything that is not a JSON object with a list of names proves no ownership.
    """
    try:
        names = json.loads((img_dir / _MANIFEST).read_text())["images"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    if not isinstance(names, list):
        return []
    return [n for n in names if isinstance(n, str) and n == Path(n).name and n.endswith(".jpg")]


def _remove_listed(img_dir: Path, keep: set[str] = frozenset()) -> None:
    """Delete the images the manifest lists (except ``keep``); never anything else."""
    for name in _listed_images(img_dir):
        if name not in keep:
            (img_dir / name).unlink(missing_ok=True)


def _remove_own_files(review_path: Path, img_dir: Path) -> None:
    review_path.unlink(missing_ok=True)
    if img_dir.is_dir():
        owned = _listed_images(img_dir)
        _remove_listed(img_dir)
        if owned:  # a manifest that proves nothing is left where it is
            (img_dir / _MANIFEST).unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            img_dir.rmdir()  # only if nothing else lives there


def write_review(
    events: list[Event],
    *,
    review_path,
    evidence,
    id_map: dict,
    fps: float,
    video_file,
    track_file,
    reclass_map: dict,
    title: str,
    run_key: str,
) -> Path | None:
    """Write ``OUT.review.html`` and ``OUT.review/*.jpg`` for the pending events (spec 8.1).

    Returns the page path, or ``None`` (after removing this output's stale page and images)
    when no event is pending.
    """
    review_path = Path(review_path)
    img_dir = review_path.parent / review_path.name.removesuffix(".html")
    pend = [
        e
        for e in events
        if e.decision is Decision.HUMAN_PENDING and e.kind not in (EventKind.FILL, EventKind.SMOOTH)
    ]
    if not pend:
        _remove_own_files(review_path, img_dir)
        return None
    images: dict = {}
    if evidence is not None:
        try:
            images = evidence.build_many(pend)
        except Exception as err:  # a damaged video must not stop the review being written
            log.warning("could not build the review images: %s", err)
    cards, stages = [], sorted({e.stage for e in pend})
    written: list[str] = []
    owned = set(_listed_images(img_dir))
    for ev in pend:
        rel = None
        data = images.get(ev.id)
        name = f"{ev.id}.jpg"
        if data is not None and name not in owned and os.path.lexists(img_dir / name):
            log.warning("not overwriting %s: it is not an image of this report", img_dir / name)
            data = None  # the card is shown without an image; the foreign file stays intact
        if data is not None:
            img_dir.mkdir(parents=True, exist_ok=True)
            (img_dir / name).write_bytes(data)
            written.append(name)
            rel = f"{img_dir.name}/{ev.id}.jpg"
            if ev.vlm:
                ev.vlm["evidence"] = rel
        snippet = _snippet(ev, id_map, fps, video_file, track_file)
        cards.append(_card(ev, rel, snippet, reclass_map))
    if written or owned:  # never start a manifest in a directory that holds none of our images
        _remove_listed(img_dir, keep=set(written))  # images of events that are gone
        img_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / _MANIFEST).write_text(json.dumps({"images": written}))
    stage_opts = '<option value="">all stages</option>' + "".join(
        f'<option value="{_e(s)}">{_e(s)}</option>' for s in stages
    )
    page = (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{_e(title)}</title><style>{_CSS}</style></head>"
        f'<body data-run="{_e(run_key)}">'
        f"<h2>{_e(title)}: {len(pend)} event(s) to review</h2>"
        '<div class="bar"><select id="stage-filter">' + stage_opts + "</select> "
        '<select id="sort"><option value="score-desc">score, high first</option>'
        '<option value="score-asc">score, low first</option><option value="order">order</option>'
        '</select> <button id="export" type="button">Export decisions</button></div>'
        '<div id="cards">' + "".join(cards) + f"</div><script>{_JS}</script></body></html>"
    )
    review_path.parent.mkdir(parents=True, exist_ok=True)  # refine writes this before the ledger
    review_path.write_text(page, encoding="utf-8")
    return review_path
