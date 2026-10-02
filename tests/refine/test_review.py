import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dnt.refine.events import Decision, Event, EventKind
from dnt.refine.review import _JS_PURE, review_image_dir, write_review
from dnt.refine.verify import decide

RECLASS_MAP = {"cyclist": 1, "motorcycle": 3, "scooter": 36}


class Images:
    def build_many(self, events):
        return {e.id: b"\xff\xd8fakejpeg" for e in events}


def pending(kind, stage, idx, score=0.6, tracks=(1,), signals=None, vlm=None, **params):
    ev = Event.propose(stage=stage, kind=kind, tracks=list(tracks),
                       lineage=[[[t, 0, 99]] for t in tracks], frames=(10, 90), params=params,
                       algo_score=score, signals=signals or {})
    ev.id = f"{stage}-r0-{idx:06d}"
    ev.vlm = vlm
    decide(ev, Decision.HUMAN_PENDING, source="auto")
    return ev


IMAGES = Images()


def write(tmp_path, events, evidence=IMAGES, **kw):
    args = dict(review_path=tmp_path / "o.review.html", evidence=evidence,
                id_map={1: 1, 2: 2}, fps=10.0, video_file="/v/cam.mp4",
                track_file=tmp_path / "o.txt", reclass_map=RECLASS_MAP, title="o",
                run_key="run-1")
    args.update(kw)
    return write_review(events, **args)


def test_one_card_per_pending_event_and_none_for_decided_or_position_records(tmp_path):
    a = pending(EventKind.LINK, "link", 1, tracks=(1, 2), gap=[50, 60], gate="normal")
    b = pending(EventKind.SPLIT, "switch", 2, cut_frame=40)
    done = pending(EventKind.SPLIT, "switch", 3, cut_frame=70)
    decide(done, Decision.AUTO_ACCEPT, source="auto")
    fill = pending(EventKind.FILL, "fill", 4, gap=[5, 9], n_rows=3)
    html_path = write(tmp_path, [a, b, done, fill])
    html = html_path.read_text()
    assert len(re.findall(r'class="card"', html)) == 2
    assert 'data-id="link-r0-000001"' in html and 'data-id="switch-r0-000002"' in html
    assert "switch-r0-000003" not in html and "fill-r0-000004" not in html
    for ev in (a, b):
        img = tmp_path / "o.review" / f"{ev.id}.jpg"
        assert img.read_bytes().startswith(b"\xff\xd8")
        assert f'src="o.review/{ev.id}.jpg"' in html


def test_the_page_is_static_and_escapes_every_text(tmp_path):
    ev = pending(EventKind.LINK, "link", 1, tracks=(1, 2), gap=[50, 60], gate="normal",
                 vlm={"backend": "b<script>", "model": "m", "answer": "unsure", "confidence": 0.4,
                      "votes": {}, "reason": "<script>alert(1)</script>", "evidence": None,
                      "cached": False, "error": "x\"><img src=y onerror=z>"})
    html = write(tmp_path, [ev], title="<b>t</b>").read_text()
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "onerror=z>" not in html
    assert "<b>t</b>" not in html
    assert not re.search(r"https?://", html)
    assert "decisions.json" in html and "localStorage" in html and "Export decisions" in html


def test_the_image_path_is_recorded_on_the_vlm_record(tmp_path):
    vlm = {"backend": "b", "model": "m", "answer": "unsure", "confidence": 0.4, "votes": {},
           "reason": "r", "evidence": None, "cached": False, "error": None}
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40, vlm=vlm)
    other = pending(EventKind.SPLIT, "switch", 2, cut_frame=60)
    write(tmp_path, [ev, other])
    assert ev.vlm["evidence"] == "o.review/switch-r0-000001.jpg"
    assert other.vlm is None  # no VLM record, none is invented


def test_without_a_video_cards_show_signals_only(tmp_path):
    ev = pending(EventKind.SPLIT, "switch", 1, signals={"mot": 0.75, "app": 0.0}, cut_frame=40)
    html = write(tmp_path, [ev], evidence=None).read_text()
    assert "<img" not in html and "no image" in html and "mot" in html
    assert not (tmp_path / "o.review").exists() or not list((tmp_path / "o.review").glob("*.jpg"))


def test_reclass_card_has_a_class_picker_and_a_link_card_shows_alternatives(tmp_path):
    rc = pending(EventKind.RECLASS, "screen", 1, new_cls=None, spans=None,
                 signals={"hypothesis": "rider", "needs_subtype": True})
    lk = pending(EventKind.LINK, "link", 2, tracks=(1, 2), gap=[50, 60], gate="normal",
                 signals={"alternatives": [{"i": 1, "j": 3, "score": 0.55}]})
    seg = pending(EventKind.DROP, "screen", 3, reason="static", spans=[[10, 40]],
                  signals={"segments": [[10, 40, 0.8], [41, 90, 0.1]], "hypothesis": "static"})
    html = write(tmp_path, [rc, lk, seg]).read_text()
    picker = re.findall(r'<select class="cls">(.*?)</select>', html, re.S)
    assert len(picker) == 1 and "cyclist" in picker[0] and 'value="36"' in picker[0]
    assert "0.55" in html and "1 &rarr; 3" in html
    assert "10-40" in html and "0.80" in html and "0.10" in html


def test_the_labeler_snippet_names_the_output_tracks_and_the_video(tmp_path):
    ev = pending(EventKind.LINK, "link", 1, tracks=(1, 2), gap=[50, 60], gate="normal")
    html = write(tmp_path, [ev], id_map={1: 7, 2: 9}).read_text()  # renumber() returns int keys
    assert "draw_track_clips(" in html and "/v/cam.mp4" in html
    assert "[7, 9]" in html and "frame_offset" in html
    # the ledger header stores string keys (JSON); a map read back from it works as well
    assert "[7, 9]" in write(tmp_path, [ev], id_map={"1": 7, "2": 9}).read_text()
    # a track that no longer exists in the output has no id; it is left out, never invented
    assert "track_ids=[9]" in write(tmp_path, [ev], id_map={2: 9}).read_text()


def test_cards_and_the_page_carry_the_proposal_key_and_the_run_key(tmp_path):
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    html = write(tmp_path, [ev], run_key="abc123").read_text()
    assert f'data-key="{ev.proposal_key}"' in html and 'data-run="abc123"' in html
    other = pending(EventKind.SPLIT, "switch", 1, cut_frame=55)  # same id, a different proposal
    assert other.id == ev.id and other.proposal_key != ev.proposal_key
    html2 = write(tmp_path, [other], run_key="abc123").read_text()
    assert f'data-key="{other.proposal_key}"' in html2 and ev.proposal_key not in html2


NODE = shutil.which("node")


def run_node(body: str):
    out = subprocess.run([NODE, "-e", _JS_PURE + "\n" + body], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_saved_choice_is_restored_only_for_the_proposal_it_was_made_on():
    body = """
    var saved = {"switch-r0-000001": {choice: "accept", cls: "3", key: "AAA"}};
    console.log(JSON.stringify([
      restorable(saved, "switch-r0-000001", "AAA"),
      restorable(saved, "switch-r0-000001", "BBB"),
      restorable(saved, "switch-r0-000002", "AAA"),
      restorable({x: {choice: "maybe", key: "K"}}, "x", "K"),
      restorable({x: {choice: "reject"}}, "x", "K"),
      restorable(null, "x", "K")
    ]));"""
    got = run_node(body)
    assert got[0] == {"choice": "accept", "cls": "3", "key": "AAA"}
    assert got[1:] == [None] * 5  # a different proposal, another id, a bad choice, no key, no store


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_storage_key_is_namespaced_by_the_run_and_decisions_export_in_the_spec_format():
    body = """
    console.log(JSON.stringify({
      a: storageKey("r1"), b: storageKey("r2"),
      out: exportDecisions([
        {id: "e1", key: "K1", run: "R", choice: "accept", cls: ""},
        {id: "e2", key: "K2", run: "R", choice: "accept", cls: "36"},
        {id: "e3", key: "K3", run: "R", choice: "reject", cls: "3"},
        {id: "e4", key: "K4", run: "R", choice: null, cls: ""}
      ])
    }));"""
    got = run_node(body)
    assert got["a"] != got["b"] and got["a"].endswith("r1")
    # objects that name the proposal and the run: an event id alone is reused after a rerun
    assert got["out"] == {
        "e1": {"accept": True, "proposal_key": "K1", "run_key": "R"},
        "e2": {"accept": True, "new_cls": 36, "proposal_key": "K2", "run_key": "R"},
        "e3": {"accept": False, "proposal_key": "K3", "run_key": "R"},
    }


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_export_button_writes_each_cards_proposal_key_and_the_pages_run_key(tmp_path):
    a = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    b = pending(EventKind.RECLASS, "screen", 2, new_cls=None, spans=None)
    c = pending(EventKind.SPLIT, "switch", 3, cut_frame=50)
    page = write(tmp_path, [a, b, c], run_key="run-xyz").read_text()
    script = page.split("<script>", 1)[1].split("</script>", 1)[0]
    cards = [
        {"id": a.id, "key": a.proposal_key, "choice": "reject", "cls": None},
        {"id": b.id, "key": b.proposal_key, "choice": "accept", "cls": "36"},
        {"id": c.id, "key": c.proposal_key, "choice": None, "cls": None},
    ]
    dom = "var CARDS = " + json.dumps(cards) + """, exported = [];
    function card(d) {
      return {dataset: {id: d.id, key: d.key, stage: "s", score: "0.5"}, style: {},
        querySelector: function (sel) {
          if (sel === "input[type=radio]:checked") { return d.choice ? {value: d.choice} : null; }
          if (sel === "select.cls") { return d.cls ? {value: d.cls} : null; }
          return null;
        }};
    }
    var cards = CARDS.map(card), els = {};
    ["export", "stage-filter", "sort"].forEach(function (k) { els[k] = {value: ""}; });
    els.cards = {appendChild: function () {}};
    globalThis.document = {
      body: {dataset: {run: "run-xyz"}},
      querySelectorAll: function (sel) { return sel === ".card" ? cards : []; },
      getElementById: function (k) { return els[k]; },
      addEventListener: function () {},
      createElement: function () { return {click: function () {}}; }
    };
    globalThis.localStorage = {getItem: function () { return null; }, setItem: function () {}};
    globalThis.Blob = function (parts) { exported.push(parts.join("")); };
    globalThis.URL = {createObjectURL: function () { return "blob:x"; }};
    """
    out = subprocess.run(
        [NODE, "-e", dom + script + "\nels.export.onclick(); console.log(exported[0]);"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == {
        a.id: {"accept": False, "proposal_key": a.proposal_key, "run_key": "run-xyz"},
        b.id: {"accept": True, "new_cls": 36, "proposal_key": b.proposal_key, "run_key": "run-xyz"},
    }


def test_cards_are_filterable_and_sortable(tmp_path):
    a = pending(EventKind.LINK, "link", 1, score=0.5, tracks=(1, 2), gap=[5, 9], gate="normal")
    b = pending(EventKind.SPLIT, "switch", 2, score=0.7, cut_frame=40)
    html = write(tmp_path, [a, b]).read_text()
    assert 'data-stage="link"' in html and 'data-stage="switch"' in html
    assert 'data-score="0.500"' in html and 'data-score="0.700"' in html
    assert 'id="stage-filter"' in html and 'id="sort"' in html


def test_a_failing_evidence_builder_still_writes_a_signals_only_page(tmp_path):
    class Broken:
        def build_many(self, events):
            raise OSError("video went away")

    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    html = write(tmp_path, [ev], evidence=Broken()).read_text()
    assert "no image" in html and f'data-id="{ev.id}"' in html


def test_nothing_pending_returns_none_and_removes_only_its_own_files(tmp_path):
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    html_path = write(tmp_path, [ev])
    d = tmp_path / "o.review"
    assert html_path.exists() and (d / f"{ev.id}.jpg").exists() and (d / ".dnt-review.json").exists()
    assert json.loads((d / ".dnt-review.json").read_text()) == {"images": [f"{ev.id}.jpg"]}
    (d / "keep.jpg").write_bytes(b"my photo")
    (d / "keep.txt").write_text("my note")
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev]) is None
    assert not html_path.exists() and not (d / f"{ev.id}.jpg").exists()
    assert not (d / ".dnt-review.json").exists()
    assert (d / "keep.jpg").read_bytes() == b"my photo" and (d / "keep.txt").exists()
    (d / "keep.jpg").unlink()
    (d / "keep.txt").unlink()
    ev2 = pending(EventKind.SPLIT, "switch", 2, cut_frame=40)
    write(tmp_path, [ev2])
    decide(ev2, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev2]) is None
    assert not d.exists()  # nothing of the user's lives there: the directory goes too


def test_without_a_manifest_no_image_is_deleted(tmp_path):
    d = tmp_path / "o.review"
    d.mkdir()
    (d / "stranger.jpg").write_bytes(b"x")
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    (tmp_path / "o.review.html").write_text("old page")
    assert write(tmp_path, [ev]) is None
    assert (d / "stranger.jpg").exists() and not (tmp_path / "o.review.html").exists()


def test_a_rerun_replaces_the_previous_images_it_owns_and_nothing_else(tmp_path):
    first = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    write(tmp_path, [first])
    d = tmp_path / "o.review"
    (d / "keep.jpg").write_bytes(b"mine")
    second = pending(EventKind.SPLIT, "switch", 2, cut_frame=40)
    write(tmp_path, [second])
    assert not (d / f"{first.id}.jpg").exists() and (d / f"{second.id}.jpg").exists()
    assert (d / "keep.jpg").read_bytes() == b"mine"
    assert json.loads((d / ".dnt-review.json").read_text()) == {"images": [f"{second.id}.jpg"]}


def test_a_foreign_file_with_an_image_name_is_never_overwritten_or_deleted(tmp_path, caplog):
    d = tmp_path / "o.review"
    d.mkdir()
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    mine = d / f"{ev.id}.jpg"
    mine.write_bytes(b"the user's own file")  # same name as a generated image, no manifest
    with caplog.at_level("WARNING"):
        html = write(tmp_path, [ev]).read_text()
    assert mine.read_bytes() == b"the user's own file" and "no image" in html
    assert "not overwriting" in caplog.text and not (d / ".dnt-review.json").exists()
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev]) is None  # cleanup deletes nothing it never wrote
    assert mine.read_bytes() == b"the user's own file"
    # a manifest that lists other names does not make this file ours either
    (d / ".dnt-review.json").write_text(json.dumps({"images": ["other.jpg"]}))
    ev2 = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    write(tmp_path, [ev2])
    assert mine.read_bytes() == b"the user's own file"
    # nothing was written and the manifest was ours (valid), so it goes; the foreign file stays
    assert not (d / ".dnt-review.json").exists() and mine.read_bytes() == b"the user's own file"


def test_a_users_manifest_json_is_left_alone(tmp_path):
    d = tmp_path / "o.review"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"images": ["x.jpg"]}))
    (d / "x.jpg").write_bytes(b"x")
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    write(tmp_path, [ev])
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev]) is None
    assert (d / "x.jpg").exists() and json.loads((d / "manifest.json").read_text())["images"] == [
        "x.jpg"
    ]


BAD_MANIFESTS = [
    "null", "1", '"text"', "[]", '{"images": null}', '{"images": 1}', '{"images": "a.jpg"}',
    '{"images": {"a.jpg": 1}}', '{"images": [1, null, "../x.jpg", "a.txt"]}', "{not json", "",
]


@pytest.mark.parametrize("text", BAD_MANIFESTS)
def test_a_damaged_manifest_never_breaks_the_review(tmp_path, text):
    d = tmp_path / "o.review"
    d.mkdir()
    (d / ".dnt-review.json").write_text(text)
    (d / "keep.jpg").write_bytes(b"mine")
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    assert write(tmp_path, [ev]) is not None  # a pending run replaces the damaged manifest
    assert json.loads((d / ".dnt-review.json").read_text()) == {"images": [f"{ev.id}.jpg"]}
    assert (d / "keep.jpg").read_bytes() == b"mine"
    (d / ".dnt-review.json").write_text(text)  # damaged again, then nothing is pending
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev]) is None
    assert (d / "keep.jpg").read_bytes() == b"mine"
    assert not (tmp_path / "o.review.html").exists()


def test_a_manifest_cannot_point_outside_the_review_directory(tmp_path):
    victim = tmp_path / "precious.jpg"
    victim.write_bytes(b"precious")
    d = tmp_path / "o.review"
    d.mkdir()
    (d / ".dnt-review.json").write_text(json.dumps({"images": ["../precious.jpg", "/etc/passwd"]}))
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    write(tmp_path, [ev])
    assert victim.read_bytes() == b"precious"


def test_an_input_inside_the_review_directory_is_rejected(tmp_path):
    from dnt.refine.refiner import check_output_paths

    d = tmp_path / "o.review"
    d.mkdir()
    for name in (".dnt-review.json", "e1.jpg", "my-photo.jpg"):
        with pytest.raises(ValueError, match="review"):
            check_output_paths(tmp_path / "o.txt", {"video_file": d / name})
    check_output_paths(tmp_path / "o.txt", {"video_file": tmp_path / "o.review.mp4"})  # a sibling


def test_the_output_directory_is_created_when_missing(tmp_path):
    # refine writes the review before the ledger, which is what creates the directory
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    page = tmp_path / "new" / "dir" / "o.review.html"
    assert write(tmp_path, [ev], review_path=page) == page
    assert page.is_file() and (page.parent / "o.review" / f"{ev.id}.jpg").is_file()
    nothing = tmp_path / "other" / "o.review.html"
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev], review_path=nothing) is None  # no directory is made for nothing
    assert not nothing.parent.exists()


def test_a_segment_without_a_score_does_not_crash_the_page(tmp_path):
    # a screen event keeps None for a segment where the hypothesis has no score (e.g. duplicate)
    ev = pending(EventKind.DROP, "screen", 1, reason="duplicate", spans=[[10, 40]], of=2,
                 signals={"segments": [[10, 40, 0.8], [41, 90, None]], "hypothesis": "duplicate"})
    html = write(tmp_path, [ev]).read_text()
    assert "10-40 0.80" in html and "41-90 -" in html


def test_an_empty_valid_manifest_is_ours_and_does_not_outlive_the_pending_events(tmp_path):
    d = tmp_path / "o.review"
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    write(tmp_path, [ev])  # run 1, with a video: an image and a manifest
    assert (d / f"{ev.id}.jpg").exists()
    write(tmp_path, [ev], evidence=None)  # run 2, no video, still pending: no image, no manifest
    assert not (d / f"{ev.id}.jpg").exists() and not (d / ".dnt-review.json").exists()
    assert not d.exists()
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev]) is None  # run 3, nothing pending
    assert not d.exists()


def test_the_same_sequence_keeps_a_foreign_file_and_drops_the_manifest(tmp_path):
    d = tmp_path / "o.review"
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    write(tmp_path, [ev])
    (d / "keep.jpg").write_bytes(b"mine")
    write(tmp_path, [ev], evidence=None)
    assert sorted(p.name for p in d.iterdir()) == ["keep.jpg"]
    decide(ev, Decision.AUTO_ACCEPT, source="auto")
    assert write(tmp_path, [ev]) is None
    assert sorted(p.name for p in d.iterdir()) == ["keep.jpg"]


def test_a_failure_while_writing_images_leaves_them_listed_for_the_next_run(tmp_path, monkeypatch):
    evs = [pending(EventKind.SPLIT, "switch", i, cut_frame=40) for i in (1, 2)]
    d = tmp_path / "o.review"
    real = Path.write_bytes
    calls = []

    def flaky(self, data):
        calls.append(self.name)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", flaky)
    with pytest.raises(OSError, match="disk full"):
        write(tmp_path, evs)
    monkeypatch.undo()
    assert (d / f"{evs[0].id}.jpg").exists()
    listed = json.loads((d / ".dnt-review.json").read_text())["images"]
    assert f"{evs[0].id}.jpg" in listed  # the first image is ours, not a stranger
    write(tmp_path, evs[1:])  # a rerun owns it: the stale first image is cleaned up
    assert not (d / f"{evs[0].id}.jpg").exists() and (d / f"{evs[1].id}.jpg").exists()


def test_a_file_named_like_the_image_directory_is_rejected_up_front(tmp_path):
    from dnt.refine.refiner import check_output_paths

    (tmp_path / "o.review").write_text("a file, not a directory")
    with pytest.raises(ValueError, match="not a directory"):
        check_output_paths(tmp_path / "o.txt", {"video_file": None})
    assert review_image_dir(tmp_path / "o.review.html") == tmp_path / "o.review"


def test_the_image_link_is_url_encoded(tmp_path):
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    page = tmp_path / "cam#2 a.review.html"
    html = write(tmp_path, [ev], review_path=page).read_text()
    assert f'src="cam%232%20a.review/{ev.id}.jpg"' in html
    assert (tmp_path / "cam#2 a.review" / f"{ev.id}.jpg").is_file()


def test_without_a_video_there_is_no_clip_snippet(tmp_path):
    ev = pending(EventKind.SPLIT, "switch", 1, cut_frame=40)
    html = write(tmp_path, [ev], video_file=None).read_text()
    assert "draw_track_clips" not in html and "None" not in html
    assert "draw_track_clips" in write(tmp_path, [ev]).read_text()


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_sorting_can_return_to_the_document_order_and_never_mutates_its_input():
    body = """
    var rows = [{id: "a", score: 0.5, index: 0}, {id: "b", score: 0.9, index: 1},
                {id: "c", score: 0.5, index: 2}];
    var ids = function (r) { return r.map(function (x) { return x.id; }); };
    console.log(JSON.stringify({
      desc: ids(sortCards(rows, "score-desc")), asc: ids(sortCards(rows, "score-asc")),
      order: ids(sortCards(sortCards(rows, "score-desc"), "order")), same: ids(rows)
    }));"""
    got = run_node(body)
    assert got == {"desc": ["b", "a", "c"], "asc": ["a", "c", "b"],
                   "order": ["a", "b", "c"], "same": ["a", "b", "c"]}


def test_cards_show_the_output_track_ids_and_load_images_lazily(tmp_path):
    ev = pending(EventKind.LINK, "link", 1, tracks=(1, 2), gap=[50, 60], gate="normal")
    page = write(tmp_path, [ev], id_map={1: 7, 2: 9}).read_text()
    assert "output id(s): 7, 9" in page and "tracks [1, 2]" in page
    assert 'loading="lazy"' in page and re.search(r'<img [^>]*loading="lazy"', page)
    assert "output id(s): 9<" in write(tmp_path, [ev], id_map={"2": 9}).read_text()
    assert "output id(s): none" in write(tmp_path, [ev], id_map={}).read_text()
