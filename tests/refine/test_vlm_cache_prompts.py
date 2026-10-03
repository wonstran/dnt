import json

import pytest

from dnt.refine.events import Event, EventKind
from dnt.refine.vlm import VLMAnswer
from dnt.refine.vlm.cache import AnswerCache
from dnt.refine.vlm.prompts import (
    PERSON_SCREEN,
    RIDER_OPTIONS,
    SAME,
    VEHICLE_SCREEN,
    build_prompt,
    options_for,
)


def _key(**over):
    args = dict(
        image_jpeg=b"img", prompt="p", options=["a", "b"], backend="fake", model="m",
        temperature=0.0, vote_index=0,
    )
    args.update(over)
    return AnswerCache.key(**args)


def test_the_key_depends_on_every_part():
    base = _key()
    assert base == _key() and len(base) == 64
    for over in (
        {"image_jpeg": b"img2"}, {"prompt": "q"}, {"options": ["a", "c"]}, {"backend": "x"},
        {"model": "m2"}, {"temperature": 0.7}, {"vote_index": 1},
    ):
        assert _key(**over) != base, over
    assert _key(options=["a", "b"]) != _key(options=["b", "a"])  # order matters


def test_put_get_round_trip_and_misses(tmp_path):
    c = AnswerCache(tmp_path / "vlm")
    k = _key()
    assert c.get(k) is None
    ans = VLMAnswer("different", 0.8, "r", '{"raw": 1}')
    c.put(k, ans)
    assert c.get(k) == ans
    path = tmp_path / "vlm" / k[:2] / f"{k}.json"
    assert path.is_file() and json.loads(path.read_text())["answer"] == "different"
    path.write_text("not json")
    assert c.get(k) is None  # damaged file is a miss
    path.write_text(json.dumps({"answer": 3}))
    assert c.get(k) is None  # wrong shape is a miss
    assert list((tmp_path / "vlm").rglob("*.tmp")) == []


@pytest.mark.parametrize(
    "entry",
    [
        {"answer": "different", "confidence": float("nan"), "reason": "", "raw": ""},
        {"answer": "different", "confidence": float("inf"), "reason": "", "raw": ""},
        {"answer": "different", "confidence": 1.5, "reason": "", "raw": ""},
        {"answer": "different", "confidence": -0.1, "reason": "", "raw": ""},
        {"answer": "different", "confidence": True, "reason": "", "raw": ""},
        {"answer": "different", "confidence": "0.9", "reason": "", "raw": ""},
        {"answer": "different", "confidence": None, "reason": "", "raw": ""},
        {"answer": "different", "reason": "", "raw": ""},
        {"answer": "maybe", "confidence": 0.9, "reason": "", "raw": ""},
        {"answer": 3, "confidence": 0.9, "reason": "", "raw": ""},
    ],
)
def test_readable_but_invalid_entries_are_misses(tmp_path, entry):
    c = AnswerCache(tmp_path)
    k = _key()
    path = tmp_path / k[:2] / f"{k}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(entry))  # json.dumps writes NaN / Infinity literals
    assert c.get(k, ["same_individual", "different", "unsure"]) is None


def test_the_option_check_is_skipped_without_options_and_a_good_entry_is_a_hit(tmp_path):
    c = AnswerCache(tmp_path)
    k = _key()
    c.put(k, VLMAnswer("maybe", 0.5, "r", "raw"))
    assert c.get(k, ["different"]) is None  # not one of the question's options
    assert c.get(k) == VLMAnswer("maybe", 0.5, "r", "raw")  # no options to check against
    c.put(k, VLMAnswer("different", 0.0, "r", "raw"))  # 0.0 and 1.0 are valid
    assert c.get(k, ["different"]).confidence == 0.0


def test_a_failing_write_is_ignored(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    c = AnswerCache(blocker / "sub")  # a directory cannot be created under a file
    c.put(_key(), VLMAnswer("a", 1.0, "", ""))  # must not raise
    assert c.get(_key()) is None


def test_the_directory_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    c = AnswerCache("~/cache")
    c.put(_key(), VLMAnswer("a", 1.0, "", ""))
    assert (tmp_path / "cache").is_dir()


def _ev(kind, stage, **params):
    return Event.propose(
        stage=stage, kind=kind, tracks=[1], lineage=[[[1, 0, 9]]], frames=(0, 9),
        params=params, algo_score=0.5, signals={},
    )


def test_options_by_kind_and_target():
    assert options_for(_ev(EventKind.SPLIT, "switch", cut_frame=5), "person") == SAME
    assert options_for(_ev(EventKind.LINK, "link", gap=[5, 8]), "vehicle") == SAME
    drop = _ev(EventKind.DROP, "screen", reason="static", spans=None)
    assert options_for(drop, "person") == PERSON_SCREEN
    assert options_for(drop, "vehicle") == VEHICLE_SCREEN
    assert options_for(_ev(EventKind.RECLASS, "screen", new_cls=None, spans=None), "person") == PERSON_SCREEN
    assert options_for(_ev(EventKind.DROP, "orphan", reason="orphan", spans=None), "person") is None
    assert options_for(_ev(EventKind.FILL, "fill", gap=[1, 4]), "person") is None
    assert "unsure" in PERSON_SCREEN and "unsure" in VEHICLE_SCREEN and "unsure" in SAME
    assert set(RIDER_OPTIONS) <= set(PERSON_SCREEN)


def test_a_partial_screen_prompt_explains_the_two_rows():
    partial = _ev(EventKind.RECLASS, "screen", new_cls=None, spans=[[10, 40]])
    whole = _ev(EventKind.RECLASS, "screen", new_cls=None, spans=None)
    p = build_prompt(partial, "person", options_for(partial, "person"), fps=10.0)
    assert "Row A shows crops from that part" in p and "Row B shows crops from the rest" in p
    assert "row A" in p
    w = build_prompt(whole, "person", options_for(whole, "person"), fps=10.0)
    assert "Row B" not in w and "Row A" not in w


@pytest.mark.parametrize(
    "ev,target,needle",
    [
        (_ev(EventKind.SPLIT, "switch", cut_frame=5), "person", "before"),
        (_ev(EventKind.LINK, "link", gap=[100, 130]), "person", "3.0 s"),
        (_ev(EventKind.DROP, "screen", reason="static", spans=None), "person", "pedestrian"),
        (_ev(EventKind.DROP, "screen", reason="duplicate", spans=None, of=2), "vehicle", "vehicle"),
    ],
)
def test_prompts_name_the_options_and_the_reply_format(ev, target, needle):
    options = options_for(ev, target)
    p = build_prompt(ev, target, options, fps=10.0)
    assert needle in p
    for o in options:
        assert o in p
    assert '"answer"' in p and '"confidence"' in p and '"reason"' in p
    assert p == build_prompt(ev, target, options, fps=10.0)  # deterministic
