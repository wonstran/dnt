"""Stage 3 must not undo stage 1: no link across a reversed heading, none between a split's halves."""

import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.events import Decision, EventKind
from dnt.refine.link import score_candidates
from dnt.refine.primitives import occlusion_flags
from dnt.refine.refiner import TrackRefiner

from ._fixtures import box_rows, table

FPS = 25.0


def _work(*rows):
    return io.to_work(table(*rows)).work


def _cands(work, cfg=None, fps=FPS):
    cfg = cfg or RefineConfig.defaults("vehicle")
    occ = occlusion_flags(work, None, cfg.encoder.occlusion_iou)
    cands, _ = score_candidates(
        work, cfg, fps, appearance=None, context=None, frame_size=None, occluded=occ
    )
    return {(c.i, c.j): c for c in cands}


def _same_raw(work, track, raw):
    """Make ``track`` a part of raw track ``raw``, as a split of ``raw`` leaves it."""
    work.loc[work["track"] == track, "raw_id"] = raw
    return work


def _pair(vx_first=3.0, vx_second=3.0, vy_second=0.0, gap=1):
    """Track 1 ends at frame 49; track 2 starts ``gap`` frames later where track 1 would be."""
    end = 100.0 + vx_first * 49
    first = box_rows(1, range(0, 50), 100.0, 100.0, vx=vx_first)
    second = box_rows(
        2, range(49 + gap, 100), end + vx_first * gap, 100.0, vx=vx_second, vy=vy_second
    )
    return first, second


# ---- the heading of the two ends on the normal gate --------------------------------------------


def test_a_straight_continuation_is_still_linked_and_records_its_heading():
    c = _cands(_work(*_pair()))[(1, 2)]
    assert c.gate == "normal" and c.signals["heading"] == pytest.approx(0.0, abs=1.0)


def test_a_reversal_across_the_gap_is_not_a_normal_link():
    # the box returns to where it was and goes back: +3 px/frame before, -3 after
    first, second = _pair(vx_second=-3.0)
    work = _work(first, second)
    assert (1, 2) not in _cands(work)  # a vehicle: the heading is checked by default
    person = RefineConfig.defaults("person")
    assert _cands(work, person)[(1, 2)].gate == "normal"  # a person: as before
    person.link.normal_heading_check = True
    assert (1, 2) not in _cands(work, person)


def test_the_check_can_be_turned_off_for_vehicles():
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.normal_heading_check = False
    first, second = _pair(vx_second=-3.0)
    assert (1, 2) in _cands(_work(first, second), cfg)


def test_a_right_angle_turn_is_still_linked():
    # 90 degrees is below link.max_heading_change (120)
    first, second = _pair(vx_second=0.0, vy_second=3.0)
    c = _cands(_work(first, second))[(1, 2)]
    assert c.signals["heading"] == pytest.approx(90.0, abs=5.0)


def test_a_slow_reversal_is_not_a_reversal():
    # 0.2 px/frame is 0.083 box heights per second at 25 fps: below link.heading_min_speed (0.2)
    first, second = _pair(vx_first=0.2, vx_second=-0.2)
    assert (1, 2) in _cands(_work(first, second))


def test_a_decelerating_end_still_shows_the_reversal():
    # a box clipped by the edge of the frame slows down just before it leaves: its last rows barely
    # move (0.3 px/frame, 0.125 box heights per second, below link.heading_min_speed), but over the
    # last two seconds it moved right at about 3 px/frame, and the next track comes back left
    first = box_rows(1, range(0, 40), 100.0, 100.0, vx=3.0)
    slow = box_rows(1, range(40, 50), 100.0 + 3.0 * 39, 100.0, vx=0.3)
    second = box_rows(2, range(51, 100), 100.0 + 3.0 * 39 + 0.3 * 10, 100.0, vx=-3.0)
    assert (1, 2) not in _cands(_work(first + slow, second))


def test_a_larger_allowed_heading_change_lets_the_reversal_through():
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.max_heading_change = 180.0
    first, second = _pair(vx_second=-3.0)
    assert (1, 2) in _cands(_work(first, second), cfg)


# ---- the two halves of a track that stage 1 split ----------------------------------------------


def test_the_halves_of_a_split_are_not_linked_again_when_asked():
    first, second = _pair()  # collinear: it would link
    work = _same_raw(_work(first, second), 2, 1)
    assert (1, 2) in _cands(work)  # off by default: stage 3 may heal a false split
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.keep_splits = True
    assert (1, 2) not in _cands(work, cfg)


def test_two_unrelated_tracks_are_still_linked():
    work = _work(*_pair())  # raw ids 1 and 2
    assert (1, 2) in _cands(work)


def test_a_split_around_an_intruder_still_lets_the_outer_halves_rejoin():
    # raw 1 held a vehicle, an intruder (frames 50-59) and the vehicle again: all three are
    # parts of raw 1; the intruder's halves stay apart, but 1 -> 3 skips it and is a real link
    a = box_rows(1, range(0, 50), 100.0, 100.0, vx=3.0)
    b = box_rows(2, range(50, 60), 700.0, 400.0, vx=0.0)
    c = box_rows(3, range(60, 110), 100.0 + 3.0 * 60, 100.0, vx=3.0)
    work = _same_raw(_same_raw(_work(a, b, c), 2, 1), 3, 1)
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.keep_splits = True
    cands = _cands(work, cfg)
    assert (1, 3) in cands
    assert (1, 2) not in cands and (2, 3) not in cands


def test_keep_splits_applies_to_the_person_target_too():
    first, second = _pair()
    work = _same_raw(_work(first, second), 2, 1)
    person = RefineConfig.defaults("person")
    assert (1, 2) in _cands(work, person)
    person.link.keep_splits = True
    assert (1, 2) not in _cands(work, person)


# ---- the settings ------------------------------------------------------------------------------


def test_link_guard_defaults_per_target():
    assert RefineConfig.defaults("vehicle").link.normal_heading_check is True
    assert RefineConfig.defaults("person").link.normal_heading_check is False
    # keep_splits is opt-in: on the afternoon clip it kept a true split (a box truck and a
    # van) apart but also two false splits (one van, one car), which stage 3 had been healing
    assert RefineConfig.defaults("vehicle").link.keep_splits is False
    assert RefineConfig.defaults("person").link.keep_splits is False
    assert RefineConfig.defaults().link.heading_seconds == 2.0
    # a saved config overlays onto its own target's default
    assert RefineConfig.from_dict({"target": "vehicle"}).link.normal_heading_check is True
    assert RefineConfig.from_dict({"target": "person"}).link.normal_heading_check is False
    cfg = RefineConfig.from_dict({"target": "vehicle", "link": {"keep_splits": True}})
    assert cfg.link.keep_splits is True and cfg.link.normal_heading_check is True


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_heading_seconds_must_be_positive(bad):
    cfg = RefineConfig.defaults()
    cfg.link.heading_seconds = bad
    with pytest.raises(ValueError, match=r"link\.heading_seconds"):
        cfg.validate()


# ---- end to end: an accepted split stays split -------------------------------------------------


def _there_and_back():
    """One track that goes right, then back left from where it stopped; frame 50 is missing."""
    first = box_rows(1, range(0, 50), 100.0, 100.0, vx=3.0, w=60.0, h=40.0, cls=2)
    second = box_rows(1, range(51, 100), 100.0 + 3.0 * 49, 100.0, vx=-3.0, w=60.0, h=40.0, cls=2)
    return table(first, second)


def _refine(tmp_path, cfg):
    src = tmp_path / "t.txt"
    _there_and_back().to_csv(src, index=False, header=False)
    refiner = TrackRefiner(cfg)
    refiner.refine(src, tmp_path / "o.txt", fps=10, verbose=False)
    return refiner.last_result


def _motion_only_config():
    cfg = RefineConfig.defaults("vehicle")
    cfg.switch.motion_only_cap = 0.95  # so that a motion-only split is accepted
    return cfg


def test_without_the_guards_link_undoes_the_accepted_split(tmp_path):
    cfg = _motion_only_config()
    cfg.link.keep_splits = False
    cfg.link.normal_heading_check = False
    res = _refine(tmp_path, cfg)
    (split,) = [e for e in res.events if e.stage == "switch"]
    assert split.decision is Decision.AUTO_ACCEPT and split.params["cut_frame"] == 51
    links = [e for e in res.events if e.kind is EventKind.LINK and e.applied]
    assert len(links) == 1 and res.tracks["track"].nunique() == 1  # split, then joined again


def test_with_the_guards_the_accepted_split_stays_split(tmp_path):
    res = _refine(tmp_path, _motion_only_config())
    (split,) = [e for e in res.events if e.stage == "switch"]
    assert split.decision is Decision.AUTO_ACCEPT and split.applied
    assert not [e for e in res.events if e.kind is EventKind.LINK and e.applied]
    assert res.tracks["track"].nunique() == 2
    parts = res.tracks.groupby("track")["frame"].agg(["min", "max"])
    assert sorted(map(tuple, parts.to_numpy().tolist())) == [(0, 49), (51, 99)]


def test_the_heading_check_alone_keeps_the_split(tmp_path):
    cfg = _motion_only_config()
    cfg.link.keep_splits = False
    res = _refine(tmp_path, cfg)
    assert res.tracks["track"].nunique() == 2


def test_keep_splits_alone_keeps_the_split(tmp_path):
    cfg = _motion_only_config()
    cfg.link.normal_heading_check = False
    cfg.link.keep_splits = True
    res = _refine(tmp_path, cfg)
    assert res.tracks["track"].nunique() == 2
