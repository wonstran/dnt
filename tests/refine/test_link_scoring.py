import numpy as np
import pytest

from dnt.refine import io
from dnt.refine.apply import merge_chains
from dnt.refine.config import RefineConfig
from dnt.refine.features import ArrayAppearance
from dnt.refine.link import legacy_link_events, link_tracklets, score_candidates
from dnt.refine.primitives import occlusion_flags

from ._fixtures import box_rows, load_raw, table

A_ = np.eye(8)[0]


def _work(*rows):
    return io.to_work(table(*rows)).work


def _cands(work, cfg=None, fps=25.0, app=None, frame_size=None):
    cfg = cfg or RefineConfig.defaults()
    occ = occlusion_flags(work, None, cfg.encoder.occlusion_iou)
    cands, _ = score_candidates(work, cfg, fps, appearance=app, context=None,
                                frame_size=frame_size, occluded=occ)
    return {(c.i, c.j): c for c in cands}


def _same_app(*spans):
    return ArrayAppearance({t: (list(fr), np.tile(A_, (len(fr), 1))) for t, fr in spans})


def _collinear(cls_a=0, cls_b=0):
    return _work(box_rows(1, range(40), 100.0, 100.0, vx=2.0, cls=cls_a),
                 box_rows(2, range(50, 90), 200.0, 100.0, vx=2.0, cls=cls_b))


def test_collinear_fragments_score_high():
    w = _collinear()
    with_app = _cands(w, app=_same_app((1, range(40)), (2, range(50, 90))))[(1, 2)]
    assert with_app.gate == "normal" and with_app.score == pytest.approx(1 - 0.15 * 11 / 25)
    motion = _cands(w)[(1, 2)]
    assert motion.score == pytest.approx(1 - 0.25 * 11 / 25) and motion.signals["motion_only"]


def test_high_cost_pair_merged_by_link_tracklets_is_not_auto_accepted():
    raw = table(box_rows(1, range(40), 100.0, 100.0), box_rows(2, range(45, 85), 125.0, 100.0))
    assert link_tracklets(raw.copy(), verbose=False)["track"].nunique() == 1
    c = _cands(io.to_work(raw).work)[(1, 2)]
    assert 0.40 <= c.score < 0.80


def test_static_gate_links_a_waiting_pedestrian():
    w = _work(box_rows(1, range(30), 100.0, 100.0), box_rows(2, range(90, 120), 100.0, 100.0))
    c = _cands(w, fps=10.0)[(1, 2)]
    assert c.gate == "static" and c.score == pytest.approx(1 - 0.25 * 61 / 100)  # g = 90 - 29
    cfg = RefineConfig.defaults()
    cfg.link.max_gap_static = 5.0
    assert (1, 2) not in _cands(w, cfg=cfg, fps=10.0)


def test_border_prior_scales_score():
    w = _collinear()
    inside = _cands(w, frame_size=(2000, 2000))[(1, 2)].score
    edge = _cands(w, frame_size=(215, 2000))[(1, 2)].score
    assert edge == pytest.approx(inside * 0.8)


def _turn(truck_frames=range(821, 883), j_start=(160.0, 240.0), j_v=(-5.0, 0.0),
          occluder=(100.0, 150.0, 300.0, 300.0)):
    rows = [box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0),
            box_rows(2, range(883, 921), *j_start, vx=j_v[0], vy=j_v[1], w=50.0, h=50.0)]
    if truck_frames is not None:
        x, y, w, h = occluder
        rows.append(box_rows(99, truck_frames, x, y, vx=0.5, w=w, h=h))
    app = _same_app((1, range(780, 821)), (2, range(883, 921)))
    return _cands(_work(*rows), fps=10.0, app=app)


def test_occluded_turn_is_witnessed_and_capped():
    c = _turn()[(1, 2)]
    assert c.gate == "occluded" and c.score == pytest.approx(0.75)
    assert c.signals["witness"] == pytest.approx(1.0) and c.signals["occluders"] == [99]
    assert c.signals["heading"] == pytest.approx(33.69, abs=0.1)


@pytest.mark.parametrize("kwargs", [
    {"truck_frames": None},
    {"truck_frames": range(821, 852)},
    {"j_start": (200.0, 400.0), "j_v": (0.0, 5.0)},
    {"j_start": (1200.0, 300.0), "occluder": (0.0, 0.0, 2000.0, 1000.0)},
])
def test_occluded_gate_rejects(kwargs):
    assert (1, 2) not in _turn(**kwargs)


def test_class_groups_decide_car_truck_links():
    w = _collinear(cls_a=2, cls_b=7)
    assert (1, 2) in _cands(w, cfg=RefineConfig.defaults("vehicle"))
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.class_groups = []
    assert (1, 2) not in _cands(w, cfg=cfg)


def _groups_by_raw(df, id_col):
    return {frozenset(g[id_col].tolist()) for _, g in df.groupby("track")}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_legacy_mode_matches_link_tracklets_grouping(seed):
    raw = load_raw(seed)
    cfg = RefineConfig.defaults()
    cfg.link.mode = "legacy"
    cfg.link.max_gap = 2.0  # 20 frames at 10 fps, the baseline's max_gap
    work = io.to_work(raw).work
    merged, _ = merge_chains(work, [tuple(e.tracks) for e in legacy_link_events(work, cfg, 10.0)])
    ours = {frozenset(set(g)) for g in merged.groupby("track")["raw_id"].apply(set)}
    out = link_tracklets(raw.copy(), max_gap=20, verbose=False)
    keyed = out.merge(raw[["frame", "x", "y", "w", "h", "track"]].rename(columns={"track": "orig"}),
                      on=["frame", "x", "y", "w", "h"])
    theirs = {frozenset(set(g)) for g in keyed.groupby("track")["orig"].apply(set)}
    assert ours == theirs


# ---- rule-pinning tests: each gate and cost term fails a specific test when removed ----


def _gap_pair(g, cls_a=0, cls_b=0, vx=2.0):
    """Collinear tracks 1 (frames 0..39) and 2 (starting g frames after 1 ends)."""
    t2 = 39 + g
    return _work(box_rows(1, range(40), 100.0, 100.0, vx=vx, cls=cls_a),
                 box_rows(2, range(t2, t2 + 40), 100.0 + vx * t2, 100.0, vx=vx, cls=cls_b))


def test_normal_gap_limit_is_max_gap():
    assert _cands(_gap_pair(25))[(1, 2)].gate == "normal"  # max_gap 1 s = 25 frames
    assert (1, 2) not in _cands(_gap_pair(26))  # moving, no static gate, no occluder
    cfg = RefineConfig.defaults()
    cfg.link.max_gap = 2.0
    assert _cands(_gap_pair(26), cfg=cfg)[(1, 2)].gate == "normal"
    assert (1, 2) not in _cands(_gap_pair(51), cfg=cfg)


def test_gap_cost_grows_with_the_gap():
    short, long = _cands(_gap_pair(5))[(1, 2)], _cands(_gap_pair(20))[(1, 2)]
    assert short.signals["c_gap"] == pytest.approx(5 / 25)
    assert long.signals["c_gap"] == pytest.approx(20 / 25)
    assert short.score == pytest.approx(1 - 0.25 * 5 / 25)
    assert long.score == pytest.approx(1 - 0.25 * 20 / 25) and long.score < short.score


def _wait(gap_end, start_x=100.0, start_y=100.0, vx=0.0, h=60.0):
    return _work(box_rows(1, range(30), 100.0, 100.0, h=h),
                 box_rows(2, range(gap_end, gap_end + 30), start_x, start_y, vx=vx, h=h))


def test_static_gate_limit_is_max_gap_static_in_frames():
    cfg = RefineConfig.defaults()
    cfg.link.max_gap_static = 6.1  # 61 frames at 10 fps; the gap below is exactly 61
    c = _cands(_wait(90), cfg=cfg, fps=10.0)[(1, 2)]
    assert c.gate == "static" and c.g == 61 and c.signals["c_gap"] == pytest.approx(1.0)
    assert c.score == pytest.approx(1 - 0.25)
    cfg.link.max_gap_static = 6.0
    assert (1, 2) not in _cands(_wait(90), cfg=cfg, fps=10.0)


def test_static_gate_needs_a_stopped_object():
    # The ended track is moving (2 px/frame = 0.33 h/s > static_speed 0.2), so a long gap with no
    # occluder is rejected, whereas the same gap after a stopped track is linked.
    moving = _work(box_rows(1, range(30), 100.0, 100.0, vx=2.0),
                   box_rows(2, range(90, 120), 170.0, 100.0))  # 12 px from where 1 stopped
    assert (1, 2) not in _cands(moving, fps=10.0)
    cfg = RefineConfig.defaults()
    cfg.link.static_speed = 1.0  # now a 0.33 h/s track counts as stopped
    assert _cands(moving, cfg=cfg, fps=10.0)[(1, 2)].gate == "static"


def test_static_gate_radius_is_half_a_height():
    near = _cands(_wait(90, start_x=125.0), fps=10.0)  # 25 px < 0.5 * 60
    assert near[(1, 2)].gate == "static"
    assert near[(1, 2)].signals["static_dist"] == pytest.approx(25.0)
    assert (1, 2) not in _cands(_wait(90, start_x=135.0), fps=10.0)  # 35 px > 30
    cfg = RefineConfig.defaults()
    cfg.link.static_radius = 0.7
    assert (1, 2) in _cands(_wait(90, start_x=135.0), cfg=cfg, fps=10.0)


def test_static_gate_rejects_a_size_jump():
    w = _work(box_rows(1, range(30), 100.0, 100.0), box_rows(2, range(90, 120), 100.0, 100.0,
                                                               w=80.0, h=120.0))
    assert (1, 2) not in _cands(w, fps=10.0)


def test_stopped_object_takes_the_normal_gate_within_max_gap():
    # max_gap is 1 s = 10 frames at 10 fps; only beyond it does the static gate apply.
    c = _cands(_wait(39), fps=10.0)[(1, 2)]
    assert c.gate == "normal" and c.g == 10
    c = _cands(_wait(40), fps=10.0)[(1, 2)]
    assert c.gate == "static" and c.g == 11


def _overlap_pair(shared, dx=0.0):
    """Track 1 ends at frame 39; track 2 starts ``shared - 1`` frames before that."""
    t2 = 40 - shared
    return _work(box_rows(1, range(40), 100.0, 100.0),
                 box_rows(2, range(t2, t2 + 40), 100.0 + dx, 100.0))


def test_overlap_gate_links_duplicate_fragments():
    c = _cands(_overlap_pair(2))[(1, 2)]
    assert c.gate == "overlap" and c.g == -1
    assert c.signals["overlap_iou"] == pytest.approx(1.0)
    assert c.score == pytest.approx(1.0)  # iou 1, no gap: c_mot 0, c_gap 0
    assert _cands(_overlap_pair(1))[(1, 2)].g == 0


def test_overlap_frames_bound_the_overlap():
    assert _cands(_overlap_pair(3))[(1, 2)].g == -2  # overlap_frames = 2
    assert (1, 2) not in _cands(_overlap_pair(4))
    cfg = RefineConfig.defaults()
    cfg.link.overlap_frames = 3
    assert _cands(_overlap_pair(4), cfg=cfg)[(1, 2)].g == -3
    cfg.link.overlap_frames = 0
    assert (1, 2) not in _cands(_overlap_pair(2), cfg=cfg)


def test_overlap_iou_gate_and_cost():
    # dnt.engine's IoU counts pixels inclusively: 25 * 61 / (2 * 31 * 61 - 25 * 61) = 25 / 37
    iou = 25.0 / 37.0
    c = _cands(_overlap_pair(2, dx=6.0))[(1, 2)]
    assert c.gate == "overlap" and c.signals["overlap_iou"] == pytest.approx(iou)
    assert c.signals["c_mot"] == pytest.approx(1.0 - iou)
    assert c.score == pytest.approx(1.0 - 0.45 / 0.6 * (1.0 - iou))  # motion-only weights
    assert (1, 2) not in _cands(_overlap_pair(2, dx=20.0))  # iou 0.2 < overlap_iou 0.5
    cfg = RefineConfig.defaults()
    cfg.link.overlap_iou = 0.1
    assert _cands(_overlap_pair(2, dx=20.0), cfg=cfg)[(1, 2)].gate == "overlap"


def test_overlap_gate_rejects_a_size_jump():
    w = _work(box_rows(1, range(40), 100.0, 100.0),
              box_rows(2, range(38, 78), 100.0, 100.0, w=90.0, h=180.0))
    assert (1, 2) not in _cands(w)


def test_class_gate_needs_equal_or_grouped_classes():
    assert (1, 2) in _cands(_collinear(cls_a=2, cls_b=2))
    assert (1, 2) not in _cands(_collinear(cls_a=0, cls_b=2))
    cfg = RefineConfig.defaults("vehicle")  # groups [[2, 7]]
    assert (1, 2) not in _cands(_collinear(cls_a=2, cls_b=5), cfg=cfg)
    assert (1, 2) in _cands(_collinear(cls_a=7, cls_b=2), cfg=cfg)  # order does not matter
    cfg.link.class_groups = [[2, 5], [5, 7]]
    assert (1, 2) not in _cands(_collinear(cls_a=2, cls_b=7), cfg=cfg)  # groups do not chain


def test_class_gate_uses_the_majority_class_not_the_last():
    rows1 = box_rows(1, range(39), 100.0, 100.0, vx=2.0, cls=2)
    rows1 += box_rows(1, [39], 100.0 + 2.0 * 39, 100.0, vx=2.0, cls=7)  # one late flip
    w = _work(rows1, box_rows(2, range(50, 90), 200.0, 100.0, vx=2.0, cls=2))
    assert (1, 2) in _cands(w)  # the legacy per-end class check would reject this
    w = _work(box_rows(1, range(40), 100.0, 100.0, vx=2.0, cls=2),
              box_rows(2, range(50, 89), 200.0, 100.0, vx=2.0, cls=7)
              + box_rows(2, [89], 200.0 + 2.0 * 39, 100.0, vx=2.0, cls=2))
    assert (1, 2) not in _cands(w)


def test_border_prior_needs_both_ends_inside():
    w = _collinear()
    base = _cands(w, frame_size=(2000, 2000))[(1, 2)]
    assert base.signals["b"] == 1
    assert _cands(w)[(1, 2)].signals["b"] == 1  # no frame size: no prior
    # rightward pair: the start box (x + w = 230) leaves the 30 px margin at width 259 while the
    # end box (x + w = 208) does not
    start_out = _cands(w, frame_size=(259, 2000))[(1, 2)]
    assert start_out.signals["b"] == 0 and start_out.score == pytest.approx(base.score * 0.8)
    left = _work(box_rows(1, range(40), 300.0, 100.0, vx=-2.0),
                 box_rows(2, range(50, 90), 200.0, 100.0, vx=-2.0))
    inside = _cands(left, frame_size=(2000, 2000))[(1, 2)]
    end_out = _cands(left, frame_size=(265, 2000))[(1, 2)]  # end 252 > 235 >= start 230
    assert end_out.signals["b"] == 0 and end_out.score == pytest.approx(inside.score * 0.8)
    top = _cands(left, frame_size=(2000, 2000))[(1, 2)]
    assert top.signals["b"] == 1
    assert _cands(w, frame_size=(2000, 130))[(1, 2)].signals["b"] == 0  # y + h = 160 > 100


def test_border_margin_scales_with_track_height():
    w = _collinear()
    cfg = RefineConfig.defaults()
    cfg.link.border_margin = 0.0  # both boxes are inside a 230 px wide frame
    assert _cands(w, cfg=cfg, frame_size=(230, 2000))[(1, 2)].signals["b"] == 1
    assert _cands(w, frame_size=(230, 2000))[(1, 2)].signals["b"] == 0  # margin 30 px


def _app(vec_j, missing=()):
    tab = {1: (list(range(40)), np.tile(A_, (40, 1))),
           2: (list(range(50, 90)), np.tile(vec_j, (40, 1)))}
    for m in missing:
        tab.pop(m)
    return ArrayAppearance(tab)


def test_appearance_term_lowers_the_score_of_dissimilar_pairs():
    w = _collinear()
    same = _cands(w, app=_app(A_))[(1, 2)]
    ortho = _cands(w, app=_app(np.eye(8)[1]))[(1, 2)]
    opposite = _cands(w, app=_app(-A_))[(1, 2)]
    assert same.signals["c_app"] == pytest.approx(0.0)
    assert ortho.signals["c_app"] == pytest.approx(0.5)
    assert opposite.signals["c_app"] == pytest.approx(1.0)
    gap = 0.15 * 11 / 25
    assert same.score == pytest.approx(1 - gap)
    assert ortho.score == pytest.approx(1 - 0.40 * 0.5 - gap)
    assert opposite.score == pytest.approx(1 - 0.40 - gap)
    assert not any(c.signals["motion_only"] for c in (same, ortho, opposite))


def test_missing_embeddings_cost_one_half():
    w = _collinear()
    for missing in [(1,), (2,), (1, 2)]:
        c = _cands(w, app=_app(A_, missing=missing))[(1, 2)]
        assert c.signals["c_app"] == pytest.approx(0.5) and not c.signals["motion_only"]
        assert c.score == pytest.approx(1 - 0.40 * 0.5 - 0.15 * 11 / 25)


def test_appearance_uses_the_k_embeddings_nearest_the_gap():
    e = np.eye(8)
    emb1 = np.vstack([np.tile(e[3], (30, 1)), np.tile(e[0], (10, 1))])  # the last 10 are e0
    emb2 = np.vstack([np.tile(e[0], (10, 1)), np.tile(e[5], (30, 1))])
    app = ArrayAppearance({1: (list(range(40)), emb1), 2: (list(range(50, 90)), emb2)})
    w = _collinear()
    assert _cands(w, app=app)[(1, 2)].signals["c_app"] == pytest.approx(0.0)  # k = 5 <= 10
    cfg = RefineConfig.defaults()
    cfg.link.k_embed = 40
    # mean1 = (30 e3 + 10 e0), mean2 = (10 e0 + 30 e5): cosine 100 / 1000 = 0.1
    assert _cands(w, cfg=cfg, app=app)[(1, 2)].signals["c_app"] == pytest.approx(0.45)


def test_motion_only_renormalizes_the_weights():
    c = _cands(_collinear())[(1, 2)]
    assert c.signals["motion_only"] and c.signals["c_app"] is None
    # mot 0.45 and gap 0.15 rescale to 0.75 and 0.25; c_mot is 0 for a collinear pair
    assert c.score == pytest.approx(1 - 0.25 * 11 / 25)
    off = _work(box_rows(1, range(40), 100.0, 100.0), box_rows(2, range(45, 85), 125.0, 100.0))
    m = _cands(off)[(1, 2)]
    w_mot, w_gap = 0.45 / 0.60, 0.15 / 0.60
    assert m.score == pytest.approx(1 - w_mot * m.signals["c_mot"] - w_gap * 6 / 25)
    assert m.signals["c_mot"] > 0.3


def _turn_expected(app_cost=0.0, g=63, limit=80):
    """Hand-built occluded cost for the _turn() geometry (see the comments there)."""
    chord = np.array([185.0 - 225.0, 265.0 - 325.0])  # end center (225, 325), start (185, 265)
    clen = float(np.hypot(*chord))
    v_need = clen / (50.0 * g / 10.0)
    heading = float(np.degrees(np.arccos(np.dot([0.0, -1.0], chord) / clen)))
    c_mot = 0.5 * v_need / (1.5 * 1.0) + 0.5 * heading / 120.0
    return c_mot, 0.25 * c_mot + 0.15 * g / limit + 0.60 * app_cost


def test_occluded_score_cap_and_uncapped_score():
    c_mot, cost = _turn_expected()
    assert c_mot == pytest.approx(0.2167, abs=1e-3)
    cfg = RefineConfig.defaults()
    cfg.link.occluded_score_cap = 1.0
    raw = _turn_cfg(cfg)[(1, 2)]
    assert raw.gate == "occluded" and raw.signals["c_mot"] == pytest.approx(c_mot)
    assert raw.score == pytest.approx(1.0 - cost) and raw.score > 0.75
    assert raw.signals["v_need"] == pytest.approx(72.111 / (50.0 * 6.3), abs=1e-3)
    assert raw.signals["v_ref"] == pytest.approx(1.0)
    assert _turn()[(1, 2)].score == pytest.approx(0.75)  # default cap
    cfg.link.occluded_score_cap = 0.5
    assert _turn_cfg(cfg)[(1, 2)].score == pytest.approx(0.5)


def _turn_cfg(cfg, **kw):
    """The _turn() scene scored under ``cfg``."""
    rows = [box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0),
            box_rows(2, range(883, 921), 160.0, 240.0, vx=-5.0, w=50.0, h=50.0),
            box_rows(99, kw.get("truck", range(821, 883)), 100.0, 150.0, vx=0.5,
                     w=300.0, h=300.0)]
    app = _same_app((1, range(780, 821)), (2, range(883, 921)))
    return _cands(_work(*rows), cfg=cfg, fps=10.0, app=app)


def test_occluded_score_uses_the_occluded_weights_with_appearance():
    rows = [box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0),
            box_rows(2, range(883, 921), 160.0, 240.0, vx=-5.0, w=50.0, h=50.0),
            box_rows(99, range(821, 883), 100.0, 150.0, vx=0.5, w=300.0, h=300.0)]
    app = ArrayAppearance({1: (list(range(780, 821)), np.tile(A_, (41, 1))),
                           2: (list(range(883, 921)), np.tile(np.eye(8)[1], (38, 1)))})
    cfg = RefineConfig.defaults()
    cfg.link.occluded_score_cap = 1.0
    c = _cands(_work(*rows), cfg=cfg, fps=10.0, app=app)[(1, 2)]
    assert c.signals["c_app"] == pytest.approx(0.5)
    assert c.score == pytest.approx(1.0 - _turn_expected(app_cost=0.5)[1])
    # motion-only occluded weights: mot 0.25 and gap 0.15 rescale to 0.625 and 0.375
    m = _cands(_work(*rows), cfg=cfg, fps=10.0)[(1, 2)]
    c_mot, _ = _turn_expected()
    assert m.score == pytest.approx(1.0 - 0.625 * c_mot - 0.375 * 63 / 80)


def test_witness_fraction_threshold_is_witness_min():
    # 62 hidden frames (821..882); 44 covered is 0.710 (passes 0.7), 43 is 0.694 (fails)
    ok = _turn(truck_frames=range(821, 865))[(1, 2)]
    assert ok.gate == "occluded" and ok.signals["witness"] == pytest.approx(44 / 62)
    assert (1, 2) not in _turn(truck_frames=range(821, 864))
    cfg = RefineConfig.defaults()
    cfg.link.witness_min = 0.5
    assert (1, 2) in _turn_cfg(cfg, truck=range(821, 852))  # 31 / 62 = 0.5
    cfg.link.witness_min = 0.9
    assert (1, 2) not in _turn_cfg(cfg, truck=range(821, 865))
    assert (1, 2) in _turn_cfg(cfg)


def _follow(offset, witness_iob=0.5):
    """An occluder that shares the hidden box's motion, shifted right by ``offset`` px.

    The hidden box runs from track 1's last box (200, 300) to track 2's first (160, 240) over
    63 frames, so with a 50 x 50 occluder the covered part is (50 - offset) / 50 of it.
    """
    vx, vy = -40.0 / 63.0, -60.0 / 63.0
    rows = [box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0),
            box_rows(2, range(883, 921), 160.0, 240.0, vx=-5.0, w=50.0, h=50.0),
            box_rows(99, range(821, 883), 200.0 + vx + offset, 300.0 + vy, vx=vx, vy=vy,
                     w=50.0, h=50.0)]
    cfg = RefineConfig.defaults()
    cfg.link.witness_iob = witness_iob
    app = _same_app((1, range(780, 821)), (2, range(883, 921)))
    return _cands(_work(*rows), cfg=cfg, fps=10.0, app=app)


def test_witness_needs_the_hidden_box_to_be_mostly_covered():
    c = _follow(20.0)[(1, 2)]  # 60% covered
    assert c.gate == "occluded" and c.signals["witness"] == pytest.approx(1.0)
    assert (1, 2) not in _follow(30.0)  # 40% covered, below witness_iob = 0.5
    assert (1, 2) in _follow(30.0, witness_iob=0.3)
    assert (1, 2) not in _follow(20.0, witness_iob=0.7)


def test_context_boxes_witness_a_gap():
    import pandas as pd

    w = _work(box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0),
              box_rows(2, range(883, 921), 160.0, 240.0, vx=-5.0, w=50.0, h=50.0))
    ctx = pd.DataFrame({"frame": list(range(821, 883)), "x": 100.0, "y": 150.0,
                        "w": 300.0, "h": 300.0})
    cfg = RefineConfig.defaults()
    occ = occlusion_flags(w, None, cfg.encoder.occlusion_iou)
    cands, _ = score_candidates(w, cfg, 10.0, appearance=None, context=ctx, frame_size=None,
                                occluded=occ)
    c = {(x.i, x.j): x for x in cands}[(1, 2)]
    assert c.gate == "occluded" and c.signals["occluders"] == [-1]


def test_heading_bound_is_max_heading_change():
    assert _turn()[(1, 2)].signals["heading"] == pytest.approx(33.69, abs=0.1)
    cfg = RefineConfig.defaults()
    cfg.link.max_heading_change = 30.0
    assert (1, 2) not in _turn_cfg(cfg)
    cfg.link.max_heading_change = 40.0
    assert _turn_cfg(cfg)[(1, 2)].gate == "occluded"
    # a turn that doubles back is rejected by default, accepted when the bound is opened up
    back = {"j_start": (200.0, 400.0), "j_v": (0.0, 5.0)}
    assert (1, 2) not in _turn(**back)
    wide = RefineConfig.defaults()
    wide.link.max_heading_change = 181.0
    rows = [box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0),
            box_rows(2, range(883, 921), 200.0, 400.0, vy=5.0, w=50.0, h=50.0),
            box_rows(99, range(821, 883), 100.0, 150.0, vx=0.5, w=300.0, h=400.0)]
    app = _same_app((1, range(780, 821)), (2, range(883, 921)))
    c = _cands(_work(*rows), cfg=wide, fps=10.0, app=app)[(1, 2)]
    assert c.signals["heading"] == pytest.approx(180.0, abs=0.1)


def test_heading_is_skipped_for_a_slow_ending_track():
    # 0.1 px/frame is 0.2 h/s at 10 fps for h = 50: below heading_min_speed, so no heading check
    rows = [box_rows(1, range(780, 821), 200.0, 500.0, vy=-0.4, w=50.0, h=50.0),
            box_rows(2, range(883, 921), 200.0, 400.0, vy=5.0, w=50.0, h=50.0),
            box_rows(99, range(821, 883), 100.0, 150.0, vx=0.5, w=300.0, h=400.0)]
    app = _same_app((1, range(780, 821)), (2, range(883, 921)))
    cfg = RefineConfig.defaults()
    cfg.link.static_speed = 0.0  # keep the static gate out of the way
    c = _cands(_work(*rows), cfg=cfg, fps=10.0, app=app)
    assert (1, 2) in c and c[(1, 2)].signals["heading"] is None


def test_speed_feasibility_is_speed_factor_times_reference_speed():
    far = {"j_start": (1200.0, 300.0), "occluder": (0.0, 0.0, 2000.0, 1000.0)}
    assert (1, 2) not in _turn(**far)
    rows = [box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0),
            box_rows(2, range(883, 921), 1200.0, 300.0, vx=-5.0, w=50.0, h=50.0),
            box_rows(99, range(821, 883), 0.0, 0.0, vx=0.5, w=2000.0, h=1000.0)]
    app = _same_app((1, range(780, 821)), (2, range(883, 921)))
    cfg = RefineConfig.defaults()
    cfg.link.speed_factor = 100.0
    c = _cands(_work(*rows), cfg=cfg, fps=10.0, app=app)[(1, 2)]
    assert c.gate == "occluded" and c.signals["v_need"] > 1.5 * c.signals["v_ref"]
    ratio = c.signals["v_need"] / c.signals["v_ref"]
    cfg.link.speed_factor = ratio * 1.01
    assert (1, 2) in _cands(_work(*rows), cfg=cfg, fps=10.0, app=app)
    cfg.link.speed_factor = ratio * 0.99
    assert (1, 2) not in _cands(_work(*rows), cfg=cfg, fps=10.0, app=app)


def test_occluded_gate_limit_is_max_gap_occluded():
    # g = 63 frames at 10 fps is 6.3 s
    cfg = RefineConfig.defaults()
    cfg.link.max_gap_occluded = 6.3
    assert (1, 2) in _turn_cfg(cfg)
    cfg.link.max_gap_occluded = 6.2
    assert (1, 2) not in _turn_cfg(cfg)


def test_occluded_gate_rejects_a_size_jump_between_clean_boxes():
    rows = [box_rows(1, range(780, 821), 200.0, 500.0, vy=-5.0, w=50.0, h=50.0),
            box_rows(2, range(883, 921), 160.0, 240.0, vx=-5.0, w=120.0, h=120.0),
            box_rows(99, range(821, 883), 100.0, 150.0, vx=0.5, w=300.0, h=300.0)]
    app = _same_app((1, range(780, 821)), (2, range(883, 921)))
    assert (1, 2) not in _cands(_work(*rows), fps=10.0, app=app)


def test_legacy_link_events_shape_and_empty_input():
    cfg = RefineConfig.defaults()
    cfg.link.mode = "legacy"
    w = _collinear()
    events = legacy_link_events(w, cfg, 25.0)
    assert len(events) == 1
    e = events[0]
    assert list(e.tracks) == [1, 2] and e.stage == "link" and e.algo_score == 1.0
    assert e.params["gate"] == "legacy" and e.params["gap"] == [39, 50]
    assert e.signals["legacy_cost"] >= 0.0
    assert legacy_link_events(w.iloc[0:0], cfg, 25.0) == []
    single = _work(box_rows(1, range(40), 100.0, 100.0))
    assert legacy_link_events(single, cfg, 25.0) == []


def test_legacy_mode_ignores_classes_groups_like_link_tracklets():
    cfg = RefineConfig.defaults("vehicle")
    cfg.link.mode = "legacy"
    assert legacy_link_events(_collinear(cls_a=2, cls_b=7), cfg, 25.0) == []


def test_vectorized_iob_matches_the_engine_iob():
    from dnt.refine.link import _iob_rows
    from dnt.refine.primitives import iob_matrix

    rng = np.random.default_rng(3)
    a = np.column_stack([rng.uniform(0, 100, 60), rng.uniform(0, 100, 60),
                         rng.uniform(0, 60, 60), rng.uniform(0, 60, 60)])
    b = np.column_stack([rng.uniform(0, 100, 60), rng.uniform(0, 100, 60),
                         rng.uniform(0, 60, 60), rng.uniform(0, 60, 60)])
    a[0, 2] = 0.0  # a zero-area hidden box
    want = np.array([iob_matrix(a[k : k + 1], b[k : k + 1])[0, 0] for k in range(60)])
    assert np.allclose(_iob_rows(a, b), want) and want.max() > 0.5
