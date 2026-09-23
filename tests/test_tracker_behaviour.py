import numpy as np
import pandas as pd
import pytest
from boxmot.trackers.ocsort.ocsort import OcSort
from synthetic import StubDetector

from dnt.track import _boxmot_compat as bx
from dnt.track.tracker import (
    BoostTrackConfig,
    BoTSORTConfig,
    ByteTrackConfig,
    DeepOCSORTConfig,
    HybridSORTConfig,
    OCSORTConfig,
    SFSORTConfig,
    StrongSORTConfig,
    Tracker,
)

# Fields whose effect the synthetic scene cannot show; claimed only by source evidence (spec §2.6 group 4).
# A required probe may move here ONLY with a written reason reviewed by the maintainer.
EVIDENCE_ONLY = {
    "bytetrack": {"frame_rate"},
    "botsort": {"proximity_thresh", "appearance_thresh", "with_reid", "cmc_method", "frame_rate"},
    "ocsort": {"asso_func", "delta_t", "inertia"},
    "deepocsort": {"asso_func", "delta_t", "inertia"},
    "strongsort": {"max_cos_dist", "max_iou_dist", "nn_budget", "ema_alpha", "mc_lambda"},
    "hybridsort": {"asso_func"},
    "boosttrack": set(),
    # reason: 2nd-tier matching only runs if the frame has >=1 high-score det (it
    # seeds unmatched_tracks); the 'low' fixture makes each frame uniformly high- or
    # low-confidence, so this branch is never entered (default=199 rows, probe=199
    # rows either way, for both fields below). Verified with boxmot's SFSORT
    # directly, independent of dnt: a same-frame low-conf-only detection never
    # reaches the low_th/match_th_second-gated branch in sfsort.py's update().
    "sfsort": {"low_th", "match_th_second"},
}

C = {"bytetrack": ByteTrackConfig, "botsort": BoTSORTConfig, "ocsort": OCSORTConfig, "deepocsort": DeepOCSORTConfig,
     "strongsort": StrongSORTConfig, "hybridsort": HybridSORTConfig, "boosttrack": BoostTrackConfig,
     "sfsort": SFSORTConfig}

# Trackers whose shared Kalman filter `unfreeze()` crashes under numpy>=2 on any
# missed-then-redetected track (upstream BoxMOT 16.0.11 defect, not a dnt wiring
# gap -- see test_boxmot_16011_unfreeze_numpy2_defect). Their PROBES rows stay in
# the parametrize list (so coverage/ids are unaffected) but are marked xfail.
UPSTREAM_BROKEN = {"ocsort", "deepocsort", "hybridsort"}
_UNFREEZE_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason=(
        "BoxMOT 16.0.11 unfreeze() crashes under numpy>=2 when a lost track is "
        "re-detected (upstream bug; see test_boxmot_16011_unfreeze_numpy2_defect)"
    ),
)


def _as_param(entry: tuple) -> object:
    tracker = entry[0]
    return pytest.param(*entry, marks=_UNFREEZE_XFAIL) if tracker in UPSTREAM_BROKEN else entry


# (tracker, field, probe value, detections variant, expected observable)
PROBES = [
    # detection threshold: above stub conf 0.9 -> no tracks
    ("bytetrack", "track_thresh", 0.95, "plain", "no_tracks"),
    ("botsort", "track_high_thresh", 0.95, "plain", "no_tracks"),
    ("botsort", "new_track_thresh", 0.95, "plain", "no_tracks"),
    ("ocsort", "det_thresh", 0.95, "plain", "no_tracks"),
    ("deepocsort", "det_thresh", 0.95, "plain", "no_tracks"),
    ("hybridsort", "det_thresh", 0.95, "plain", "no_tracks"),
    ("boosttrack", "det_thresh", 0.95, "plain", "no_tracks"),
    ("sfsort", "high_th", 0.95, "plain", "no_tracks"),
    ("sfsort", "new_track_th", 0.95, "plain", "no_tracks"),
    # low-score band: odd frames at conf 0.3 -> excluding them loses rows
    ("bytetrack", "min_conf", 0.5, "low", "fewer_rows"),
    ("botsort", "track_low_thresh", 0.5, "low", "fewer_rows"),
    # lifetime below the 8-frame gap -> new ID after the gap
    ("bytetrack", "track_buffer", 2, "plain", "more_ids"),
    ("botsort", "track_buffer", 2, "plain", "more_ids"),
    ("ocsort", "max_age", 2, "plain", "more_ids"),
    ("deepocsort", "max_age", 2, "plain", "more_ids"),
    ("hybridsort", "max_age", 2, "plain", "more_ids"),
    ("boosttrack", "max_age", 2, "plain", "more_ids"),
    ("strongsort", "max_age", 2, "plain", "more_ids"),
    # confirmation: object 3 enters at frame 30 -> its first rows are suppressed
    ("ocsort", "min_hits", 20, "plain", "fewer_rows"),
    ("deepocsort", "min_hits", 20, "plain", "fewer_rows"),
    ("hybridsort", "min_hits", 20, "plain", "fewer_rows"),
    ("boosttrack", "min_hits", 20, "plain", "fewer_rows"),
    ("strongsort", "n_init", 20, "plain", "fewer_rows"),
    # match threshold: near-impossible matching -> ID count increases
    ("bytetrack", "match_thresh", 0.01, "plain", "more_ids"),
    ("botsort", "match_thresh", 0.01, "plain", "more_ids"),
    ("sfsort", "match_th_first", 0.01, "plain", "more_ids"),
    ("ocsort", "iou_threshold", 0.99, "plain", "more_ids"),
    ("deepocsort", "iou_threshold", 0.99, "plain", "more_ids"),
    ("hybridsort", "iou_threshold", 0.99, "plain", "more_ids"),
    ("boosttrack", "iou_threshold", 0.99, "plain", "more_ids"),
]

_cache: dict[tuple, pd.DataFrame] = {}


@pytest.fixture(scope="module")
def dets_files(synthetic_video, stub_dets, tmp_path_factory):
    video, truth = synthetic_video
    low = tmp_path_factory.mktemp("low") / "low_iou.txt"
    StubDetector(truth, low_conf=0.3).detect(video, iou_file=low)
    return video, {"plain": stub_dets, "low": low}


def _run(tracker, variant, dets_files, **kwargs):
    key = (tracker, variant, tuple(sorted(kwargs.items())))
    if key not in _cache:
        video, files = dets_files
        cfg = C[tracker](**kwargs)
        _cache[key] = Tracker(config=cfg, device="cpu").track(str(files[variant]), "", str(video))
    return _cache[key]


def test_every_effective_field_is_probed_or_evidence_only():
    probed = {}
    for tracker, field, *_ in PROBES:
        probed.setdefault(tracker, set()).add(field)
    for tracker, fields in bx.EFFECTIVE_PARAMS.items():
        assert fields == probed.get(tracker, set()) | EVIDENCE_ONLY[tracker], tracker


@pytest.mark.parametrize(("tracker", "field", "value", "variant", "kind"),
                         [_as_param(p) for p in PROBES],
                         ids=[f"{t}-{f}" for t, f, *_ in PROBES])
def test_probe_changes_behaviour(tracker, field, value, variant, kind, dets_files):
    default = _run(tracker, variant, dets_files)
    probe = _run(tracker, variant, dets_files, **{field: value})
    assert len(default) > 0
    if kind == "no_tracks":
        assert len(probe) == 0
    elif kind == "more_ids":
        assert probe["track"].nunique() > default["track"].nunique()
    else:
        assert len(probe) < len(default)


def test_boxmot_16011_unfreeze_numpy2_defect():
    """Documents an upstream BoxMOT 16.0.11 defect, independent of dnt.

    OcSort's Kalman filter `unfreeze()` (boxmot/motion/kalman_filters/aabb/xysr_kf.py)
    unpacks a stored history box into 4 names via `x1, y1, s1, r1 = box1`; when the
    stored box is a (4, 1)-shaped array, each name ends up shape-(1,) instead of a
    scalar. Under numpy>=2, `float()` of a non-0-dimensional array raises TypeError
    (numpy<2 silently coerced it). This crashes ocsort, deepocsort and hybridsort
    (which share this Kalman filter code) whenever a track is missed for one frame
    and then re-detected -- see the 12 PROBES rows marked xfail above.

    If BoxMOT or numpy change this behaviour, this test will start failing; when it
    does, remove the xfail marks on the ocsort/deepocsort/hybridsort PROBES rows
    (drop UPSTREAM_BROKEN/_as_param) and delete this test.
    """
    tracker = OcSort(det_thresh=0.1, max_age=30, min_hits=1, iou_threshold=0.1, per_class=False)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    box = np.array([[10, 10, 30, 30, 0.9, 0]], dtype=float)
    empty = np.empty((0, 6), dtype=float)
    with pytest.raises(TypeError, match="0-dimensional"):
        for i in range(5):
            tracker.update(empty if i == 2 else box, img)


def test_update_type_error_propagates_once(monkeypatch, synthetic_video, stub_dets):
    calls = []

    class Boom:
        def update(self, dets, frame):
            calls.append(1)
            raise TypeError("inside update")

    monkeypatch.setattr(Tracker, "_build_boxmot_tracker", staticmethod(lambda *a, **k: Boom()))
    video, _ = synthetic_video
    with pytest.raises(TypeError, match="inside update"):
        Tracker(config=ByteTrackConfig(), device="cpu").track(str(stub_dets), "", str(video))
    assert calls == [1]


def test_track_empty_det_file(tmp_path, synthetic_video):
    video, _ = synthetic_video
    empty = tmp_path / "empty_iou.txt"
    empty.write_text("")
    out = tmp_path / "tracks.txt"
    df = Tracker(config=ByteTrackConfig(), device="cpu").track(str(empty), str(out), str(video))
    assert list(df.columns) == Tracker.TRACK_FIELDS and df.empty
    assert out.exists() and out.read_text() == ""
