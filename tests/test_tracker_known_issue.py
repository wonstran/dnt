"""Known BoxMOT 16.0.11 defect: friendlier warning and error note, unchanged failure identity."""

import traceback
import warnings
from pathlib import Path

import pytest

from dnt.track import _boxmot_compat as bx
from dnt.track.tracker import ByteTrackConfig, OCSORTConfig, Tracker


@pytest.mark.parametrize(
    "cfg",
    [OCSORTConfig(), ByteTrackConfig(extra_kwargs={"tracker_type": "ocsort"})],
    ids=["ocsort", "bytetrack_to_ocsort"],
)
def test_affected_tracker_warns_on_build(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.warns(UserWarning, match="Known issue"):
            Tracker._build_boxmot_tracker(cfg.model, cfg, device="cpu", half=False)


def test_unaffected_tracker_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Tracker._build_boxmot_tracker(None, ByteTrackConfig(), device="cpu", half=False)


def test_affected_set_matches_known_issue():
    assert frozenset({"ocsort", "deepocsort", "hybridsort"}) == bx.UNFREEZE_DEFECT_TRACKERS


def test_crash_keeps_identity_and_gains_note(synthetic_video, stub_dets):
    video, _ = synthetic_video
    with pytest.warns(UserWarning, match="Known issue"), pytest.raises(TypeError) as info:
        Tracker(config=OCSORTConfig(), device="cpu").track(str(stub_dets), "", str(video))
    exc = info.value
    # identity unchanged: reviewed golden defect entries (type, full message, raise site) still match
    assert str(exc) == "only 0-dimensional arrays can be converted to Python scalars"
    innermost = traceback.extract_tb(exc.__traceback__)[-1]
    assert f"{Path(innermost.filename).name}:{innermost.name}" == "xysr_kf.py:unfreeze"
    notes = getattr(exc, "__notes__", [])
    assert any("Known issue" in n and "ByteTrack" in n for n in notes), notes


def test_unrelated_type_error_gets_no_note(monkeypatch, synthetic_video, stub_dets):
    class Boom:
        def update(self, dets, frame):
            raise TypeError("inside update")

    monkeypatch.setattr(Tracker, "_build_boxmot_tracker", staticmethod(lambda *a, **k: Boom()))
    video, _ = synthetic_video
    with pytest.raises(TypeError, match="inside update") as info:
        Tracker(config=ByteTrackConfig(), device="cpu").track(str(stub_dets), "", str(video))
    assert not getattr(info.value, "__notes__", [])
