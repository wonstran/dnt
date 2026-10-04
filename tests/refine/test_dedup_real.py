import copy
import shutil
from pathlib import Path

import pytest

from dnt.refine import io
from dnt.refine.config import RefineConfig
from dnt.refine.dedup import apply_merges, propose_merges
from dnt.refine.events import EventKind, Ledger
from dnt.refine.refiner import TrackRefiner
from dnt.refine.verify import Band, route_without_vlm

from ._dedup_checks import check_retained

DATA = Path("/mnt/e/videos/miami/dets")
CLIPS = Path("/mnt/e/videos/miami/clips")
pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="the Miami clips are not mounted")
EVENING, MIDDLE = "evening_1900_190000.00", "middle_120000.00"
GROUP = (122, 125, 134)  # one person, confirmed on video (spec 1)


def _files(clip):
    track_file = next(DATA.glob(f"*{clip}_ped_track.txt"))
    ledger = next(DATA.glob(f"*{clip}_ped_track_refined_p3.ledger.jsonl"))
    return track_file, ledger, Ledger.read(ledger).header


def _out_id(header, track):
    """The output ID of an input track: through the absorbed map, then the final renumbering."""
    survivor = header["absorbed"].get(str(track), track)
    return header["id_map"][str(survivor)]


# ---- the stage alone: a focused calibration regression

def _stage_only(clip):
    track_file, _, header = _files(clip)
    work = io.read_tracks(track_file, fmt="dnt", class_id=0).work
    cfg = RefineConfig.defaults()
    events = propose_merges(work, cfg, header["fps"])
    route_without_vlm(events, Band.of(cfg.dedup))
    return work, events, apply_merges(work, events, cfg)


def test_stage_only_the_evening_group_becomes_one_track():
    work, events, out = _stage_only(EVENING)
    assert {out.absorbed.get(t, t) for t in GROUP} == {122}
    merged = out.work[out.work["raw_id"].isin(GROUP)]
    assert merged["track"].nunique() == 1 and not merged.duplicated("frame").any()
    ours = [e for e in events if set(e.tracks) <= set(GROUP) and e.applied]
    assert sum(e.signals["dropped_rows"] for e in ours) <= 13
    assert out.work["track"].nunique() < work["track"].nunique()


def test_stage_only_side_by_side_pairs_in_the_middle_clip_stay_apart():
    _, _, out = _stage_only(MIDDLE)
    track_of = out.work.groupby("raw_id")["track"].first()
    assert track_of[266] != track_of[270] and track_of[15] != track_of[19]


# ---- the whole pipeline

def _check_evening(refiner, tracks, raw_rows):
    header = Ledger.read(refiner.last_result.ledger_path).header
    ids = {_out_id(header, t) for t in GROUP}
    assert len(ids) == 1, f"the three raw tracks ended as output IDs {sorted(ids)}"
    (tid,) = ids
    rows = tracks[tracks["track"] == tid]
    assert not rows.duplicated("frame").any()  # observed and filled rows never share a frame
    events = refiner.last_result.events
    merges = [e for e in events if e.kind is EventKind.MERGE and e.applied
              and set(e.tracks) <= set(GROUP)]
    assert merges and sum(e.signals["dropped_rows"] for e in merges) <= 13
    dropped = {(r, f) for e in merges for r, a, b in e.signals["dropped"] for f in range(a, b + 1)}
    # if an outside track had joined the group, its drops would be missing here: investigate
    check_retained(rows, raw_rows, dropped, GROUP)


def _raw_rows(track_file):
    work = io.read_tracks(track_file, fmt="dnt", class_id=0).work
    return set(zip(work["raw_id"], work["frame"], strict=True))


def test_pipeline_motion_only_merges_the_evening_group(tmp_path):
    track_file, _, header = _files(EVENING)
    cfg = RefineConfig.defaults()
    cfg.encoder.kind = "none"
    refiner = TrackRefiner(cfg)
    tracks = refiner.refine(track_file, tmp_path / "x.txt", fps=header["fps"], verbose=False)
    _check_evening(refiner, tracks, _raw_rows(track_file))


def test_pipeline_motion_only_keeps_the_middle_negatives_apart(tmp_path):
    track_file, _, header = _files(MIDDLE)
    cfg = RefineConfig.defaults()
    cfg.encoder.kind = "none"
    refiner = TrackRefiner(cfg)
    refiner.refine(track_file, tmp_path / "x.txt", fps=header["fps"], verbose=False)
    h = Ledger.read(refiner.last_result.ledger_path).header
    assert _out_id(h, 266) != _out_id(h, 270) and _out_id(h, 15) != _out_id(h, 19)


@pytest.mark.skipif(not CLIPS.is_dir(), reason="the Miami videos are not mounted")
def test_pipeline_with_the_recorded_p3_setup_and_the_video(tmp_path):
    track_file, _, header = _files(EVENING)
    kind = header["config"]["encoder"]["kind"]
    if kind == "dino":
        pytest.importorskip("transformers")
    elif kind == "reid":
        pytest.importorskip("torchreid")
    video = next(CLIPS.glob(f"*{EVENING}.mp4"))
    recorded = copy.deepcopy(header["config"])
    recorded["vlm"]["backend"] = "none"  # no network: appearance is real, the VLM is not used
    recorded["vlm"]["use"] = None
    cfg = RefineConfig.from_dict(recorded)  # the P3 run predates dedup: it gets the defaults
    cache = next(DATA.glob(f"*{EVENING}_ped_track_refined_p3.features.npz"), None)
    if cache is not None:
        shutil.copy(cache, tmp_path / "x.features.npz")  # reuse the embeddings if the key matches
    refiner = TrackRefiner(cfg)
    tracks = refiner.refine(track_file, tmp_path / "x.txt", video_file=str(video), verbose=False)
    _check_evening(refiner, tracks, _raw_rows(track_file))
