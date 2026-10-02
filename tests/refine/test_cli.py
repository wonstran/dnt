import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from dnt.refine import cli
from dnt.refine.config import RefineConfig

from ._fixtures import box_rows, table

ROOT = Path(__file__).resolve().parents[2]


def _src(tmp_path):
    p = tmp_path / "t.txt"
    table(box_rows(1, range(40), 100.0, 100.0, vx=2.0),
          box_rows(2, range(46, 90), 192.0, 100.0, vx=2.0)).to_csv(p, index=False, header=False)
    return p


def _run(*args):
    return subprocess.run([sys.executable, "-m", "dnt.refine.cli", *map(str, args)],
                          capture_output=True, text=True)


def _cfg(tmp_path, name="c.yaml", **encoder):
    cfg = tmp_path / name
    c = RefineConfig.defaults()
    for k, v in encoder.items():
        setattr(c.encoder, k, v)
    c.to_yaml(cfg)
    return cfg


def _applied(stdout, prefix):
    """Count the summary's events whose key starts with ``prefix`` and ends in AUTO_ACCEPT."""
    events = json.loads(stdout)["summary"]["events"]
    return sum(n for k, n in events.items() if k.startswith(prefix + "/") and "AUTO" in k)


def _main(capsys, *args):
    code = cli.main([str(a) for a in args])
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_cli_run_writes_outputs(tmp_path):
    cfg = tmp_path / "c.yaml"
    RefineConfig.defaults().to_yaml(cfg)
    r = _run("run", _src(tmp_path), "--fps", 10, "--config", cfg, "--out", tmp_path / "o.csv")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o.csv").exists() and (tmp_path / "o.ledger.jsonl").exists()
    assert json.loads(r.stdout)["ledger"].endswith("o.ledger.jsonl")


def test_cli_reports_errors_with_exit_code_2(tmp_path):
    cfg = tmp_path / "c.yaml"
    RefineConfig.defaults().to_yaml(cfg)
    r = _run("run", _src(tmp_path), "--config", cfg, "--out", tmp_path / "o.csv")
    assert r.returncode == 2 and "fps=" in r.stderr


def test_entry_point_is_declared():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["dnt-refine"] == "dnt.refine.cli:main"


def test_public_api():
    import dnt.refine as refine

    assert set(refine.__all__) == {"RefineConfig", "RefineResult", "TrackRefiner",
                                 "interpolate_tracks_rts", "link_tracklets"}


# ---- exit codes, JSON output, and the options ------------------------------------------------


def test_success_prints_json_with_out_ledger_and_summary(tmp_path, capsys):
    out = tmp_path / "o.csv"
    code, stdout, _ = _main(capsys, "run", _src(tmp_path), "--fps", 10,
                            "--config", _cfg(tmp_path), "--out", out)
    assert code == 0
    doc = json.loads(stdout)
    assert set(doc) == {"out", "ledger", "summary"}
    assert doc["out"] == str(out) and doc["ledger"] == str(tmp_path / "o.ledger.jsonl")
    assert doc["summary"]["after"]["tracks"] == 1  # the two fragments are linked
    assert len(pd.read_csv(out, header=None)) == 90 - 46 + 40 + 6  # 6 filled rows


def test_missing_fps_without_video_exits_2(tmp_path, capsys):
    out = tmp_path / "o.csv"
    code, stdout, err = _main(capsys, "run", _src(tmp_path), "--config", _cfg(tmp_path),
                              "--out", out)
    assert code == 2 and stdout == "" and "fps=" in err and "dnt-refine: error:" in err
    assert not out.exists()


def test_missing_track_file_exits_2(tmp_path, capsys):
    code, stdout, err = _main(capsys, "run", tmp_path / "nope.txt", "--fps", 10,
                              "--config", _cfg(tmp_path), "--out", tmp_path / "o.csv")
    assert code == 2 and stdout == "" and "dnt-refine: error:" in err


def test_bad_config_path_exits_2(tmp_path, capsys):
    code, stdout, err = _main(capsys, "run", _src(tmp_path), "--fps", 10,
                              "--config", tmp_path / "nope.yaml", "--out", tmp_path / "o.csv")
    assert code == 2 and stdout == "" and "dnt-refine: error:" in err


def test_unknown_config_key_exits_2(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    cfg.write_text(cfg.read_text() + "bogus_key: 1\n")
    code, stdout, err = _main(capsys, "run", _src(tmp_path), "--fps", 10, "--config", cfg,
                              "--out", tmp_path / "o.csv")
    assert code == 2 and stdout == "" and "bogus_key" in err
    assert not (tmp_path / "o.csv").exists()


def test_format_mot_reads_a_mot_file(tmp_path, capsys):
    cfg = RefineConfig.defaults()
    cfg.class_ids = [2]
    cfg_file = tmp_path / "c.yaml"
    cfg.to_yaml(cfg_file)
    mot = tmp_path / "mot.txt"
    dnt = _src(tmp_path)
    pd.read_csv(dnt, header=None).iloc[:, :7].to_csv(mot, index=False, header=False)
    code, stdout, err = _main(capsys, "run", mot, "--format", "mot", "--fps", 10,
                              "--config", cfg_file, "--out", tmp_path / "o.csv")
    assert code == 0, err
    assert json.loads(stdout)["summary"]["after"]["tracks"] == 1
    assert set(pd.read_csv(tmp_path / "o.csv", header=None)[7]) == {2}
    header = json.loads((tmp_path / "o.ledger.jsonl").read_text().splitlines()[0])
    assert header["inputs"]["tracks"]["format"] == "mot"
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["run", str(mot), "--format", "csv", "--config", "c",
                                       "--out", "o"])
    assert exc.value.code == 2


def _passenger_scene(tmp_path):
    src = tmp_path / "p.txt"
    table(box_rows(5, range(100, 150), 100.0, 810.0, vx=3.0, w=20.0, h=40.0)).to_csv(
        src, index=False, header=False)
    ctx = tmp_path / "ctx.txt"
    table(box_rows(9, range(100, 150), 80.0, 800.0, vx=3.0, w=80.0, h=60.0, cls=2)).to_csv(
        ctx, index=False, header=False)
    return src, ctx


def test_context_option_is_passed_through(tmp_path, capsys):
    src, ctx = _passenger_scene(tmp_path)
    cfg = _cfg(tmp_path)
    code, stdout, err = _main(capsys, "run", src, "--fps", 10, "--config", cfg,
                              "--out", tmp_path / "a.csv", "--context", ctx)
    assert code == 0, err
    assert _applied(stdout, "screen/DROP") == 1
    assert (tmp_path / "a.csv").read_text().strip() == ""  # the passenger is dropped
    code, stdout, err = _main(capsys, "run", src, "--fps", 10, "--config", cfg,
                              "--out", tmp_path / "b.csv")
    assert code == 0, err
    assert _applied(stdout, "screen/DROP") == 0
    assert len(pd.read_csv(tmp_path / "b.csv", header=None)) == 50  # kept without the context
    header = json.loads((tmp_path / "a.ledger.jsonl").read_text().splitlines()[0])
    assert header["inputs"]["context"]["path"] == str(ctx)


def test_reclass_hints_option_is_passed_through(tmp_path, capsys):
    src = tmp_path / "r.txt"
    table(box_rows(1, range(50), 100.0, 110.0, vx=12.0, w=20.0, h=40.0)).to_csv(
        src, index=False, header=False)
    hints = tmp_path / "hints.csv"
    hints.write_text("track,cls,avg_score\n1,3,0.95\n")
    cfg = _cfg(tmp_path)
    code, stdout, err = _main(capsys, "run", src, "--fps", 10, "--config", cfg,
                              "--out", tmp_path / "a.csv", "--reclass-hints", hints)
    assert code == 0, err
    assert set(pd.read_csv(tmp_path / "a.csv", header=None)[7]) == {3}
    assert _applied(stdout, "screen/RECLASS") == 1
    code, _, err = _main(capsys, "run", src, "--fps", 10, "--config", cfg,
                         "--out", tmp_path / "b.csv")
    assert code == 0, err
    assert set(pd.read_csv(tmp_path / "b.csv", header=None)[7]) == {0}  # unhinted: unchanged


def test_video_option_supplies_the_frame_rate(tmp_path, capsys, synthetic_video):
    video, _ = synthetic_video
    src = tmp_path / "v.txt"
    table(box_rows(1, range(0, 20), 100.0, 100.0, vx=2.0),
          box_rows(2, range(24, 60), 148.0, 100.0, vx=2.0)).to_csv(src, index=False, header=False)
    out = tmp_path / "o.csv"
    code, stdout, err = _main(capsys, "run", src, "--video", video,
                              "--config", _cfg(tmp_path, kind="none"), "--out", out)
    assert code == 0, err
    header = json.loads((tmp_path / "o.ledger.jsonl").read_text().splitlines()[0])
    assert header["fps"] == 25.0 and header["fps_source"] == "video"
    assert header["inputs"]["video"]["path"] == str(video)
    assert json.loads(stdout)["out"] == str(out)


def test_a_value_error_is_reported_not_raised(tmp_path, capsys):
    bad = tmp_path / "bad.txt"
    bad.write_text("1,2,3\n")  # not a dnt track file
    code, stdout, err = _main(capsys, "run", bad, "--fps", 10, "--config", _cfg(tmp_path),
                              "--out", tmp_path / "o.csv")
    assert code == 2 and stdout == "" and err.startswith("dnt-refine: error:")


def test_parser_requires_config_and_out(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["run", str(_src(tmp_path)), "--out", "o.csv"])
    assert exc.value.code == 2


# ---- the console script ----------------------------------------------------------------------


def test_console_script_runs():
    exe = Path(sys.executable).with_name("dnt-refine")
    found = exe if exe.exists() else shutil.which("dnt-refine")
    if not found:
        pytest.skip("console script not installed; run: .venv/bin/pip install -e . --no-deps")
    r = subprocess.run([str(found), "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "dnt-refine" in r.stdout and "run" in r.stdout
    r = subprocess.run([str(found), "run", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--reclass-hints" in r.stdout


# ---- the public API and imports --------------------------------------------------------------


def test_public_api_objects_are_the_module_objects():
    import dnt.refine as refine
    import dnt.track
    from dnt.refine import config, interpolate, link, refiner

    assert sorted(refine.__all__) == ["RefineConfig", "RefineResult", "TrackRefiner",
                                      "interpolate_tracks_rts", "link_tracklets"]
    assert refine.TrackRefiner is refiner.TrackRefiner
    assert refine.RefineResult is refiner.RefineResult
    assert refine.RefineConfig is config.RefineConfig
    assert refine.interpolate_tracks_rts is interpolate.interpolate_tracks_rts
    assert refine.link_tracklets is link.link_tracklets
    assert dnt.track.interpolate_tracks_rts is interpolate.interpolate_tracks_rts
    assert dnt.track.link_tracklets is link.link_tracklets
    for name in refine.__all__:
        assert getattr(refine, name) is not None


@pytest.mark.parametrize("order", [("dnt.track", "dnt.refine"), ("dnt.refine", "dnt.track")])
def test_track_and_refine_import_in_either_order(order):
    code = (
        f"import {order[0]}\nimport {order[1]}\n"
        "import dnt.refine, dnt.track\n"
        "from dnt.track import post_process\n"
        "assert dnt.track.link_tracklets is dnt.refine.link_tracklets\n"
        "assert post_process.interpolate_tracks_rts is dnt.refine.interpolate_tracks_rts\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


# ---- docs and changelog ----------------------------------------------------------------------


def test_docs_pages_exist_and_are_in_the_nav():
    nav = (ROOT / "mkdocs.yml").read_text()
    for page in ("index", "refiner", "config", "events", "interpolate", "link"):
        path = ROOT / "docs" / "api" / "refine" / f"{page}.md"
        assert path.exists(), path
        assert f"api/refine/{page}.md" in nav
    assert "::: dnt.refine\n" in (ROOT / "docs/api/refine/index.md").read_text()
    for page, module in (("refiner", "refiner"), ("config", "config"), ("events", "events"),
                         ("interpolate", "interpolate"), ("link", "link")):
        assert f"::: dnt.refine.{module}\n" in (
            ROOT / "docs" / "api" / "refine" / f"{page}.md").read_text()
    post = (ROOT / "docs/api/track/post_process.md").read_text()
    assert "::: dnt.refine.link.link_tracklets" in post
    assert "::: dnt.refine.interpolate.interpolate_tracks_rts" in post


def _note_block(text, start_marker):
    """Return the lines of the block that begins at ``start_marker`` and is indented after it."""
    lines = text.splitlines()
    first = next(i for i, ln in enumerate(lines) if ln.startswith(start_marker))
    block = [lines[first]]
    for ln in lines[first + 1 :]:
        if not ln.startswith("  "):  # a blank line, heading, directive or next bullet ends it
            break
        block.append(ln)
    return " ".join(" ".join(block).split())


def test_docs_and_changelog_state_the_limitation_accurately():
    index = (ROOT / "docs/api/refine/index.md").read_text()
    log = (ROOT / "docs/changelog.md").read_text()
    unreleased = log.split("## Unreleased", 1)[1].split("\n## ", 1)[0]
    notes = {
        "index.md": _note_block(index, '!!! note "Current limitations"'),
        "changelog": _note_block(unreleased, "- This release scores with motion only"),
    }
    for where, note in notes.items():
        for word in ("HUMAN_PENDING", "motion", "occlusion", "ambiguous", "static",
                     "not applied yet" if where == "changelog" else "never applied yet"):
            assert word in note, (where, word)
        assert "static objects" in note and "mixed tracks" in note, where
        assert "ID-switch splits" in note and "ambiguous assignment margin" in note, where
        # final review M9: hint-less rider reclasses stay pending; in-vehicle needs context
        assert "Rider reclasses whose subtype no ReClass hint settles are pending too" in note
        assert "In-vehicle drops need a context file" in note and "`--context`" in note, where
    assert "dnt-refine run" in index and "dnt-refine run" in unreleased
    assert log.index("## Unreleased") < log.index("## 0.3.3")
    # final review M8: the root CHANGELOG.md mirrors docs/changelog.md
    assert (ROOT / "CHANGELOG.md").read_text() == log


def test_missing_encoder_package_is_a_clean_error(tmp_path, capsys, synthetic_video, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    video, _ = synthetic_video
    src = tmp_path / "t.txt"
    table(box_rows(1, range(100), 10.0, 40.0, vx=1.5)).to_csv(src, index=False, header=False)
    code, stdout, err = _main(
        capsys, "run", src, "--video", video, "--config", _cfg(tmp_path),
        "--out", tmp_path / "o.txt",
    )
    assert code == 2 and "refine-dino" in err and stdout == ""
    assert not (tmp_path / "o.txt").exists()
