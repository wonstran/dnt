# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`dnt` is a Python package (published to PyPI as `dnt`) for video-based traffic analysis. It runs object detection, multi-object tracking, track post-processing, and video labeling. The source uses a `src/` layout (`src/dnt`). The docs site is https://wonstran.github.io/dnt/.

## Commands

A virtualenv lives at `.venv/`. Run `source .venv/bin/activate` first.

```bash
pip install -r requirements.txt   # deps (torch index pinned to cu116 in requirements.txt)
pip install -e .                  # editable install of src/dnt
python -m build                   # sdist + wheel into dist/ (setuptools backend)
ruff check src                    # lint (rules: E,F,I,UP,B,SIM,RUF,D; line-length 100)
ruff format src                   # format
mkdocs serve                      # API docs from numpy-style docstrings (mkdocstrings)
mkdocs build                      # writes site/ and site/dnt-manual.pdf (with-pdf plugin)
```

Tests: `.venv/bin/python -m pytest` (default excludes `model`, `golden`, `slow`), a single test:
`.venv/bin/python -m pytest tests/test_tracker_args.py::test_unknown_extra_key_raises -v`.
`-m slow` builds the wheel and runs the Python 3.11 smoke pipeline (`DNT_SMOKE_PYTHON`);
`-m golden` compares against 0.3.2.4 and needs `DNT_REF_CLIP`, `DNT_GOLDEN_DIR` and the reference
environment in `tests/golden/reference-env.txt` (see `tools/make_golden.py`).

`examples/*.py` are ad-hoc driver scripts. They put `src/` on `sys.path` themselves and use hard-coded `/mnt/d/videos/...` paths. Some of them may lag behind API changes.

Releases: bump the version in **both** `pyproject.toml` and `src/dnt/__init__.py` (`__version__`).

## Architecture

The pipeline passes data between stages as **headerless CSV text files**. Each stage reads the previous stage's file and writes its own:

1. **Detection**: `dnt.detect.Detector` (`detect/yolo/detector.py`) wraps Ultralytics YOLO (v8/11/26) or RT-DETR. You choose the model with the `DetectorModel` enum. Weights resolve under `src/dnt/detect/yolo/models/`, which is gitignored. Ultralytics fetches known weight names there on first use. A custom `weights=` path is treated as relative to that directory unless it is absolute. Output columns (`Detector.DET_FIELDS`): `frame, res, x, y, w, h, conf, class`.
2. **Tracking**: `dnt.track.Tracker` (`track/tracker.py`) wraps **BoxMOT**. You pick the backend by passing one of the per-algorithm config dataclasses (`BoTSORTConfig`, `ByteTrackConfig`, `OCSORTConfig`, `StrongSORTConfig`, `DeepOCSORTConfig`, `HybridSORTConfig`, `BoostTrackConfig`, `SFSORTConfig`, all subclasses of `MOTBaseConfig`) as `config=`. The config's `model` field selects the `MOTModels` backend. Configs round-trip through YAML (`export_yaml`/`import_yaml`, or `Tracker(config_yaml=...)`). ReID weights resolve under `track/reid_weights/`. The call is `track(det_file, out_file, video_file=...)`, and `track_batch` handles multiple videos. Output columns (`Tracker.TRACK_FIELDS`): `frame, track, x, y, w, h, score, cls, r3, r4`. The tracker also patches BoxMOT logging and BoxMOT's auto-requirements installer at import time.
3. **Post-processing** (`track/post_process.py`):
   - `interpolate_tracks_rts` fills gaps with a Kalman RTS smoother. An `interp == 1` flag marks filled frames, and legacy files use `-1` for real frames.
   - `link_tracklets` stitches broken IDs. It applies gating, builds a cost matrix, runs a Hungarian assignment, then merges IDs with union-find.
   - `track/re_class.py` (`ReClass`) does class re-assignment. It is optional: `track/__init__.py` sets it to `None` if `cython_bbox` is missing.
4. **Filtering**: `dnt.filter.Filter` filters detections or tracks by zones and lines (shapely/geopandas).
5. **Labeling**: `dnt.label.Labeler` (`label/labeler.py`) renders dets, tracks, and shapes onto video, cuts clips, and exports frames.

Supporting modules:
- `dnt.engine`: numeric helpers for bbox IoU, IoB, bbox interpolation, and gap clustering.
- `dnt.shared`: CSV read/write helpers (`files.py`), class-name lists (`shared/data/*.names`, loaded via `util.load_classes`), downloads, and a multi-video `Synchronizer`.
- `detect/signal/`: a pedestrian-signal detector with bundled `.pt` weights.
- `detect/timestamp.py`: reads timestamps from frames with OCR (easyocr).

### Import quirk

Every package `__init__.py` (and some modules) append their own directory to `sys.path`. As a result, some modules use **non-relative sibling imports**, such as `from shared.util import ...` in `labeler.py` and `from shared.download import ...` in `signal/detector.py`. Keep this in mind when moving modules or adding imports. Changing these to package-relative imports is safer, but they have to keep working both from an installed wheel and from `src/` on `sys.path`.

### Conventions

- Docstrings are numpy-style. mkdocstrings renders them into the API manual, and ruff's `D` rules enforce them. `docs/api/*.md` pages just point mkdocstrings at modules, so a new public module needs a page there and an entry in `mkdocs.yml` `nav`.
- In `Detector`, `device="auto"` resolves in the order cuda → xpu → mps → cpu.
- `build/`, `dist/`, `site/`, and `.venv/` are build outputs. Don't edit them by hand. Note that `site/` is committed.
