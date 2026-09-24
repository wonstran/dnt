# Changelog

## 0.3.3 — 2026-09-23

### Important
- **Tracker settings now take effect.** In 0.3.2.x every tuning field of the tracker configs was silently
  ignored and BoxMOT's defaults were used. Settings now reach BoxMOT. Configs you never tuned keep their
  results: with the dependency versions in `tests/golden/reference-env.txt`, untuned tracks are unchanged
  from 0.3.2.4. Tuning fields now default to `None`, meaning "BoxMOT's value".
- Tracker configs are keyword-only: `BoTSORTConfig(ReIDWeights.X)` now raises `TypeError`
  (it previously set `model` by mistake). Use `BoTSORTConfig(reid_weights=ReIDWeights.X)`.
- Settings that cannot affect the chosen tracker raise `ValueError`:
  SF-SORT `det_thresh`, `max_age`, `min_hits`, `iou_threshold`, `asso_func`; BoostTrack `asso_func`.
  SF-SORT values outside the range BoxMOT would silently clamp also raise.
- `extra_kwargs`: 0.3.2.x factory keys that worked (`evolve_param_dict`, `tracker_config`, `per_class`,
  `reid_weights` on ReID trackers, and a `tracker_type` override on an untuned config) still work with
  identical results and are deprecated for 0.4. A `tracker_type` override combined with the source config's
  own tuning fields (which 0.3.2.4 discarded) now raises. Keys that were silently ignored now raise; other
  tracker parameters are validated like fields.
- Saved configs: `from_dict()`, `import_yaml()` and `Tracker(config_yaml=...)` recognise 0.3.2.x output
  (YAML files and `to_dict()` dicts) and migrate it — untouched defaults follow BoxMOT (results unchanged),
  changed values take effect. Limitation: a value deliberately set equal to the old dnt default also
  migrates; the load warning lists every such field. 0.3.3 mappings carry `dnt_config_version: 2`.
  `XConfig(**old_dict)` is not migrated (dnt warns when it sees that pattern); use `from_dict`.
- `Synchronizer`: timestamps now continue across videos and `offsets` are applied (formula in the docstring);
  results change for every video after the first and wherever offsets are non-zero.
  A non-zero `offsets[0]` raises `ValueError`.
- Golden comparison: with the reference environment, untuned tracks are byte-identical to 0.3.2.4 for
  ByteTrack, BoT-SORT, StrongSORT, BoostTrack and SF-SORT and the `evolve_param_dict` passthrough. OC-SORT,
  Deep OC-SORT, HybridSORT and a `tracker_type` override to OC-SORT are exempt (reviewed defects
  BX16-NP2-*): they crash identically on 0.3.2.4 and 0.3.3 (see Known issues).

### Fixed
- `interpolate_tracks_rts(track_file=...)` no longer drops the first row (B3).
- `Filter.interpolate_tracks_rts` uses the package module instead of loading a second copy (B4).
- `Filter.filter_iou` works with zones (Polygon, MultiPolygon or list) (B5).
- `ReClass()` works with its defaults; `Detector(model=...)` accepts names such as `"rtdetr-x"` (B6).
- FP16 is used on indexed CUDA devices (`cuda:0`); the tracker only uses half precision on CUDA (B8).
- `detect_frames` rounds like `detect` (B9).
- `Labeler.draw_dets` accepts `Detector.detect()` output (B10).
- `SignalDetector` handles a short last batch (B13), no longer downloads ImageNet weights and loads on CPU-only machines.
- `Labeler` releases video captures; `export_track_frames(bbox=False)` writes frames (B14).
- `Tracker.track` no longer re-runs `update()` after a `TypeError`; empty detection files produce an empty track file.
- `track_batch` output names strip only a trailing `_iou`.

### Packaging
- Requires Python 3.11+ (the code already did).
- `boxmot==16.0.11` pinned (0.3.2.4 allowed BoxMOT 25, whose API breaks tracking).
- The class-name files (`coco/openimages/voc.names`) now ship in the wheel (B7).
- The default ReID weight (`osnet_x1_0_msmt17.pt`, used by `Tracker()`'s default BoT-SORT) and the
  pedestrian-signal weight (`ped_signal.pt`) now ship in the wheel, so a clean install no longer downloads them on first use.
- Removed dependencies dnt never imports — install them yourself if you relied on them:
  torchaudio, opencv-contrib-python, faiss-cpu, faiss-gpu-cu12, cython, easydict, h5py, motmetrics, ninja,
  numpy_indexed, prettytable, pycocotools, scikit-image, shapelysmooth, sympy, tabulate, tensorboard,
  termcolor, thop, vidgear, openpyxl, matplotlib, gdown, lapx, loguru, yacs, scikit-learn.
- Top-level module names such as `detector`, `shared` and `filter` are no longer importable (they were
  side effects of `sys.path` manipulation, never public API).

### Deprecated
- `StrongSORTConfig(max_dist=...)`: use `max_cos_dist`.
- `extra_kwargs` keys `evolve_param_dict`, `tracker_config`, `per_class`, `reid_weights`, `tracker_type`,
  `device`, `half` (removed in 0.4).

### Known issues
- OC-SORT, Deep OC-SORT and HybridSORT (including `extra_kwargs["tracker_type"]` overrides targeting them)
  raise `TypeError: only 0-dimensional arrays can be converted to Python scalars` as soon as a lost track
  is re-detected. This is a BoxMOT 16.0.11 defect in `unfreeze()` under numpy >= 2 and was already present
  in 0.3.2.x. Use ByteTrack, BoT-SORT, StrongSORT, BoostTrack or SF-SORT until the BoxMOT upgrade planned
  for dnt 0.4.
- `extra_kwargs["tracker_type"]` overrides from a non-ReID config (ByteTrack, OC-SORT, SF-SORT) to a ReID
  tracker now raise `ValueError`; in 0.3.2.4 all five such combinations crashed inside BoxMOT.
