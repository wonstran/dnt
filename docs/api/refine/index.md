# Track Refinement (`dnt.refine`)

`dnt.refine` refines a track file after tracking: it splits ID switches, screens false tracks,
links fragments (including across static waits and witnessed occlusions), drops orphans, and
fills gaps. Every edit is recorded in a JSONL ledger next to the output.

Used like `Detector` and `Tracker`:

```python
from dnt.refine import RefineConfig, TrackRefiner

refiner = TrackRefiner(config=RefineConfig.defaults("vehicle"))  # or config_yaml="veh.yaml"
tracks = refiner.refine("cam1_track.txt", "cam1_refined.txt", video_file="cam1.mp4")
refiner.refine_batch(track_files, video_files=video_files, output_path="refined/")
```

Command line: `dnt-refine run TRACKS --fps 10 --config refine.yaml --out clean.csv`.

!!! note "Current limitations"
    This release scores with motion only, and cannot yet apply decisions made in review.
    Edits it is sure of are applied: false-track drops and reclasses, links across short gaps
    and static waits, orphan drops, and gap filling. Two kinds of edit are proposed but never
    applied yet, because their scores are capped below auto-accept: ID-switch splits found
    from motion alone, and links across occlusions. Both appear in the ledger as
    `HUMAN_PENDING` and leave the tracks unchanged. VLM verification and applying review
    decisions follow in later releases.

::: dnt.refine
