# Track Refinement (`dnt.refine`)

`dnt.refine` refines a track file after tracking: it splits ID switches, screens false tracks,
links fragments (including across static waits and witnessed occlusions), drops orphans, and
fills gaps. Every edit is recorded in a JSONL ledger next to the output.

Used like `Detector` and `Tracker`:

```python
from dnt.refine import RefineConfig, TrackRefiner

cfg = RefineConfig.defaults("vehicle")
# motion only; the default encoder (dino) needs pip install 'dnt[refine-dino]' (see Appearance)
cfg.encoder.kind = "none"
refiner = TrackRefiner(config=cfg)  # or config_yaml="veh.yaml"
tracks = refiner.refine("cam1_track.txt", "cam1_refined.txt", video_file="cam1.mp4")
refiner.refine_batch(track_files, video_files=video_files, output_path="refined/")
```

Command line: `dnt-refine run TRACKS --fps 10 --config refine.yaml --out clean.csv`.

!!! note "Current limitations"
    This release scores with motion only unless you give a video and an appearance encoder
    (see Appearance below), and cannot yet apply decisions made in review.
    Edits it is sure of are applied: in-vehicle and duplicate false-track drops, rider
    reclasses whose subtype a ReClass hint settles, links across short gaps and static waits
    with a clear assignment margin, orphan drops, and gap filling. With an encoder, ID-switch
    splits and links are also scored by appearance and applied when they score high enough.
    Other edits are proposed but never applied yet, because their scores are capped below
    auto-accept: ID-switch splits found from motion alone, links across occlusions, links with
    an ambiguous assignment margin, and false-track drops of static objects or of mixed tracks.
    Rider reclasses whose subtype no ReClass hint settles are pending too, however high they
    score, because only a hint can choose the subtype in this release. They appear in the
    ledger as `HUMAN_PENDING` and leave the tracks unchanged. In-vehicle drops need a context
    file with the vehicles' boxes (`context_file=`, or `--context`); without one the in-vehicle
    cue is skipped. VLM verification and applying review decisions follow in later releases.

## Link band

A link is applied when its score reaches `link.accept_above` (0.62 by default, in motion-only
mode too). Links across occlusions and links with an ambiguous assignment margin are capped at
`link.occluded_score_cap` and `link.ambiguous_cap` (0.60 by default), so they stay pending.
Raise `link.accept_above` to apply fewer links; config validation keeps both caps below it.
The score is multiplied by a border prior, `0.8 + 0.2 * b`, where `b` is 0 for a pair that ends
or starts near the image border. With `accept_above` 0.62, such border links can now be
applied too: on the three audited clips, 2 of the 8 applied links (beginning 98->134, middle
221->222) were border links, and both were among the audited, correct ones.

## Appearance

With a video, `refine` also compares what tracks look like. It crops each box, skips crops that
another box overlaps, embeds the rest with a pretrained encoder, and uses the embeddings to
find ID switches (stage 1) and to score links (stage 3). Install an encoder first:

```bash
pip install 'dnt[refine-dino]'   # DINOv2 (the default, kind: dino)
pip install 'dnt[refine-reid]'   # torchreid OSNet (kind: reid), with tensorboard
```

```yaml
encoder:
  kind: dino        # dino | reid | none
  model: facebook/dinov2-small
  device: auto      # cuda, xpu, mps, then cpu
  sample_every: 5   # embed every 5th observed frame; stage 1 densifies around candidates
```

`encoder.kind: none`, or no video, scores with motion only and needs no extra. A video with the
default `dino` encoder and no package installed raises `ImportError` before any work starts.
The first `dino` run downloads the model from the Hugging Face Hub, so it needs network access;
`transformers` caches it for later runs. If `kind: reid` reports that torchreid has no
`FeatureExtractor`, install deep-person-reid from GitHub instead:
`pip install git+https://github.com/KaiyangZhou/deep-person-reid.git`.
The embeddings are saved as `OUT.features.npz` next to the output and reused when the track
file, video, context file, and encoder model, weights, and sampling settings are unchanged.
Each stage has a minimum box size. Stage 1 ignores the appearance of boxes whose longer side
is below `switch.min_crop_px` (40 px by default); stage 3 does the same with
`link.min_crop_px` (0 by default: every clean crop). A track whose boxes are all smaller gets
no ID-switch proposal at all, and a link whose ends have no crop left is scored with
appearance unknown. On three 640x480 pedestrian clips, where most boxes were smaller than
40 px, the ID-switch splits found from small crops cut single pedestrians, while most links
scored from small crops were correct. On three 640x480 pedestrian clips (beginning, evening, middle), stage 1 proposed 2, 20 and
5 splits with the earlier default (no filter), 0, 3 and 1 with `switch.min_crop_px: 40`, and
42, 209 and 85 in motion-only mode (`encoder.kind: none`); the filter removed 58-70% of the
coarse samples. The filtered counts are approximate: dense frames missing from the cached
embeddings were left out.
Set a value to `0` to use every crop in that stage when your boxes are mostly larger than the
default (distant pedestrians stay small at any resolution); a higher value uses fewer, larger
crops. `refine` logs how many samples each stage uses.

```yaml
switch:
  min_crop_px: 40   # stage 1 ignores the appearance of smaller boxes (longer side, px)
link:
  min_crop_px: 0    # stage 3 compares every clean crop
```

For vehicles, `reid` needs `encoder.weights`. The cache key includes a digest of the weights
that were actually loaded, so a model that changes under the same name never reuses old
embeddings.

::: dnt.refine
