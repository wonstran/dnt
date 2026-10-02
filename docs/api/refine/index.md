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
    Without a VLM backend (the default), other edits are proposed but never applied yet,
    because their scores are capped below auto-accept: ID-switch splits found from motion alone,
    links across occlusions, links with an ambiguous assignment margin, and false-track drops of
    static objects or of mixed tracks. Rider reclasses whose subtype no ReClass hint settles are
    pending too, however high they score, unless a VLM backend names the subtype. They appear in
    the ledger as `HUMAN_PENDING` and leave the tracks unchanged. With a VLM backend and a video
    (see Verification with a VLM below), the VLM decides these edits when it is sure; the rest
    stay `HUMAN_PENDING` and go on a review page. In-vehicle drops need a context file with the
    vehicles' boxes (`context_file=`, or `--context`); without one the in-vehicle cue is
    skipped. Applying review decisions follows in a later release.

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

## Verification with a VLM

With a video and a vision-language model (VLM), `refine` can settle the edits whose scores fall
in the uncertain band (between `accept_above` and `reject_below` of their stage), and the rider
reclasses that still lack a subtype. For each one it builds a composite image (crops of the
tracks, plus context frames unless `vlm.send_context_frames` is `false`), asks the model to pick
one option, and maps the answer to a decision. Only ID-switch splits, links, false-track drops,
and reclasses are asked about; orphan drops, fills, and smoothing never are.

Install the client library for your backend (neither is a required dependency):

```bash
pip install 'dnt[refine-vlm]'   # openai and anthropic
```

Set the backend in the `vlm` block of the config file (`--config`, or `config_yaml=`), or in
`cfg.vlm` in code. A local server that speaks the OpenAI chat API (vLLM, Ollama, ...) needs
`base_url` and `model`. Its key is optional (`vlm.api_key_env` names the variable; `OPENAI_API_KEY` by default):

```yaml
vlm:
  backend: openai_compat
  base_url: http://localhost:8000/v1
  model: Qwen/Qwen2.5-VL-7B-Instruct
  min_conf: 0.7
```

Anthropic needs the key in `ANTHROPIC_API_KEY` (or the variable named by `vlm.api_key_env`);
`model` defaults to `claude-sonnet-5-5`:

```yaml
vlm:
  backend: anthropic
  model: claude-sonnet-5-5   # optional
  max_calls: 200
```

The images you send leave your machine for a remote backend; set `vlm.send_context_frames:
false` to send only the crops. Keys are read from the environment and never written to the
ledger or the config.

The model must reply with an option, a confidence, and a reason. The answer maps to a decision
only when its confidence reaches `vlm.min_conf`; `unsure`, a lower confidence, a tie between
votes, or an error leaves the event `HUMAN_PENDING`.

| Event | Answer | Decision |
|---|---|---|
| ID-switch split | `different` | `VLM_ACCEPT`: the split is applied |
| ID-switch split | `same_individual` | `VLM_REJECT` |
| Link | `same_individual` | `VLM_ACCEPT`: the link is applied |
| Link | `different` | `VLM_REJECT` |
| Person screen | `pedestrian` | `VLM_REJECT` |
| Person screen | `person_in_vehicle` | `VLM_ACCEPT` as a drop (in-vehicle) |
| Person screen | `not_a_person` | `VLM_ACCEPT` as a drop (static) |
| Person screen | `cyclist`, `motorcycle_rider`, `scooter_rider` | `VLM_ACCEPT` as a reclass to the `reclass_map` class (`HUMAN_PENDING` if the map has no entry) |
| Vehicle screen | `vehicle` | `VLM_REJECT` |
| Vehicle screen | `part_or_duplicate_of_another_vehicle` | `VLM_ACCEPT` for a duplicate drop; `HUMAN_PENDING` for any other event |
| Vehicle screen | `not_a_vehicle` | `VLM_ACCEPT` as a drop (static) |
| Rider reclass without a subtype | a rider answer | the reclass is applied with that subtype (decision `AUTO_ACCEPT`, since the rider score already accepted it); if `reclass_map` has no entry for the subtype, it stays `HUMAN_PENDING` with `needs_subtype` in its signals |
| Rider reclass without a subtype | any other answer | `HUMAN_PENDING`: the VLM and the rider score disagree |

An answer can redirect the edit: for a proposed static drop, `cyclist` applies a reclass
instead. The event keeps its proposal; only its `edit` changes.

**Budget.** `vlm.max_calls` (500 by default) is a hard limit on requests to the backend:
the `calls` and `retries` counts never add up to more. A question is asked only if all its
uncached votes fit in what is left; a question that does not fit stays `HUMAN_PENDING` with
`vlm.error: "budget"`, and a later one that fits still runs. Questions are admitted in order of
how close the event's score is to the middle of its band, but only within each routing batch:
the switch stage, the screen stage, and each link pass, which run in that order, so earlier
stages spend first. The `max_calls` cap holds for the whole run, but the closest-first order
does not: later stages only exist after earlier stages' edits are applied, so one run-wide
order is not possible. If the budget runs out before the later stages, raise `vlm.max_calls`
instead of expecting global ordering (with `max_calls: 200` and 250 uncertain splits, every
link question is skipped, however close to its band middle). A retry (after a timeout, HTTP 429 or 5xx, or an invalid reply)
uses what admission left over; with nothing left, the event stays pending with the same error.
`refiner.last_result.summary["vlm"]` reports `calls`, `retries`, `cache_hits`, `failures`, and
`budget_skipped`.

**Cache.** Each answer is stored under `vlm.cache_dir` (`~/.cache/dnt/vlm` by default), keyed
by the image, prompt, options, backend, model, temperature, and vote number. A rerun on the same
inputs does not ask again the questions that were answered, and costs no budget for them; a
question that failed, got an invalid reply, or ended in a vote tie is not cached and is asked
again. A damaged entry is ignored.

**Votes.** With `vlm.votes` above 1, each question is asked that many times at
`vlm.vote_temperature`. The majority answer wins, its confidence is the share of votes it got
(the model's own confidence is ignored), and a tie is `HUMAN_PENDING`. Every vote that is sent
(not one answered from the cache) counts against `max_calls`, and so does every retry.

**Failures** never stop a run and never apply an edit: a timeout, a rate limit, a server error,
an invalid reply, a missing evidence image, or an exhausted budget leaves the event
`HUMAN_PENDING` with the reason in its `vlm.error`. Without a video the backend is ignored with
a warning. With a video, a backend whose package is missing raises `ImportError` before any work
starts, naming `pip install 'dnt[refine-vlm]'`.

**Review page.** Events left `HUMAN_PENDING` (except fills and smoothing) are collected on
`OUT.review.html`, whether or not a backend was used (with `vlm.backend: none` too), with their images in `OUT.review/`. One card per event shows the evidence
image, the signals, the VLM's answer and reason, and accept and reject choices (with a class
picker for reclasses). Cards can be filtered by stage and sorted by score. A copy button gives
a `Labeler.draw_track_clips(...)` snippet that cuts one clip per track of the event, each from
the track's first to last frame plus 2 s on each side (it is left out when there is no video).
The page makes no network requests and keeps your choices in the browser, namespaced by run.
**Export decisions** downloads them as `decisions.json`; applying that file arrives in a later
release. A rerun deletes only the images listed in `OUT.review/.dnt-review.json`, the page's own
manifest, and never overwrites another file. Without a video the page is still written, with
signals only.

::: dnt.refine
