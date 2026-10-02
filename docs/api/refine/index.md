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
`base_url` and `model`, and usually no key (see [Endpoint and key](#endpoint-and-key)). The
`openai_compat` request sends `max_tokens=300` and `temperature`, which suits vLLM, Ollama, and
GPT-4-class chat models; reasoning models that reject `max_tokens` or `temperature=0` are not
supported by this backend:

```yaml
vlm:
  backend: openai_compat
  base_url: http://localhost:8000/v1
  model: Qwen/Qwen2.5-VL-7B-Instruct
  min_conf: 0.7
```

Anthropic needs a key (by default from `ANTHROPIC_API_KEY`); `model` defaults to
`claude-sonnet-5-5`:

```yaml
vlm:
  backend: anthropic
  model: claude-sonnet-5-5   # optional
  max_calls: 200
```

The Anthropic request asks for effort `low`. `vlm.vote_temperature` is sent only to models that
accept sampling parameters: `openai_compat` servers and the older Claude models (for example
`claude-haiku-4-5` or `claude-sonnet-4-6`). The newer Claude models (`claude-sonnet-5*`, the
default among them, `claude-opus-5*`, `claude-opus-4-7`, `claude-opus-4-8`, `claude-fable*`,
`claude-mythos*`) reject a non-default temperature, so no temperature is sent to them and with
`vlm.votes` above 1 their votes sample at the model's default temperature. A refusal, or a reply
cut off at the token limit before its answer, leaves the event `HUMAN_PENDING` with that error
and is not retried.

### Endpoint and key

`vlm.base_url` sets the endpoint of either backend: the server for `openai_compat`, or a proxy
or gateway for `anthropic` (unset: Anthropic's own API). It must be an `http://` or `https://`
URL with a host and a valid port, and without whitespace, a user name, password, query string,
or fragment (the config is copied into the ledger, so nothing secret may sit in it). The
`anthropic` backend used to ignore `base_url`; it now honours it, so a `base_url` left in an
older `backend: anthropic` config takes effect: remove it unless you mean it. When a key is sent
over plain `http://` to a host other than this machine, a warning names the host (use
`https://` unless the network is trusted). A local vLLM or Ollama server needs no key:

```yaml
vlm:
  backend: openai_compat
  base_url: http://localhost:8000/v1    # Ollama: http://localhost:11434/v1
  model: Qwen/Qwen2.5-VL-7B-Instruct
```

A hosted OpenAI-compatible endpoint reads its key from an environment variable you name (the
name, never the key):

```yaml
vlm:
  backend: openai_compat
  base_url: https://llm.example.com/v1
  model: some-vision-model
  api_key_env: EXAMPLE_API_KEY   # export EXAMPLE_API_KEY=... before the run
```

Anthropic, directly (key in `ANTHROPIC_API_KEY`) or through a gateway. The SDK appends
`/v1/messages` to `base_url`, so give the gateway's root, not a URL ending in `/v1`:

```yaml
vlm:
  backend: anthropic             # the key comes from ANTHROPIC_API_KEY
```

```yaml
vlm:
  backend: anthropic
  base_url: https://gateway.example.com/anthropic
  api_key_env: GATEWAY_KEY
```

A key can also live in a file. Surrounding whitespace (such as a trailing newline or CRLF)
and a UTF-8 byte order mark are stripped and `~` is expanded. The file must hold the key alone:
one line of printable ASCII without whitespace, not a `NAME=value` line or a comment (at most
64 KiB is read). The same rule applies to a key from Python or from the environment (a blank
variable counts as unset); a key that breaks it is an error naming the file, argument or
variable, never the key. The path goes into the ledger; the content never does. The file is
read once per `refine()` call, so once per video under `refine_batch`; a one-shot source such
as `/dev/stdin` works for a single video only:

```yaml
vlm:
  backend: anthropic
  api_key_file: ~/.config/dnt/anthropic.key   # chmod 600
```

In Python, pass the key to the refiner. It stays on the `TrackRefiner` object and is never
written to the config, the ledger, the summary, or a log line. It cannot be combined with a
custom `vlm_backend_factory`, which does not receive it (give the key to your factory):

```python
key = vault.read("vlm/anthropic")  # for example, from your secrets manager
refiner = TrackRefiner(config_yaml="ped.yaml", vlm_api_key=key)
```

On the command line, `dnt-refine run` takes `--vlm-backend`, `--vlm-model`, `--vlm-base-url`,
`--vlm-api-key-env`, and `--vlm-api-key-file`. They are applied to the `--config` file's `vlm`
block before it is checked, so a flag can complete a file (for example `--vlm-model` for a
template without one). An empty value, such as `--vlm-base-url ""`, resets the field; a flag
not given leaves the file's value. When `--vlm-backend` names a different backend than the file,
the file's `model`, `base_url`, `api_key_env`, and `api_key_file` belong to the other backend
and are not used (give them again with their flags if you want them): a vLLM file run with
`--vlm-backend anthropic` never sends the Anthropic key to the vLLM server. The ledger header
records the settings in effect:

```bash
dnt-refine run ped_track.txt --video cam1.mp4 --config ped.yaml --out ped_refined.txt \
    --vlm-backend anthropic --vlm-api-key-file ~/.config/dnt/anthropic.key
```

The key is taken from the first of these that is set: `vlm_api_key=` in Python, then
`vlm.api_key_file`, then the variable `vlm.api_key_env` names (`OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` by default). An unreadable or empty key file is an error, not a fall-back
to the variable. The key is resolved when the backend is built, which happens only with a video
(without one the backend is ignored): then, with no key, `anthropic` fails before any work
starts, and so does `openai_compat` when it has no `base_url` (it would call api.openai.com);
with a `base_url` it sends a placeholder key. Each error names the three ways to give a key and
never shows one; `dnt-refine` reports it with exit code 2. The key is replaced by `***` in every
error text that `refine` records in the ledger or logs. The `openai` and `anthropic` SDKs' own
debug logging is not covered: do not turn it on in logs that others read. The check that
`vlm.api_key_file` is not a pasted key is best-effort (it catches the `sk-` prefix of OpenAI and
Anthropic keys only).

There is no `--vlm-api-key` option and no key field in the config, on purpose: a key typed on
the command line is kept in the shell history and shown to other users in the process list, and
the config is copied into every ledger header.

The images you send leave your machine for a remote backend; set `vlm.send_context_frames:
false` to send only the crops.

### Answers and decisions

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

**Budget.** `vlm.max_calls` (500 by default) is a hard limit on requests to the backend in each
`refine()` call: the `calls` and `retries` counts never add up to more. Under `refine_batch` it
is a budget per video (each video gets its own runner; the answer cache is shared). A question
is asked only if all its uncached votes fit in what is left; a question that does not fit stays
`HUMAN_PENDING` with `vlm.error: "budget"`, and a later one that fits still runs. Questions are
admitted in order of how close the event's score is to the middle of its band, but only within
each routing batch: the switch stage, the screen stage, and each link pass, which run in that
order, so earlier stages spend first. The `max_calls` cap holds for the whole run, but the
closest-first order does not: later stages only exist after earlier stages' edits are applied,
so one run-wide order is not possible. If the budget runs out before the later stages, raise
`vlm.max_calls` instead of expecting global ordering (with `max_calls: 200` and 250 uncertain
splits, every link question is skipped, however close to its band middle). A retry (after a
timeout, HTTP 429 or 5xx, or an invalid reply) uses what admission left over; with nothing left,
the event stays pending with the same error.
`refiner.last_result.summary["vlm"]` reports `calls`, `retries`, `cache_hits`, `failures`,
`budget_skipped`, and `no_evidence` (events not asked because no evidence image could be made).

**Cache.** Each answer is stored under `vlm.cache_dir` (`~/.cache/dnt/vlm` by default), keyed
by the image, prompt, options, backend, model, temperature, and vote number. Every vote is
cached on its own, so a rerun on the same inputs answers them from the cache and costs no budget
for them; a vote tie is reproduced from the cached votes and stays `HUMAN_PENDING`. Only a vote
that failed or got an invalid reply is not cached, and only that vote is asked again. A damaged
entry is ignored.

**Votes.** With `vlm.votes` above 1, each question is asked that many times at
`vlm.vote_temperature` (at the default temperature for the newer Claude models; see above). The majority answer wins, its confidence is the share of votes it got
(the model's own confidence is ignored), and a tie is `HUMAN_PENDING`. Every vote that is sent
(not one answered from the cache) counts against `max_calls`, and so does every retry.

**Failures** never stop a run and never apply an edit: a timeout, a rate limit, a server error,
an invalid reply, a missing evidence image, or an exhausted budget leaves the event
`HUMAN_PENDING` with the reason in its `vlm.error`. An error that will not go away (HTTP 400,
401, 403, 404 or 422: a bad request, key, permission, or model), or three other unexpected
errors in a row, stops the run's remaining questions: one warning is logged, and every question
not yet sent ends with `vlm.error: "aborted after a fatal API error: ..."` without a request. A
batch with failures logs one warning with their count and the first error. Without a video the
backend is ignored with a warning. With a video, a backend whose package is missing raises
`ImportError` before any work starts, naming `pip install 'dnt[refine-vlm]'`. A video whose
container does not report its frame count (raw `.h264`, some `.ts` or `.mkv`) still works: a
warning says the evidence frames are not range-checked, and a frame that cannot be read is left
out of the image.

**Review page.** Events left `HUMAN_PENDING` (except fills and smoothing) are collected on
`OUT.review.html`, whether or not a backend was used (with `vlm.backend: none` too), with their images in `OUT.review/`. One card per event shows the evidence
image, the event's tracks with their output track ids, the signals, the VLM's answer and
reason, and accept and reject choices (with a class picker for reclasses). Cards can be filtered by stage and sorted by score. A copy button gives
a `Labeler.draw_track_clips(...)` snippet that cuts one clip per track of the event, each from
the track's first to last frame plus 2 s on each side (it is left out when there is no video).
The page makes no network requests and keeps your choices in the browser, namespaced by run.
**Export decisions** downloads them as `decisions.json`, one object per decided event:
`{"switch-r0-000001": {"accept": true, "new_cls": 3, "proposal_key": "...", "run_key": "..."}}`
(`new_cls` only when a class was picked for an accepted event). Event IDs are reused for
different proposals after a rerun, so `proposal_key` (the event's) and `run_key` (the page's)
say which proposal and run each decision was made on; `apply` will refuse an entry whose
`proposal_key` does not match the ledger and warn when the `run_key` differs. Hand-written files
may also use `"accept"`, `"reject"`, or `{"accept": true, "new_cls": 3}`. Applying the file
arrives in a later release. A rerun deletes only the images listed in `OUT.review/.dnt-review.json`, the page's own
manifest, and never overwrites another file. Without a video the page is still written, with
signals only.

::: dnt.refine
