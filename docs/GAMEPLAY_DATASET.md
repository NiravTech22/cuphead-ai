# Gameplay dataset collection and gate

Current status: collection has begun but is **not complete**. The main real
keyboard recording captured **3,031 frames over 101.020 seconds** at 256×256
and 30 FPS. Review corrected its provisional `root_pack` label to
**Forest Follies** and split it into **150 menu/entry frames** (4.986 seconds)
and **2,881 attempt frames** (96.031 seconds): two full attempts, both deaths.
The original recording is preserved under
`data/recordings/root_pack_20260915T224811Z_647131753/`; its provisional metadata
is superseded by `experiments/gameplay_segmentation_20260915.json`.

Including the two earlier menu recordings, the integrity audit confirms
**3,640 eligible frames over 121.287 seconds**: attempt **2,881**, menu **759**,
idle **0**, full attempts **2**, successful clears **0**, with no invalid segments
or late intervals. The saved 240-sample encoder probe measured mean variance
**0.000143183** for independent images and **0.000130838** for consecutive clips;
both remain below the configured **0.00025602** diversity threshold. Training
remains blocked by insufficient coverage and the low-diversity result.
Evidence: `experiments/gameplay_dataset_gate.json` and
`experiments/gameplay_segmentation_20260915.json`. Raw recording images remain
local and are excluded from Git; source recording metadata and input/timing logs
are versioned.

This task collects data only. It does not train, load a Cuphead predictor, select
actions, or change the menu-navigation runner. The existing v1 memory bank is a
nearest-neighbor lookup over frozen embeddings and requires zero training.

## Capture settings and format

Provisional capture contract: **256×256 RGB, 30 FPS**, matching the spatial input
of the existing frozen encoder's gameplay clip branch. These settings are explicit
in each segment and the audit invocation; use the intended downstream training
settings consistently if they differ. At 30 FPS, 100,000–200,000 frames correspond
to about 56–111 minutes of capture, excluding breaks. This is sampled screen
capture, not a guarantee of capturing every internal 60 Hz game frame.

`scripts/record_session.py` reuses `X11WindowSource`, `IntegrityTracker`,
`ReplayWriter`, and the existing action normalization. Keyboard capture uses
X11/XWayland held-key queries; `--input-source gamepad` retains the evdev reader.
Both persist the same replay files:

- `frames/00000000.png`, etc.: lossless PNG, RGB, bilinear stretch to target size.
- `actions.jsonl`: unchanged frame-indexed normalized action format.
- `frames.jsonl`: matching frame index, capture timestamp, input-poll timestamp,
  source checksum, image path, and a raw input snapshot at that input sample.
  Keyboard snapshots have `backend: x11_keyboard`, named held-key states,
  `axes: {}`, and `focused: true`; gamepad snapshots retain `backend: evdev`.
- `meta.json`: type, boss/level, phase, observed outcome, full-attempt flag,
  reviewed-label flag, notes, resolution/rate, timing and image repetition counts.
- `events.json` and `hud.jsonl`: existing replay files; no invented event/HUD truth.

The physical input snapshot preserves menu, weapon-switch and super buttons that
the pruned action representation omits. Inputs are sampled immediately after the
image; timestamps expose this skew. Very short taps between samples can be
missed. The raw snapshot is cached from the same poll used for the normalized
action, not sampled again after saving the image. Nothing sends input to the game.

### Keyboard controls and schema compatibility

Use Cuphead's default bindings (restore defaults in-game if customized):

| Control | Key | Existing normalized action field |
| --- | --- | --- |
| Movement / aim | Arrow keys | `stick_x`, `stick_y` through `Action.from_raw` |
| Jump / parry | Z | `a` |
| Shoot | X | `x` |
| Dash | Left Shift | `b` |
| Aim lock | C | `rt` |
| EX / super | V | Raw snapshot only |
| Switch weapon | Tab | Raw snapshot only |
| Menu controls | Enter, Escape, Backspace (plus arrows/Z/X) | Raw snapshot preserves each key |

Gameplay defaults are documented in the [Cuphead controls guide](https://steamcommunity.com/sharedfiles/filedetails/?id=1310872602).
These output names describe the existing normalized action representation, not
the game's default physical controller layout.

`actions.jsonl` stays exactly `{ "frame": i, "action": { "stick_x": float,
"stick_y": float, "a": bool, "x": bool, "b": bool, "rt": bool } }`.
Arrows first map to -1/0/+1 (up positive; opposite arrows cancel), then pass
through the same `Action.from_raw(...).to_buttons()` path as gamepad input,
including diagonal aim normalization and simultaneous-button pruning. No new
action dimensions, frame indices, or timestamp fields are introduced.

**Existing limitation:** this pruned combat representation drops EX/super,
weapon switches, menu controls, and some simultaneous inputs; it does not fully
represent every human action. Raw snapshots preserve these controls for both
sources. A future predictor using only the six normalized fields cannot condition
on those omitted controls without a separate, explicit schema change. The current
sequence predictor accepts numeric action tensors; this task does not add a
replay-to-training tensor adapter or claim training validation.

Keyboard snapshots include only the listed Cuphead controls, not arbitrary typed
text. Segment metadata records `input_source` and `keyboard_bindings`; the gate
checks raw keyboard states against normalized actions and rejects unfocused,
missing-key, or malformed snapshots. Existing gamepad segments remain valid.

Real repeated images are retained and counted, including idle frames. Invalid
indices, backward/nonfinite timestamps, window resize and input-reader errors
abort finalization. Missed deadlines are never filled with fabricated frames.
The audit rejects >5% capture-rate error, gaps >3 nominal frame intervals, >1%
intervals longer than 1.5 frames, or input samples more than one frame late.
These are operational capture tolerances, not a claim of exact frame synchronization.

## Collect separate segments

Open Cuphead with default keyboard bindings and keep its window size fixed.
Install the existing capture dependencies from `requirements-memory.txt` (includes
`python-xlib`). Keyboard is the default input source. Start the recorder in a
terminal, then focus Cuphead within 60 seconds. The capture clock starts only when
the game has focus. It accepts focus on the game drawable or its child windows.
Switch away from Cuphead at the segment boundary to stop and save the complete
frames, then enter the observed labels in the terminal with `--label-after`.
Ctrl+C and the time/frame limit also stop capture. An unfocused final frame is
discarded before its image/action is written. A keyboard server error aborts the
segment rather than silently reusing stale input.

Keep recording from level entry through death/results to claim a full attempt.
Keyboard support targets the same Linux X11/XWayland game window as the existing
screen source; native Wayland/Windows keyboard recording is not implemented.
For future gamepad sessions use `--input-source gamepad`, optionally with
`--device-path /dev/input/eventN`. Do not mix input sources within a segment.

```bash
# Capture title → save → map → level entry manually as its own segment.
.venv/bin/python scripts/record_session.py --real --input-source keyboard --segment-type menu \
  --boss forest_follies --phase title_save_map_entry --fps 30 --width 256 --height 256 \
  --max-seconds 300 --label-after

# Repeat for at least five FULL attempts. Keep recording through death/results.
.venv/bin/python scripts/record_session.py --real --input-source keyboard --segment-type attempt \
  --boss forest_follies --phase whole_attempt --fps 30 --width 256 --height 256 \
  --max-seconds 1200 --label-after

# A short no-input baseline; report IDLE only if no game input occurred.
.venv/bin/python scripts/record_session.py --real --input-source keyboard --segment-type idle \
  --boss forest_follies --phase level_start --fps 30 --width 256 --height 256 \
  --max-seconds 30 --label-after
```

Record at least two attempts ending in death, and a clean clear if manually
achievable. A clear uses the existing replay outcome `KNOCKOUT`; it must be
witnessed, never inferred from reaching Forest Follies. Use phase notes for the
boss phases or level sections actually reached. Additional separately recorded
phase segments can use `--phase phase_2`, but do not label a partial phase as a
full attempt. Continue varied attempts until at least 100,000 frames (target
200,000), not just five short deaths. At least 70% of eligible frames must be
attempt footage; menu and idle each need at least 100 frames. These conservative
coverage choices prevent padding the requested total with baseline footage.

Labels can also be applied after reviewing a saved segment:

```bash
.venv/bin/python scripts/check_gameplay_dataset.py label data/replays/RUN_ID \
  --outcome DEATH --full-attempt --notes 'From level start through death; reached flower section'
```

Before review, a replay is marked unverified even if `--outcome` supplied a
provisional label. Failed/unfinalized segments are excluded. Synthetic plumbing
tests never count. Existing navigation videos remain historical evidence: their
20 FPS images and end-of-action transition logs cannot supply exact per-frame
human inputs at this dataset's 30 FPS. Re-record the navigation segment with this
recorder; do not invent missing actions or upsample it into qualifying frames.

## Mandatory diversity gate

```bash
.venv/bin/python scripts/check_gameplay_dataset.py audit \
  --root data/replays --boss forest_follies --width 256 --height 256 --fps 30 \
  --samples 240 --output experiments/gameplay_dataset_gate.json
```

The audit validates every stored PNG, action/image indices, raw inputs, labels,
and actual timing. It reports counts by type and outcome, full attempts, clears,
and summed timestamp-based capture duration (with the final frame interval
explicitly estimated). It fingerprints the validated dataset, including pixels,
so the report identifies the exact data checked. Re-run after any data change;
this report is not a permanent authorization to train on different data.

Sampling is deterministic and evenly spaced through type/boss/phase/outcome
strata. It requires **at least 200 distinct frame indices**; default 240. The
frozen encoder probe reports:

1. Independent-image 128px mode, matching the previous eight-screenshot probe.
2. Consecutive 256px clips from within each segment, with separate variances by
   segment type so diverse menus cannot hide uniform gameplay embeddings.

Both use the same frozen encoder, 384-dimensional unit-normalized embeddings and
population variance averaged over dimensions. The operational low-variance floor
is **0.00025602**, twice the earlier **0.00012801**. This is an explicit conservative
interpretation of “comparable,” not a scientifically calibrated collapse cutoff.
If either overall probe or the gameplay-only clip probe is at/below that floor,
the report flags **still-collapsed / low diversity** and calls for checking the
encoder/preprocessing before simply collecting more of the same footage. Low
variance alone cannot distinguish an encoder problem from homogeneous inputs.

`training_allowed` and `data_collection_done` remain false unless both coverage
and measured diversity pass; failed audits exit 2. `--coverage-only` always leaves
training blocked. Errors first invalidate an older passing report. The gate runs
inference only using the existing upstream frozen encoder weights. No training
pipeline is added or invoked, and no Cuphead predictor checkpoint is expected.
