# Gameplay dataset collection and gate

Current status: recorder and gate implemented and tested; collection is **not
complete**. Live preflight found no visible Cuphead X11 window and no readable
physical gamepad. No human player has confirmed availability. The real dataset
audit reports **0 eligible frames**: attempt 0, menu 0, idle 0; **0 seconds** of
captured playtime; **encoder variance unmeasured**. Training remains blocked.
Evidence: `experiments/gameplay_capture_preflight.json` and
`experiments/gameplay_dataset_gate.json`.

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
`open_gamepad_source`, and `ReplayWriter`. Previously the recorder discarded
pixels and timestamps; it now persists them beside the existing replay files:

- `frames/00000000.png`, etc.: lossless PNG, RGB, bilinear stretch to target size.
- `actions.jsonl`: unchanged frame-indexed normalized action format.
- `frames.jsonl`: matching frame index, capture timestamp, input-poll timestamp,
  source checksum, image path, and all evdev keys/axes at that input sample.
- `meta.json`: type, boss/level, phase, observed outcome, full-attempt flag,
  reviewed-label flag, notes, resolution/rate, timing and image repetition counts.
- `events.json` and `hud.jsonl`: existing replay files; no invented event/HUD truth.

The physical input snapshot preserves menu, weapon-switch and super buttons that
the pruned action representation omits. Inputs are sampled immediately after the
image; timestamps expose this skew. Very short taps between samples can be
missed. Use the physical gamepad being recorded for gameplay and navigation;
keyboard gameplay is not recorded by this backend. Nothing sends input to the game.

Real repeated images are retained and counted, including idle frames. Invalid
indices, backward/nonfinite timestamps, window resize and input-reader errors
abort finalization. Missed deadlines are never filled with fabricated frames.
The audit rejects >5% capture-rate error, gaps >3 nominal frame intervals, >1%
intervals longer than 1.5 frames, or input samples more than one frame late.
These are operational capture tolerances, not a claim of exact frame synchronization.

## Collect separate segments

Open Cuphead and connect a readable physical gamepad. Keep the game window size
fixed. Run the recorder in a terminal, then focus the game; stop with Ctrl+C at the
segment boundary and label the actual outcome after recording. Remove loading or
terminal-switch overhead from a full-attempt claim if its start was missed.
For cleaner boundaries another person can start/stop capture while the player
keeps the game focused. Choose the actual device path with `--device-path` when
more than one gamepad is present.

```bash
# Capture title → save → map → level entry manually as its own segment.
.venv/bin/python scripts/record_session.py --real --segment-type menu \
  --boss forest_follies --phase title_save_map_entry --fps 30 --width 256 --height 256 \
  --max-seconds 300 --label-after

# Repeat for at least five FULL attempts. Keep recording through death/results.
.venv/bin/python scripts/record_session.py --real --segment-type attempt \
  --boss forest_follies --phase whole_attempt --fps 30 --width 256 --height 256 \
  --max-seconds 1200 --label-after

# A short no-input baseline; report IDLE only if no game input occurred.
.venv/bin/python scripts/record_session.py --real --segment-type idle \
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
