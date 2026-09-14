> Status correction (2026-09-13): navigation title → Forest Follies is complete.
> This zero-training v1 uses a frozen encoder and a nearest-neighbor memory bank;
> no Cuphead predictor checkpoint is missing or required. The checkpoint/training
> prerequisites below were mistaken and are superseded. The latest active task
> is gameplay dataset collection and its pre-training diversity gate only; see
> [GAMEPLAY_DATASET.md](GAMEPLAY_DATASET.md). No training or combat controller is
> authorized by that task. The Component 5 placeholder was superseded before implementation.

# V1 execution record — 2026-09-13

The requested title-to-first-level-clear run has **not** been achieved. There
have been four navigation preflights and no combat attempts in this session.

## Training state

The local checkout and model-file search under `/home/nirav` found no Cuphead
predictor training checkpoint, saved T5 adapters, or earlier Phase 3 sanity-check
artifact. The only project `.pt` file is Meta's upstream pretrained encoder
checkpoint, with SHA-256
`848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`.
Its epoch 40 and loss 0.5920411388079325 belong to upstream pretraining. They are
not Cuphead training progress. No training was restarted or resumed.

Strict encoder loading succeeded. Across eight calibration screenshots, mean
per-dimension population variance of normalized encoder outputs was
0.00012801017021758887; mean latent norm was 1.0000000051652387. This probes
mostly menu imagery and is not a combat collapse check at a resumed training step.

The saved world model is a kNN transition bank. Freezing its first 16 navigation
rows and evaluating the seven subsequent transitions from preflight 02 gave:

| Metric | Prediction | No-change | Random unit vector |
|---|---:|---:|---:|
| Mean cosine similarity | 0.99685731 | 0.96236067 | -0.00888584 |
| Mean latent MSE | 0.0000165894 | 0.0001960382 | — |

All seven transitions had neighbor support. This small navigation-only result
does not pass a combat-training gate. The audited banks had zero hit and zero
terminal examples, and no combat scopes. The T5 implementation remains untrained.

Reproduce the diagnostic with `.venv/bin/python scripts/audit_v1_state.py`.
Its fixed split refers specifically to preflight 02. Full output is in
`experiments/v1_state_audit.json`.

## Action selection and preflights

The existing one-step planner scores predicted reward plus reduction in distance
to a goal latent, minus 5 times predicted hit probability and 0.25 times
uncertainty. Navigation prefers the calibrated route action when its predicted
consequence passes the existing checks. This is not a learned combat controller.
Two known synthetic scenarios confirmed that predicted hits favor ducking and
predicted goal progress overrides action names. Neither result proves gameplay
competence.

| Preflight | Actual result | Recording |
|---|---|---|
| 01 | Step budget exhausted after title → Start | Last frame only |
| 02 | Title → playable Forest Follies, virtual gamepad, no manual input | Last frame only |
| 03 | Cold-start exploration selected Start on overworld; pause menu caused unknown-screen stop | `experiments/v1_gamepad_recorded_03.avi` |
| 04 | Title → playable Forest Follies after excluding pause from overworld candidates | `experiments/v1_gamepad_recorded_04.avi` |

Every run has a JSON summary and JSONL trace. `experiments/v1_attempts.jsonl`
links all four results. All explicitly retain `cuphead_victory_verified: false`.

Preflight 04 video was fully decoded: 425 frames, 20 fps, 21.25 seconds,
966×572 pixels, 425 strictly increasing capture timestamps, maximum capture gap
64.38 ms. Visual inspection confirmed the title screen and the playable level.
The recorder runs independently of inference and writes actual capture times
to `.frames.jsonl`; AVI playback uses a fixed rate. The recording contains no
audio. See `experiments/v1_gamepad_recorded_04.verification.json`.

The live runner now accepts `--record-video path.avi`; its traces include monotonic
timestamps. Subsequent transition logs also include chosen prediction score, hit
probability and uncertainty. A regression test covers exhausting the overworld
exploration set without pausing. Validation: 224 tests run, 8 skipped; no failures.

## Remaining prerequisite and work

To obey the instruction to resume rather than restart training, the Cuphead
predictor checkpoint and its training/evaluation data must be located. The
checkpoint location was requested during this session but has not been supplied.
There is no resumed step, resumed loss, or combat similarity measurement to report.

Once that state is available: restore model and optimizer, measure the requested
held-out baselines and encoder variance, connect the trained predictor to combat
candidate rollouts, calibrate combat/death/victory detection, then record repeated
complete attempts. Navigation arrival is not level completion.
