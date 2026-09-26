# Native Windows collection pilot

`scripts/windows_capture_pilot.py` collects real RGB frames and exact held
vgamepad reports. It uses an untrained, seeded forward/jump/shoot exploration
policy. It does not update model weights or the frozen-encoder memory bank.

The pilot requires `--launch` pointing directly to Cuphead.exe. Close any
existing Cuphead session first. Attaching a new controller to an existing
keyboard session can assign it to player two, so this is now rejected.
Inspect both the character and HUD before accepting a run as single-player data.

```powershell
.venv/Scripts/python.exe scripts/windows_capture_pilot.py --output data/recordings/unique_run --seconds 300 --launch "C:/GOG Games/Cuphead/Cuphead.exe"
```

The output directory must be new. When collection stops, the pilot releases
input, closes capture and writes its summary, then keeps the controller
connected until Cuphead closes. Keep its terminal open and close the game
normally to finish. Repeated Ctrl+C during this wait does not disconnect the
controller. Setup waits at most 15 minutes; collection ends after the configured
wall-clock budget. See [controller recovery](CONTROLLER_RECOVERY.md).

During setup, inspect `latest.png`, then write a unique command ID to
`command.json` in the output directory:

```json
{"id": 1, "buttons": {"a": true}, "hold_seconds": 0.12}
```

After visually confirming a playable level, use `{"id":2,"mode":"explore"}`.
Stop with `{"id":3,"mode":"stop"}`. Do not issue setup commands without a
fresh image. The pilot waits for the exact `Cuphead` foreground title and
does not reactivate a background game.

`--retry-reference path/to/verified-death-screen.png` enables automatic retries
after three consecutive matches to the Forest Follies death-card artwork.
It waits four seconds after the confirm pulse before exploration. This is
calibrated to that card and is not a general death/victory classifier.
It also starts collection from a verified matching death card automatically.

Files:

- `frames/*.png`: 256 by 256 RGB observations.
- `frames.jsonl`: contiguous indices, capture times, held button reports,
  report-publication times, image paths, checksums and death-template matches.
- `commands.jsonl`: setup, retry and exploration report publications.
- `summary.json`: rate measurements, termination and explicit untrained status.

This is an experimental raw collection format, separate from human replays.
Menu/loading frames and exploratory attempts need segmentation and review
before conversion into a training dataset. Time limits do not imply a clear.

On September 17, 2026, native input and capture were exercised with 1,628
verified images at 19.60 FPS. That pilot was excluded from single-player
training because player-two assignment was observed. See
`experiments/windows_capture_pilot_20260917.json`. The target 30 FPS was not
met; subsequent runs use lower render resolution and cheaper preview writes.
