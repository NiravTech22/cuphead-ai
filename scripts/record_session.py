#!/usr/bin/env python3
"""Record a labeled human-play segment using the existing capture/replay pipeline.

Real mode stores lossless RGB PNGs and frame-indexed capture/input timestamps
alongside ReplayWriter's actions, events, HUD, and metadata. Equal images are
retained (idle is valid), while invalid indices/timestamps abort the segment.
Synthetic mode is a plumbing test only, explicitly excluded from dataset gates.
Use --label-after to report the observed outcome once recording stops.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.control.action_space import enumerate_actions
from cuphead.control.human_input import ScriptedInputSource
from cuphead.control.keyboard_input import KEYBOARD_BINDINGS, KeyboardFocusLost, X11KeyboardSource
from cuphead.events.schema import EventTrace
from cuphead.memory.replay import ReplayWriter
from cuphead.perception.capture import FrameIntegrityError, IntegrityTracker, SyntheticFrameSource

DEFAULT_ROOT = REPO / "data" / "replays"


def parse_loadout(raw: str) -> dict[str, str]:
    out = {}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise ValueError(f"loadout entry {pair!r} is not key=value")
        key, _, value = pair.partition("=")
        out[key.strip()] = value.strip()
    return out


def default_run_id(boss: str) -> str:
    return f"{boss}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{time.time_ns() % 1_000_000_000:09d}"


def record(*, mode: str, root: Path, run_id: str, boss: str,
           loadout: dict[str, str], game_build: str, fps: float, outcome: str,
           frames: int | None, max_seconds: float | None,
           segment_type: str = "attempt", phase: str = "whole_attempt",
           width: int = 256, height: int = 256, device_path: str | None = None,
           label_after: bool = False, input_source: str = "keyboard") -> Path:
    if mode not in {"real", "synthetic"}:
        raise ValueError("mode must be real or synthetic")
    if input_source not in {"keyboard", "gamepad"}:
        raise ValueError("input_source must be keyboard or gamepad")
    if input_source == "keyboard" and device_path is not None:
        raise ValueError("--device-path applies only to --input-source gamepad")
    if not math.isfinite(fps) or fps <= 0 or min(width, height) < 1:
        raise ValueError("fps and image dimensions must be positive and finite")
    if frames is not None and frames < 1:
        raise ValueError("frames must be positive")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive and finite")
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    if segment_type not in {"attempt", "menu", "idle"}:
        raise ValueError("segment_type must be attempt, menu, or idle")
    if not boss.strip() or not phase.strip():
        raise ValueError("boss and phase labels are required")
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    if mode == "synthetic":
        source = SyntheticFrameSource(fps=fps)
        actions = enumerate_actions()
        budget = frames or 300
        inputs = ScriptedInputSource([actions[i % len(actions)] for i in range(budget)])
    else:
        from cuphead.control.human_input import open_gamepad_source
        from cuphead.perception.window_capture import X11WindowSource

        source = X11WindowSource()
        try:
            inputs = (X11KeyboardSource(window_id=source.window_id) if input_source == "keyboard"
                      else open_gamepad_source(device_path=device_path))
        except BaseException:
            source.close()
            raise
        budget = frames if frames is not None else float("inf")

    tracker = IntegrityTracker(source, strict=(mode == "synthetic"))
    writer = None
    dt = 1 / fps
    count = 0
    first_t = last_t = None
    source_size = None
    max_gap = 0.0
    late_intervals = 0
    stop_reason = "capture_limit"
    try:
        if mode == "real" and input_source == "keyboard":
            print("Focus Cuphead within 60s to start; switching away ends the segment.", flush=True)
            inputs.wait_for_focus()
        started = time.perf_counter()
        deadline = started + max_seconds if max_seconds else float("inf")
        next_tick = started
        print(f"Recording {segment_type} / {boss} / {phase}; Ctrl+C stops capture.", flush=True)
        writer = ReplayWriter(root, run_id=run_id, boss=boss, loadout=loadout,
                              game_build=game_build, fps=fps)
        with (directory / "frames.jsonl").open("x") as times:
            try:
                while count < budget and time.perf_counter() < deadline:
                    frame = tracker.read()
                    action = inputs.poll()
                    action_t = time.perf_counter()
                    raw_input = getattr(inputs, "snapshot", lambda: None)()
                    if frame.index != count or not math.isfinite(frame.t_capture):
                        raise FrameIntegrityError("invalid source index or capture timestamp")
                    if last_t is not None:
                        gap = frame.t_capture - last_t
                        if gap <= 0:
                            raise FrameIntegrityError("capture timestamps did not advance")
                        max_gap = max(max_gap, gap)
                        late_intervals += int(gap > 1.5 * dt)
                    image_path = None
                    if mode == "real":
                        from PIL import Image

                        size = (frame.width, frame.height)
                        if source_size is not None and size != source_size:
                            raise FrameIntegrityError("game window resized during segment")
                        source_size = size
                        image_path = f"frames/{count:08d}.png"
                        pixels = Image.frombytes("RGB", size, frame.payload)
                        pixels.resize((width, height), Image.Resampling.BILINEAR).save(
                            directory / image_path, compress_level=1)
                    writer.log_action(count, action.to_buttons())
                    times.write(json.dumps({
                        "frame": count, "capture_t": frame.t_capture, "action_t": action_t,
                        "checksum": frame.checksum, "image": image_path,
                        "raw_input": raw_input,
                    }) + "\n")
                    times.flush()
                    first_t = frame.t_capture if first_t is None else first_t
                    last_t = frame.t_capture
                    count += 1
                    if mode == "real":
                        next_tick += dt
                        now = time.perf_counter()
                        # No fabricated catch-up frames after a stall.
                        next_tick = max(next_tick, now)
                        time.sleep(max(0, next_tick - now))
            except (KeyboardInterrupt, KeyboardFocusLost) as exc:
                stop_reason = "focus_lost" if isinstance(exc, KeyboardFocusLost) else "interrupt"
                # A partially written final row must not be certified as aligned.
                print(f"\nStopped at {count} complete frames.", file=sys.stderr)
        if count == 0:
            raise RuntimeError("no frames captured")
        # Verify the two logs before finalization, including interrupted writes.
        rows = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
        if [r["frame"] for r in rows] != list(range(count)):
            raise FrameIntegrityError("incomplete timestamp row; segment requires recovery")
        meta = writer.finalize(
            events=EventTrace(boss=boss, events=[], fps=fps), outcome=outcome, frame_count=count,
            extras={
                "schema": "dataset_segment_v1", "mode": mode,
                "segment_type": segment_type, "phase": phase,
                "labels_verified": False, "full_attempt": False,
                "resolution": [width, height], "source_resolution": source_size,
                "pixel_format": "RGB", "resize": "bilinear_stretch",
                "timestamps": "frames.jsonl",
                "action_semantics": "human input polled immediately after capture; normalized action and raw input snapshot",
                "capture_span_seconds": last_t - first_t,
                "captured_seconds": last_t - first_t + dt,
                "max_capture_gap_seconds": max_gap, "late_intervals": late_intervals,
                "repeated_images": tracker.report.duplicates,
                "input_source": input_source if mode == "real" else "scripted",
                "input_device": ("scripted" if mode == "synthetic" else
                                 "x11_keyboard" if input_source == "keyboard" else (device_path or "auto")),
                "keyboard_bindings": KEYBOARD_BINDINGS if mode == "real" and input_source == "keyboard" else None,
                "stop_reason": stop_reason,
            })
    except BaseException:
        if writer is not None:
            writer.abandon()
        raise
    finally:
        tracker.close()
        inputs.close()
    print(f"Recorded {meta.frame_count} frames -> {directory}", flush=True)
    if label_after:
        from cuphead.evaluation.dataset_gate import OUTCOMES, label_segment

        allowed = OUTCOMES[segment_type]
        while True:
            outcome = input(f"Observed outcome ({'/'.join(sorted(allowed))}): ").strip().upper()
            if outcome in allowed:
                break
        full = segment_type == "attempt" and input(
            "Captured from attempt start through its ending? [yes/no]: ").strip().lower() == "yes"
        notes = input("Phase/coverage notes: ").strip()
        label_segment(directory, outcome=outcome, full_attempt=full, notes=notes)
    return directory


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--synthetic", action="store_const", dest="mode", const="synthetic")
    modes.add_argument("--real", action="store_const", dest="mode", const="real")
    ap.add_argument("--boss", default="forest_follies")
    ap.add_argument("--loadout", default="")
    ap.add_argument("--game-build", default="unknown")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--segment-type", choices=["attempt", "menu", "idle"], default="attempt")
    ap.add_argument("--phase", default="whole_attempt")
    ap.add_argument("--input-source", choices=["keyboard", "gamepad"], default="keyboard",
                    help="Human controls to sample (default: keyboard, Cuphead default bindings)")
    ap.add_argument("--device-path", help="Physical evdev device for --input-source gamepad")
    ap.add_argument("--label-after", action="store_true")
    ap.add_argument("--outcome", default="INCOMPLETE", choices=["KNOCKOUT", "DEATH", "INCOMPLETE", "IDLE", "NAVIGATION"],
                    help="Provisional until reviewed after recording; use --label-after")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--run-id")
    ap.add_argument("--frames", type=int)
    ap.add_argument("--max-seconds", type=float)
    args = ap.parse_args()
    if args.mode == "real" and args.frames is None and args.max_seconds is None:
        ap.error("--real needs --max-seconds or --frames; Ctrl+C stops early")
    if args.input_source == "keyboard" and args.device_path is not None:
        ap.error("--device-path requires --input-source gamepad")
    args.run_id = args.run_id or default_run_id(args.boss)
    args.loadout = parse_loadout(args.loadout)
    record(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
