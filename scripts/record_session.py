#!/usr/bin/env python3
"""Record one attempt: synchronized frames and frame-indexed human input.

This is what task ``harness-human-demo-recorder`` closes, and it is the tool
``docs/SPEEDRUN_PLAN.md`` section 2.4 and section 4.4 assume exists before
any world-model training starts: human demonstrations are the seed data for
the imitation-learning bootstrap, and they need the exact same frame-action
alignment guarantee every other replay does.

Two modes, sharing one pipeline:

    python3 scripts/record_session.py --synthetic --frames 300
        Self-test with no game and no hardware: a deterministic synthetic
        capture source and a scripted input sequence, cycling through the
        full legal action set. Proves the recording pipeline -- capture,
        integrity checking, action normalization, replay writing -- works
        end to end, and is what CI exercises. Not a real demonstration.

    python3 scripts/record_session.py --real --boss goopy_le_grande \\
        --loadout weapon=peashooter,charm=smoke_bomb --max-seconds 120
        Records real gameplay: the actual screen via mss and the actual
        controller via evdev. Run this on the machine with Cuphead open and
        the controller in hand. Ctrl+C ends the session early and still
        finalizes a valid, if shorter, replay.

A capture-integrity violation (a dropped or duplicated frame -- see
``perception.capture``) aborts the session and abandons the replay rather
than finalizing a stream nothing downstream can trust. Fix the capture setup
and record again; a corrupted replay is worse than no replay.

HUD ground truth and the event trace are still placeholders here -- they
arrive with tasks ``perception-hud-templates`` and ``perception-event-
detector``. Until then, replays this tool writes carry frames and actions
(the imitation-learning bootstrap needs nothing more) with an empty event
trace and whatever ``--outcome`` the human recording the session reports.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.control.action_space import enumerate_actions  # noqa: E402
from cuphead.control.human_input import ScriptedInputSource  # noqa: E402
from cuphead.events.schema import EventTrace  # noqa: E402
from cuphead.memory.replay import ReplayAlignmentError, ReplayWriter  # noqa: E402
from cuphead.perception.capture import (  # noqa: E402
    FrameIntegrityError,
    IntegrityTracker,
    SyntheticFrameSource,
)

DEFAULT_ROOT = REPO / "data" / "replays"


def parse_loadout(raw: str) -> dict[str, str]:
    """``"weapon=peashooter,charm=smoke_bomb"`` -> ``{"weapon": ..., "charm": ...}``"""
    if not raw:
        return {}
    out = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"loadout entry {pair!r} is not key=value")
        key, _, value = pair.partition("=")
        out[key.strip()] = value.strip()
    return out


def default_run_id(boss: str) -> str:
    return f"{boss}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def record(
    *,
    mode: str,
    root: Path,
    run_id: str,
    boss: str,
    loadout: dict[str, str],
    game_build: str,
    fps: float,
    outcome: str,
    frames: int | None,
    max_seconds: float | None,
) -> Path:
    if mode == "synthetic":
        capture_source = SyntheticFrameSource(fps=fps)
        actions = enumerate_actions()
        n = frames or 300
        script = [actions[i % len(actions)] for i in range(n)]
        input_source = ScriptedInputSource(script)
        frame_budget = n
    else:
        from cuphead.control.human_input import open_gamepad_source
        from cuphead.perception.capture import open_screen_source

        capture_source = open_screen_source()
        input_source = open_gamepad_source()
        frame_budget = frames if frames is not None else float("inf")

    tracker = IntegrityTracker(capture_source, strict=True)
    writer = ReplayWriter(root, run_id=run_id, boss=boss, loadout=loadout, game_build=game_build, fps=fps)

    dt = 1.0 / fps
    frame_index = 0
    deadline = (time.monotonic() + max_seconds) if max_seconds else None

    try:
        while frame_index < frame_budget:
            if deadline is not None and time.monotonic() >= deadline:
                break
            loop_start = time.monotonic()

            tracker.read()  # advances and integrity-checks the capture stream
            action = input_source.poll()
            writer.log_action(frame_index, action.to_buttons())
            frame_index += 1

            if mode == "real":
                elapsed = time.monotonic() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print(f"\ninterrupted after {frame_index} frames; finalizing what was captured", file=sys.stderr)
    except StopIteration:
        print(f"\nscripted input exhausted after {frame_index} frames", file=sys.stderr)
    except FrameIntegrityError as exc:
        writer.abandon()
        print(f"FAIL: capture integrity violated: {exc}", file=sys.stderr)
        print("The replay was NOT finalized. Fix the capture setup and record again.", file=sys.stderr)
        raise SystemExit(2)
    finally:
        tracker.close()
        input_source.close()

    try:
        meta = writer.finalize(
            events=EventTrace(boss=boss, events=[], fps=fps),
            outcome=outcome,
            frame_count=frame_index,
        )
    except ReplayAlignmentError as exc:
        # Should be unreachable given the loop above logs exactly one action
        # per frame in order, but this is the invariant the whole module
        # exists to protect, so it is checked here too rather than trusted.
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)

    print(f"recorded {meta.frame_count} frames -> {(root / run_id).relative_to(REPO) if root.is_relative_to(REPO) else root / run_id}")
    print(f"  boss={meta.boss} loadout={meta.loadout} outcome={meta.outcome} fps={meta.fps}")
    print(f"  integrity: {tracker.report.frames_seen} frames, "
          f"{tracker.report.duplicates} duplicate(s), {tracker.report.drops} drop(s)")
    return root / run_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--synthetic", action="store_const", dest="mode", const="synthetic")
    mode_group.add_argument("--real", action="store_const", dest="mode", const="real")
    ap.add_argument("--boss", default="goopy_le_grande")
    ap.add_argument("--loadout", default="", help="weapon=peashooter,charm=smoke_bomb")
    ap.add_argument("--game-build", default="unknown")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--outcome", default="INCOMPLETE", choices=["KNOCKOUT", "DEATH", "INCOMPLETE"],
                     help="Automatic outcome detection needs perception-event-detector; "
                          "for now, report what actually happened.")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--frames", type=int, default=None,
                     help="Frame budget. Synthetic mode defaults to 300; real mode is "
                          "unbounded unless --max-seconds is also given.")
    ap.add_argument("--max-seconds", type=float, default=None,
                     help="Real mode only: stop recording after this many seconds.")
    args = ap.parse_args()

    if args.mode == "real" and args.frames is None and args.max_seconds is None:
        ap.error("--real needs --max-seconds or --frames (or Ctrl+C to stop manually)")

    run_id = args.run_id or default_run_id(args.boss)
    record(
        mode=args.mode,
        root=args.root,
        run_id=run_id,
        boss=args.boss,
        loadout=parse_loadout(args.loadout),
        game_build=args.game_build,
        fps=args.fps,
        outcome=args.outcome,
        frames=args.frames,
        max_seconds=args.max_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
