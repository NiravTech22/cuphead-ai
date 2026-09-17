#!/usr/bin/env python3
"""Launch Cuphead after creating a persistent Windows or Linux Xbox controller.

Usage:

    python3 scripts/launch_with_vgamepad.py -- wine /path/to/Cuphead.exe

The process keeps the virtual pad alive for the entire game session. Do not
start Cuphead separately first: Wine commonly enumerates XInput only at launch.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.control.actuator import open_vgamepad_actuator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--keep-alive", action="store_true",
        help="keep the pad alive after a launcher command exits; stop it with Ctrl+C",
    )
    ap.add_argument("command", nargs=argparse.REMAINDER, help="game command; put -- before it")
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("provide a game command after --, for example: -- wine /path/to/Cuphead.exe")

    actuator = open_vgamepad_actuator()
    print("Virtual Xbox 360 controller is ready; starting Cuphead now.", flush=True)
    try:
        child = subprocess.Popen(command)
    except OSError as exc:
        actuator.close()
        print(f"FAIL: could not start {command[0]!r}: {exc}", file=sys.stderr)
        return 2

    try:
        if args.keep_alive:
            print("Keeping the virtual pad alive. Press Ctrl+C when the game exits.", flush=True)
            while True:
                time.sleep(0.25)
        return child.wait()
    except KeyboardInterrupt:
        print("Stopping virtual controller.", file=sys.stderr)
        if child.poll() is None:
            child.terminate()
        return 130
    finally:
        actuator.close()


if __name__ == "__main__":
    raise SystemExit(main())
