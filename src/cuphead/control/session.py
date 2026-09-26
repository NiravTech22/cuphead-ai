"""Own the controller for the entire lifetime of a directly launched game."""
from __future__ import annotations

import csv
import platform
import subprocess
import time
from pathlib import Path

from .actuator import open_vgamepad_actuator


class GamepadSession:
    """Stop automation without unplugging the controller from a running game.

    On Windows a direct launch is required: a new pad attached to an existing
    game can become player two, and a launcher process cannot track game exit.
    Cleanup deliberately waits for the user to close the game normally.
    """

    def __init__(self, command=None, *, launch_kwargs=None, announce=print):
        self.command = list(command) if command else None
        self.launch_kwargs = launch_kwargs or {}
        self.announce = announce
        self.actuator = None
        self.child = None

    def __enter__(self):
        if platform.system() == "Windows":
            if not self.command or Path(self.command[0]).name.lower() != "cuphead.exe":
                raise ValueError(
                    "Windows gamepad sessions require a direct Cuphead.exe launch. "
                    "Close Cuphead first, then use this script's --launch option "
                    "(or the launcher script's command after --)."
                )
            processes = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq Cuphead.exe", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, check=True,
            )
            if any(row and row[0].lower() == "cuphead.exe"
                   for row in csv.reader(processes.stdout.splitlines())):
                raise RuntimeError(
                    "Cuphead is already running. Close it normally before starting "
                    "a new controller session; attaching another pad can select player two."
                )
        try:
            self.actuator = open_vgamepad_actuator()
            if self.command:
                self.child = subprocess.Popen(self.command, **self.launch_kwargs)
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.actuator is None:
            return
        # Retry neutral on each iteration if a transient device error occurred.
        # Keep the device connected even when input release failed.
        release_error = None
        try:
            self.actuator.neutral()
        except BaseException as exc:
            release_error = exc
        if self.child is not None and self.child.poll() is None:
            self.announce(
                "Automation stopped. Keeping the controller connected until Cuphead "
                "closes. Close the game normally to finish this session."
            )
            while True:
                try:
                    if self.child.poll() is not None:
                        break
                    if release_error is not None:
                        try:
                            self.actuator.neutral()
                            release_error = None
                        except Exception:
                            pass
                    time.sleep(0.25)
                except KeyboardInterrupt:
                    self.announce("Controller still connected; close Cuphead to finish safely.")
        actuator, self.actuator = self.actuator, None
        actuator.close()
        if release_error is not None:
            raise release_error

    def __exit__(self, exc_type, exc, traceback):
        self.close()
