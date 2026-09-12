"""Timed controller macros with explicit button release even on capture failure."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ControlInput:
    name: str
    stick_x: float = 0.0
    stick_y: float = 0.0
    a: bool = False
    b: bool = False
    x: bool = False
    y: bool = False
    rb: bool = False
    start: bool = False
    hold_frames: int = 4
    release_frames: int = 2

    def __post_init__(self):
        if not self.name or not -1 <= self.stick_x <= 1 or not -1 <= self.stick_y <= 1:
            raise ValueError("named input with valid stick axes required")
        if not isinstance(self.hold_frames, int) or not isinstance(
            self.release_frames, int
        ):
            raise TypeError("frame durations must be integers")
        if self.hold_frames < 1 or self.release_frames < 1:
            raise ValueError("hold and release durations must be positive")

    @property
    def key(self) -> str:
        # Changing duration or a binding must never silently reuse another action's memory.
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


NAVIGATION_INPUTS = (
    ControlInput("confirm", a=True),
    ControlInput("start", start=True),
    ControlInput("right", stick_x=1, hold_frames=12),
    ControlInput("down", stick_y=-1, hold_frames=12),
    ControlInput("left", stick_x=-1, hold_frames=12),
    ControlInput("up", stick_y=1, hold_frames=12),
)
OVERWORLD_INPUTS = NAVIGATION_INPUTS + (
    ControlInput("northwest", stick_x=-1, stick_y=1, hold_frames=8),
    ControlInput("northeast", stick_x=1, stick_y=1, hold_frames=8),
    ControlInput("southwest", stick_x=-1, stick_y=-1, hold_frames=8),
    ControlInput("southeast", stick_x=1, stick_y=-1, hold_frames=8),
)
COMBAT_INPUTS = (
    ControlInput("shoot", x=True),
    ControlInput("jump_shoot", a=True, x=True, hold_frames=12),
    ControlInput("right_shoot", stick_x=1, x=True),
    ControlInput("left_shoot", stick_x=-1, x=True),
    ControlInput("duck_shoot", stick_y=-1, x=True),
    ControlInput("dash_right", stick_x=1, b=True),
    ControlInput("dash_left", stick_x=-1, b=True),
)


class TimedExecutor:
    def __init__(
        self,
        actuator,
        *,
        fps: float = 60,
        clock: Callable = time.perf_counter,
        sleep: Callable = time.sleep,
    ):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("positive finite fps required")
        self.actuator, self.fps, self.clock, self.sleep = actuator, fps, clock, sleep

    def execute(self, action: ControlInput, sample: Callable | None = None):
        start = self.clock()
        observations = []
        try:
            self.actuator.send_buttons(action)
            for index in range(action.hold_frames):
                remaining = start + (index + 1) / self.fps - self.clock()
                if remaining > 0:
                    self.sleep(remaining)
                if sample:
                    observations.append(sample())
        finally:
            self.actuator.neutral()
        self.sleep(action.release_frames / self.fps)
        if sample:
            observations.append(sample())
        return observations
