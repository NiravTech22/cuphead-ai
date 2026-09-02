"""Reading a human's real input during demonstration capture.

This is deliberately a separate concern from ``actuator.py``. Actuation
*drives* the game (the agent's virtual pad); this module *reads* the human's
real controller so a demonstration session can log what they actually did,
frame-indexed, for imitation learning -- the recommended bootstrap before any
RL or planning runs, per ``docs/SPEEDRUN_PLAN.md``.

Raw device state is normalized to the pruned 56-action space through
``Action.from_raw`` so recorded demonstrations are always legal actions the
planner could itself have chosen -- an imitation target outside that space
is not learnable by a policy that only ever outputs legal actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Protocol, Sequence

from .action_space import Action


class HumanInputSource(Protocol):
    """One polled sample of held input, already normalized to ``Action``."""

    def poll(self) -> Action: ...

    def close(self) -> None: ...


class ScriptedInputSource:
    """Replays a fixed, pre-normalized action sequence.

    Used by tests and by ``record_session.py --synthetic`` to exercise the
    full recording pipeline without a real controller attached. Raises when
    exhausted rather than looping silently -- a recorder that keeps reading
    past the scripted sequence and gets stale data would corrupt a replay
    the exact same way a duplicated capture frame does.
    """

    def __init__(self, actions: Sequence[Action]) -> None:
        if not actions:
            raise ValueError("ScriptedInputSource needs at least one action")
        self._actions = list(actions)
        self._i = 0
        self._closed = False

    def poll(self) -> Action:
        if self._closed:
            raise RuntimeError("poll() on a closed HumanInputSource")
        if self._i >= len(self._actions):
            raise StopIteration("ScriptedInputSource exhausted its scripted actions")
        action = self._actions[self._i]
        self._i += 1
        return action

    def close(self) -> None:
        self._closed = True


def open_gamepad_source(*, device_path: str | None = None) -> HumanInputSource:
    """Real controller reading via ``evdev``, imported lazily.

    Maintains live button/axis state from device events and normalizes it
    through ``Action.from_raw`` on every ``poll()``. Requires read access to
    the input device (usually the ``input`` group on Linux) and is not
    exercised by the test suite for the same reason the real capture and
    actuation backends aren't: no hardware in CI. Its correctness rests on
    ``Action.from_raw`` being fully covered by stdlib tests, which it is.
    """
    try:
        import evdev  # type: ignore[import-untyped]
        from evdev import ecodes
    except ImportError as exc:
        raise RuntimeError(
            "evdev is not installed. Run `pip install -r requirements.txt` on the "
            "machine actually running Cuphead before recording human input."
        ) from exc

    def _pick_device() -> "evdev.InputDevice":
        if device_path:
            return evdev.InputDevice(device_path)
        for path in evdev.list_devices():
            dev = evdev.InputDevice(path)
            caps = dev.capabilities().get(ecodes.EV_ABS, [])
            if any(code in (ecodes.ABS_X, ecodes.ABS_HAT0X) for code, _ in caps):
                return dev
        raise RuntimeError("no gamepad-like input device found; pass device_path explicitly")

    class _EvdevInputSource:
        def __init__(self) -> None:
            self._dev = _pick_device()
            info = self._dev.absinfo(ecodes.ABS_X)
            self._x_range = (info.min, info.max)
            info = self._dev.absinfo(ecodes.ABS_Y)
            self._y_range = (info.min, info.max)
            self._raw = {
                "stick_x": 0.0,
                "stick_y": 0.0,
                "jump": False,
                "duck": False,
                "dash": False,
                "shoot": False,
                "lock": False,
            }

        def _normalize_axis(self, value: int, lo: int, hi: int) -> float:
            span = (hi - lo) or 1
            return (2.0 * (value - lo) / span) - 1.0

        def poll(self) -> Action:
            # Drain pending events without blocking; the last value for each
            # axis/button wins, which is the correct semantics for a held
            # controller state sampled once per decision window.
            while True:
                event = self._dev.read_one()
                if event is None:
                    break
                if event.type == ecodes.EV_ABS and event.code == ecodes.ABS_X:
                    self._raw["stick_x"] = self._normalize_axis(event.value, *self._x_range)
                elif event.type == ecodes.EV_ABS and event.code == ecodes.ABS_Y:
                    self._raw["stick_y"] = -self._normalize_axis(event.value, *self._y_range)
                elif event.type == ecodes.EV_KEY and event.code == ecodes.BTN_SOUTH:
                    self._raw["jump"] = bool(event.value)
                elif event.type == ecodes.EV_KEY and event.code == ecodes.BTN_WEST:
                    self._raw["shoot"] = bool(event.value)
                elif event.type == ecodes.EV_KEY and event.code == ecodes.BTN_EAST:
                    self._raw["dash"] = bool(event.value)
                elif event.type == ecodes.EV_KEY and event.code == ecodes.BTN_TR:
                    self._raw["lock"] = bool(event.value)
            return Action.from_raw(**self._raw)

        def close(self) -> None:
            self._dev.close()

    return _EvdevInputSource()
