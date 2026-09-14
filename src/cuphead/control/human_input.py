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

import platform
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Protocol, Sequence

from .action_space import Action


def _normalize_axis(value: int, lo: int, hi: int) -> float:
    """Map a raw axis reading in ``[lo, hi]`` onto ``[-1.0, 1.0]``."""
    span = (hi - lo) or 1
    return (2.0 * (value - lo) / span) - 1.0


def _blank_raw_state() -> dict:
    """The raw device state both backends fill in and hand to ``Action.from_raw``.

    Shared deliberately: the two backends must produce demonstrations that are
    interchangeable in the replay buffer, so they populate the *same* keys with
    the same meaning (stick in [-1, 1], y positive = up) and differ only in how
    they talk to the OS.
    """
    return {
        "stick_x": 0.0,
        "stick_y": 0.0,
        "jump": False,
        "duck": False,
        "dash": False,
        "shoot": False,
        "lock": False,
    }


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


def open_gamepad_source(
    *, device_path: str | None = None, backend: str | None = None
) -> HumanInputSource:
    """Open the real controller for the host platform.

    Dispatches on ``platform.system()``: ``evdev`` on Linux, the ``inputs``
    package on Windows. Both backends are lazily imported inside their own
    factory, so importing this module costs nothing and needs neither package
    on the platform that doesn't use it -- ``requirements.txt`` markers make
    each one install-only on its own OS.

    Whatever the backend, ``poll()`` returns an ``Action`` produced by
    ``Action.from_raw``, so a demonstration recorded on Windows is the same
    kind of imitation target as one recorded on Linux and the two can share a
    replay buffer without a per-platform decoder.

    ``backend`` overrides the platform sniff; it exists for diagnostics
    (forcing a specific path on a machine that has both) and for tests.
    """
    backend = backend or ("windows" if platform.system() == "Windows" else "linux")
    if backend == "windows":
        return _open_inputs_gamepad_source()
    if backend == "linux":
        return _open_evdev_gamepad_source(device_path=device_path)
    raise ValueError(f"unknown human-input backend {backend!r}; expected 'linux' or 'windows'")


def _open_evdev_gamepad_source(*, device_path: str | None = None) -> HumanInputSource:
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
            caps = dev.capabilities(absinfo=False)
            if ecodes.BTN_GAMEPAD in caps.get(ecodes.EV_KEY, []):
                return dev
            dev.close()
        raise RuntimeError("no gamepad-like input device found; pass device_path explicitly")

    class _EvdevInputSource:
        def __init__(self) -> None:
            self._dev = _pick_device()
            info = self._dev.absinfo(ecodes.ABS_X)
            self._x_range = (info.min, info.max) if info is not None else (-1, 1)
            info = self._dev.absinfo(ecodes.ABS_Y)
            self._y_range = (info.min, info.max) if info is not None else (-1, 1)
            self._raw = _blank_raw_state()
            # Preserve all physical controls, including menu/weapon/super inputs
            # that the pruned combat Action representation intentionally omits.
            self._physical = {"keys": {}, "axes": {}}
            self._physical["keys"] = {str(k): 1 for k in self._dev.active_keys()}
            for code, info in self._dev.capabilities(absinfo=True).get(ecodes.EV_ABS, []):
                self._physical["axes"][str(code)] = info.value

        def snapshot(self) -> dict:
            return {"backend": "evdev", "device": self._dev.path,
                    "keys": dict(self._physical["keys"]),
                    "axes": dict(self._physical["axes"])}

        def poll(self) -> Action:
            # Drain pending events without blocking; the last value for each
            # axis/button wins, which is the correct semantics for a held
            # controller state sampled once per decision window.
            while True:
                event = self._dev.read_one()
                if event is None:
                    break
                if event.type == ecodes.EV_SYN and event.code == ecodes.SYN_DROPPED:
                    raise RuntimeError("evdev input events dropped; stop recording to avoid stale labels")
                if event.type == ecodes.EV_KEY:
                    self._physical["keys"][str(event.code)] = int(bool(event.value))
                elif event.type == ecodes.EV_ABS:
                    self._physical["axes"][str(event.code)] = event.value
            axes, keys = self._physical["axes"], self._physical["keys"]
            self._raw["stick_x"] = _normalize_axis(axes.get(str(ecodes.ABS_X), 0), *self._x_range)
            self._raw["stick_y"] = -_normalize_axis(axes.get(str(ecodes.ABS_Y), 0), *self._y_range)
            hat_x = axes.get(str(ecodes.ABS_HAT0X), 0)
            hat_y = axes.get(str(ecodes.ABS_HAT0Y), 0)
            if hat_x:
                self._raw["stick_x"] = float(hat_x)
            if hat_y:
                self._raw["stick_y"] = -float(hat_y)
            for code, flag in ((ecodes.BTN_SOUTH, "jump"), (ecodes.BTN_WEST, "shoot"),
                               (ecodes.BTN_EAST, "dash"), (ecodes.BTN_TR, "lock")):
                self._raw[flag] = bool(keys.get(str(code), 0))
            self._raw["duck"] = self._raw["stick_y"] < -0.35
            return Action.from_raw(**self._raw)

        def close(self) -> None:
            self._dev.close()

    return _EvdevInputSource()


# XInput reports both thumbstick axes as signed 16-bit. Unlike evdev, its Y is
# already positive-up, so the Windows backend does *not* negate it -- both
# backends hand `Action.from_raw` the same "y positive = up" convention.
_XINPUT_AXIS_MIN = -32768
_XINPUT_AXIS_MAX = 32767

# Button map, kept deliberately identical to the evdev backend above so the two
# platforms record the same physical button as the same raw flag. A divergence
# here would be invisible in the loss curve and would quietly make Windows
# demonstrations mislabelled relative to Linux ones in a shared buffer.
_INPUTS_BUTTON_MAP = {
    "BTN_SOUTH": "jump",
    "BTN_WEST": "shoot",
    "BTN_EAST": "dash",
    "BTN_TR": "lock",
}


class _InputsInputSource:
    """Real controller reading via the ``inputs`` package (Windows/XInput).

    ``inputs``' ``gamepad.read()`` *blocks* until an event arrives, which a
    fixed-rate recorder cannot tolerate: a blocking read inside the sampling
    loop stalls the frame clock and slides actions onto later frames than the
    ones they were made on. That is precisely the frame-action misalignment
    that silently destroys world-model training. So the blocking read lives on
    a daemon thread that keeps a live copy of held state, and ``poll()`` only
    ever snapshots that copy -- non-blocking, same contract as the evdev path.

    The device is injected rather than discovered so this class is testable
    without hardware or the ``inputs`` package; ``_open_inputs_gamepad_source``
    does the lazy import and discovery.
    """

    def __init__(self, device: Any, *, start_reader: bool = True) -> None:
        self._device = device
        self._raw = _blank_raw_state()
        self._lock = threading.Lock()
        self._closed = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        if start_reader:
            self._thread = threading.Thread(
                target=self._read_loop, name="human-input-inputs", daemon=True
            )
            self._thread.start()

    def _read_loop(self) -> None:
        while not self._closed:
            try:
                events = self._device.read()
            except BaseException as exc:  # unplugged, driver error, shutdown
                if not self._closed:
                    self._error = exc
                return
            for event in events or ():
                self._apply(event)

    def _apply(self, event: Any) -> None:
        """Fold one ``inputs`` event into the live raw state.

        ``inputs`` events carry ``ev_type`` ("Absolute"/"Key"/"Sync"), a string
        ``code`` and an int ``state``. Anything unrecognised (triggers, sync,
        rumble acks) is ignored rather than raising -- an unmapped button must
        not kill a recording session mid-run.
        """
        ev_type = getattr(event, "ev_type", None)
        code = getattr(event, "code", None)
        state = getattr(event, "state", 0)

        with self._lock:
            if ev_type == "Absolute":
                if code == "ABS_X":
                    self._raw["stick_x"] = _normalize_axis(
                        state, _XINPUT_AXIS_MIN, _XINPUT_AXIS_MAX
                    )
                elif code == "ABS_Y":
                    self._raw["stick_y"] = _normalize_axis(
                        state, _XINPUT_AXIS_MIN, _XINPUT_AXIS_MAX
                    )
                elif code == "ABS_HAT0X":
                    # D-pad is digital: drive the stick to the rail directly.
                    self._raw["stick_x"] = float(max(-1, min(1, state)))
                elif code == "ABS_HAT0Y":
                    # evdev hats are positive-down; inputs mirrors that, and the
                    # raw contract is positive-up, hence the negation.
                    self._raw["stick_y"] = -float(max(-1, min(1, state)))
            elif ev_type == "Key":
                flag = _INPUTS_BUTTON_MAP.get(code or "")
                if flag is not None:
                    self._raw[flag] = bool(state)
                elif code == "BTN_DPAD_LEFT":
                    self._raw["stick_x"] = -1.0 if state else 0.0
                elif code == "BTN_DPAD_RIGHT":
                    self._raw["stick_x"] = 1.0 if state else 0.0
                elif code == "BTN_DPAD_UP":
                    self._raw["stick_y"] = 1.0 if state else 0.0
                elif code == "BTN_DPAD_DOWN":
                    self._raw["stick_y"] = -1.0 if state else 0.0

    def poll(self) -> Action:
        if self._closed:
            raise RuntimeError("poll() on a closed HumanInputSource")
        if self._error is not None:
            # Fail loudly. A dead reader thread would otherwise keep returning
            # the last held state forever, writing a long tail of confidently
            # wrong actions into the replay -- worse than no data at all.
            raise RuntimeError(
                "the Windows gamepad reader stopped; the recorded actions after this "
                f"point would be stale: {self._error!r}"
            ) from self._error
        with self._lock:
            snapshot = dict(self._raw)
        return Action.from_raw(**snapshot)

    def close(self) -> None:
        self._closed = True
        closer = getattr(self._device, "close", None)
        if callable(closer):
            closer()


def _open_inputs_gamepad_source() -> HumanInputSource:
    """Real controller reading via ``inputs``, imported lazily.

    Like the evdev backend, the hardware path is not exercised in CI (no
    controller, and the package is Windows-only). What *is* exercised is
    ``_InputsInputSource``'s event folding and normalization, which is where
    the mapping bugs would actually live.
    """
    try:
        import inputs  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "the `inputs` package is not installed. Run `pip install -r requirements.txt` "
            "on the Windows machine actually running Cuphead before recording human input."
        ) from exc

    gamepads = list(getattr(inputs.devices, "gamepads", []))
    if not gamepads:
        raise RuntimeError("no gamepad found; connect a controller before recording")
    return _InputsInputSource(gamepads[0])
