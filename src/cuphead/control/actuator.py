"""Action output: sending an ``Action`` to a virtual controller.

The counterpart to ``perception.capture`` on the input side. The latency
canary needs both: it sends an action through here and watches
``perception.capture`` for the first frame whose content changed after the
send timestamp.

Real actuation (a virtual XInput pad via uinput/evdev on Linux) imports
lazily, same reasoning as the capture backend: nothing here should require
hardware access to import or test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Protocol

from .action_space import Action


@dataclass(frozen=True)
class Actuation:
    """A record of one send: what was sent, and exactly when."""

    action: Action
    t_sent: float


class Actuator(Protocol):
    def send(self, action: Action) -> Actuation: ...

    def close(self) -> None: ...


class RecordingActuator:
    """In-memory actuator for tests and the latency canary's synthetic mode.

    Records every send with a ``perf_counter`` timestamp and does nothing
    else -- there is no real controller to drive. This is the stdlib-only
    half of the ``Actuator`` contract that ``open_vgamepad_actuator``'s real
    backend must also satisfy.
    """

    def __init__(self) -> None:
        self.history: List[Actuation] = []

    def send(self, action: Action) -> Actuation:
        act = Actuation(action=action, t_sent=time.perf_counter())
        self.history.append(act)
        return act

    def close(self) -> None:
        pass


def open_vgamepad_actuator() -> Actuator:
    """Real virtual-gamepad actuation via ``evdev``/``uinput``, imported lazily.

    This is the backend an actual training session drives the game with. It
    requires ``/dev/uinput`` access (root, or the ``uinput`` group on most
    distros) and is not exercised by the test suite for the same reason
    ``capture.open_screen_source`` isn't: there is no hardware in CI.
    """
    try:
        import evdev  # type: ignore[import-untyped]
        from evdev import AbsInfo, UInput, ecodes
    except ImportError as exc:
        raise RuntimeError(
            "evdev is not installed. Run `pip install -r requirements.txt` on the "
            "machine actually running Cuphead before using the real actuator."
        ) from exc

    capabilities = {
        ecodes.EV_KEY: [ecodes.BTN_SOUTH, ecodes.BTN_EAST, ecodes.BTN_WEST, ecodes.BTN_TR],
        ecodes.EV_ABS: [
            (ecodes.ABS_X, AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=0, resolution=0)),
            (ecodes.ABS_Y, AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=0, resolution=0)),
        ],
    }

    class _UInputActuator:
        def __init__(self) -> None:
            self._dev = UInput(capabilities, name="cuphead-ai-virtual-pad")

        def send(self, action: Action) -> Actuation:
            buttons = action.to_buttons()
            self._dev.write(ecodes.EV_ABS, ecodes.ABS_X, int(buttons["stick_x"] * 32767))
            self._dev.write(ecodes.EV_ABS, ecodes.ABS_Y, int(buttons["stick_y"] * 32767))
            self._dev.write(ecodes.EV_KEY, ecodes.BTN_SOUTH, int(buttons["a"]))
            self._dev.write(ecodes.EV_KEY, ecodes.BTN_WEST, int(buttons["x"]))
            self._dev.write(ecodes.EV_KEY, ecodes.BTN_EAST, int(buttons["b"]))
            self._dev.write(ecodes.EV_KEY, ecodes.BTN_TR, int(buttons["rt"]))
            self._dev.syn()
            return Actuation(action=action, t_sent=time.perf_counter())

        def close(self) -> None:
            self._dev.close()

    return _UInputActuator()
