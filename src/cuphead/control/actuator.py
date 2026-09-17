"""Action output: sending an ``Action`` to a virtual controller.

The counterpart to ``perception.capture`` on the input side. The latency
canary needs both: it sends an action through here and watches
``perception.capture`` for the first frame whose content changed after the
send timestamp.

Real actuation (a virtual XInput pad via uinput/evdev on Linux) imports
lazily, same reasoning as the capture backend: nothing here should require
hardware access to import or test.

The *device signature* -- name, USB IDs, and the full capability set -- is
declared here as plain stdlib data (``DeviceSignature`` / ``AxisSpec``) and
only lowered to ``evdev`` types at open time. That split exists so the thing
most likely to be wrong (the capability set Wine's XInput layer inspects) is
unit-testable with no ``/dev/uinput``, no root, and no evdev installed.
"""

from __future__ import annotations

import time
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple

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


# ---------------------------------------------------------------------------
# Device signature
# ---------------------------------------------------------------------------
#
# WHY THIS IS NOT A MINIMAL PAD
#
# Cuphead under Wine/Proton does not read evdev. It reads XInput, which Wine
# synthesizes: winebus.sys enumerates Linux input devices (via udev/SDL),
# decides which ones are *game controllers*, builds an HID descriptor for
# each, and only devices that classify as an Xbox-compatible gamepad reach
# xinput1_3/xinput1_4 and hence the game. A uinput device with four buttons
# and two axes classifies as a generic joystick at best, which is why
# `latency_canary.py --real` saw zero response: the pad existed, udev saw it,
# and the game never had it in its XInput slot list.
#
# Three properties drive that classification, and all three must hold:
#
#   1. udev's `input_id` builtin must tag the node ID_INPUT_JOYSTICK. It does
#      that from capabilities alone: an ABS_X/ABS_Y pair plus buttons in the
#      BTN_GAMEPAD range. A device missing the gamepad button block, or one
#      carrying only BTN_TR-ish odds and ends, can land as ID_INPUT_KEY.
#   2. SDL (Wine's controller backend on most modern builds) maps a device to
#      the *game controller* API only when it recognises its GUID, which SDL
#      composes from bustype + vendor + product + version. 0x045e/0x028e on
#      BUS_USB is the Microsoft Xbox 360 wired pad and is in SDL's built-in
#      mapping database, so it maps with zero user configuration. An unknown
#      VID/PID gets no mapping and stays a bare joystick.
#   3. Wine's fallback heuristics (and several community mapping layers) also
#      string-match "Xbox 360 Controller" in the device name, so the name is
#      not cosmetic.
#
# Hence: identify as the Xbox 360 wired pad, exactly, and expose the exact
# capability set the kernel's `xpad` driver exposes for it. Each entry below
# carries the reason it must be present.

_AXIS_MIN = -32768
_AXIS_MAX = 32767
#: xpad reports the analog triggers as unsigned bytes, not signed shorts.
_TRIGGER_MAX = 255


@dataclass(frozen=True)
class AxisSpec:
    """One absolute axis, named by its ``evdev.ecodes`` symbol.

    Symbolic rather than numeric so this module -- and its tests -- never
    need evdev installed. ``build_capabilities`` resolves the names.
    """

    code: str
    minimum: int
    maximum: int
    fuzz: int = 0
    flat: int = 0
    resolution: int = 0
    why: str = ""


@dataclass(frozen=True)
class DeviceSignature:
    """Everything Wine/SDL/udev inspect when deciding "is this an Xbox pad?"."""

    name: str
    vendor: int
    product: int
    version: int
    bustype: int
    keys: Tuple[str, ...]
    axes: Tuple[AxisSpec, ...]

    def axis(self, code: str) -> AxisSpec:
        for spec in self.axes:
            if spec.code == code:
                return spec
        raise KeyError(code)


#: BUS_USB. Declared here rather than read from ``ecodes`` so the signature
#: stays importable without evdev; the value is a stable kernel ABI constant.
BUS_USB = 0x03

#: The Xbox 360 wired pad, as the kernel's ``xpad`` driver presents it.
XBOX360_SIGNATURE = DeviceSignature(
    # String-matched by Wine's and SDL's fallback mapping heuristics. Keep the
    # "Xbox 360 Controller" substring; the vendor prefix mirrors xpad's own
    # "Microsoft X-Box 360 pad" style naming closely enough for both.
    name="Microsoft X-Box 360 pad (cuphead-ai virtual)",
    vendor=0x045E,  # Microsoft
    product=0x028E,  # Xbox 360 wired controller -- in SDL's built-in GUID map
    version=0x0110,  # what xpad reports; part of the SDL GUID, so it matters
    bustype=BUS_USB,  # a "virtual"/BUS_VIRTUAL pad is not recognised as USB HID
    keys=(
        # The BTN_GAMEPAD block. udev's input_id needs BTN_GAMEPAD (== BTN_SOUTH)
        # present to tag ID_INPUT_JOYSTICK at all, and XInput expects the whole
        # face/shoulder/menu set to exist even for buttons we never press --
        # a missing button shifts nothing, but a missing *block* changes the
        # HID report descriptor Wine builds and can drop the pad to "joystick".
        "BTN_SOUTH",  # A -- jump
        "BTN_EAST",  # B -- dash
        "BTN_NORTH",  # Y -- weapon switch (unused by the planner; xpad has it)
        "BTN_WEST",  # X -- shoot
        "BTN_TL",  # LB -- unused, present for a complete XInput button map
        "BTN_TR",  # RB -- aim lock (see _UInputActuator.send)
        "BTN_SELECT",  # Back
        "BTN_START",  # Start -- also how a menu is dismissed if a run wedges
        "BTN_MODE",  # Guide; xpad exposes it, SDL's 360 mapping references it
        "BTN_THUMBL",  # L3 -- unused; part of the standard block
        "BTN_THUMBR",  # R3 -- unused; part of the standard block
    ),
    axes=(
        AxisSpec(
            "ABS_X", _AXIS_MIN, _AXIS_MAX, fuzz=16, flat=128,
            why="left stick X -- movement; half of the ABS_X/ABS_Y pair udev "
                "requires to classify the node as a joystick",
        ),
        AxisSpec(
            "ABS_Y", _AXIS_MIN, _AXIS_MAX, fuzz=16, flat=128,
            why="left stick Y -- duck/aim; evdev Y is positive-DOWN",
        ),
        AxisSpec(
            "ABS_RX", _AXIS_MIN, _AXIS_MAX, fuzz=16, flat=128,
            why="right stick X -- never driven, but XInput's gamepad struct has "
                "it and Wine's descriptor builder expects both sticks",
        ),
        AxisSpec(
            "ABS_RY", _AXIS_MIN, _AXIS_MAX, fuzz=16, flat=128,
            why="right stick Y -- same reason as ABS_RX",
        ),
        AxisSpec(
            "ABS_Z", 0, _TRIGGER_MAX,
            why="left trigger -- unsigned byte like xpad; a 360 pad without "
                "triggers does not match SDL's 360 mapping",
        ),
        AxisSpec(
            "ABS_RZ", 0, _TRIGGER_MAX,
            why="right trigger -- Cuphead's default Xbox binding for aim lock",
        ),
        AxisSpec(
            "ABS_HAT0X", -1, 1,
            why="D-pad X -- never driven (movement goes through the stick), but "
                "its absence is one of the things that makes a device read as a "
                "generic joystick rather than a gamepad",
        ),
        AxisSpec(
            "ABS_HAT0Y", -1, 1,
            why="D-pad Y -- same reason as ABS_HAT0X",
        ),
    ),
)

#: udev/Wine need a moment between device creation and first use: udev has to
#: run input_id and create the node, and winebus only rescans on the hotplug
#: event. Writing immediately after ``UInput(...)`` means the first actuations
#: land before the game has the pad in an XInput slot -- which reads exactly
#: like the latency failure this signature fixes.
UINPUT_SETTLE_SECONDS = 0.5


def build_capabilities(ecodes: Any, abs_info: Any) -> Dict[int, Any]:
    """Lower ``XBOX360_SIGNATURE`` to the dict ``evdev.UInput`` wants.

    ``ecodes`` and ``abs_info`` are injected (rather than imported) so this is
    testable without evdev and without a device.
    """
    keys = [getattr(ecodes, name) for name in XBOX360_SIGNATURE.keys]
    axes = [
        (
            getattr(ecodes, spec.code),
            abs_info(
                value=0,
                min=spec.minimum,
                max=spec.maximum,
                fuzz=spec.fuzz,
                flat=spec.flat,
                resolution=spec.resolution,
            ),
        )
        for spec in XBOX360_SIGNATURE.axes
    ]
    return {ecodes.EV_KEY: keys, ecodes.EV_ABS: axes}


def build_uinput_kwargs(ecodes: Any, abs_info: Any) -> Dict[str, Any]:
    """The full ``UInput(**kwargs)`` call: capabilities *and* USB identity.

    Passing the identity is the half that is easy to forget -- evdev defaults
    to vendor/product/version 1 on BUS_USB, which composes an SDL GUID that
    matches nothing, so the pad enumerates but never maps to XInput.
    """
    return {
        "events": build_capabilities(ecodes, abs_info),
        "name": XBOX360_SIGNATURE.name,
        "vendor": XBOX360_SIGNATURE.vendor,
        "product": XBOX360_SIGNATURE.product,
        "version": XBOX360_SIGNATURE.version,
        "bustype": XBOX360_SIGNATURE.bustype,
        # A physical-path string makes the node look like a real USB input to
        # udev's rules. INPUT_PROP_* (``input_props``) is deliberately left
        # unset: every property in that set describes pointers/touchpads, and
        # tagging a pad with one actively misclassifies it.
        "phys": "cuphead-ai/virtual-xbox360",
    }


def _stick_value(v: float) -> int:
    """Map a [-1, 1] ``to_buttons`` stick component to a signed-16 axis value."""
    return max(_AXIS_MIN, min(_AXIS_MAX, int(round(v * _AXIS_MAX))))


class _UInputActuator:
    """Drives a real (or injected fake) uinput device from an ``Action``.

    The device and ``ecodes`` are constructor arguments so the write path --
    which is where a mapping bug would actually live -- can be exercised with
    a recording double instead of hardware.
    """

    def __init__(self, dev: Any, ecodes: Any) -> None:
        self._dev = dev
        self._ecodes = ecodes
        self.neutral()

    def neutral(self) -> None:
        """Publish a defined all-zero state.

        A uinput device starts with every axis unset. Wine reads the current
        value when it builds its first HID report, so without this the game's
        first view of the pad can be an undefined stick position.
        """
        e = self._ecodes
        for spec in XBOX360_SIGNATURE.axes:
            self._dev.write(e.EV_ABS, getattr(e, spec.code), 0)
        for name in XBOX360_SIGNATURE.keys:
            self._dev.write(e.EV_KEY, getattr(e, name), 0)
        self._dev.syn()

    def send(self, action: Action) -> Actuation:
        e = self._ecodes
        buttons = action.to_buttons()
        # `to_buttons` uses the project-wide "y positive = up" convention (see
        # human_input._blank_raw_state). evdev's ABS_Y is positive-DOWN, so it
        # is negated here -- without this, a duck actuates as up and every
        # aim-lock vertical is mirrored.
        self._dev.write(e.EV_ABS, e.ABS_X, _stick_value(buttons["stick_x"]))
        self._dev.write(e.EV_ABS, e.ABS_Y, _stick_value(-buttons["stick_y"]))
        self._dev.write(e.EV_KEY, e.BTN_SOUTH, int(buttons["a"]))
        self._dev.write(e.EV_KEY, e.BTN_WEST, int(buttons["x"]))
        self._dev.write(e.EV_KEY, e.BTN_EAST, int(buttons["b"]))
        # Aim lock is driven on both surfaces Cuphead accepts for it: RB (the
        # convention human_input records demonstrations with) and the right
        # trigger (the default Xbox binding). Under XInput these are distinct
        # controls, and we do not know which the running config binds, so hold
        # both -- neither is bound to anything else in the default layout.
        self._dev.write(e.EV_KEY, e.BTN_TR, int(buttons["rt"]))
        self._dev.write(e.EV_ABS, e.ABS_RZ, _TRIGGER_MAX if buttons["rt"] else 0)
        self._dev.syn()
        return Actuation(action=action, t_sent=time.perf_counter())

    def send_buttons(self, state: Any) -> None:
        """Apply a complete menu/overworld/combat input (Y-up project convention)."""
        e = self._ecodes
        self._dev.write(e.EV_ABS, e.ABS_X, _stick_value(state.stick_x))
        self._dev.write(e.EV_ABS, e.ABS_Y, _stick_value(-state.stick_y))
        bindings = {"BTN_SOUTH": state.a, "BTN_EAST": state.b,
                    "BTN_WEST": state.x, "BTN_NORTH": state.y,
                    "BTN_TR": state.rb, "BTN_START": state.start}
        for name in XBOX360_SIGNATURE.keys:
            self._dev.write(e.EV_KEY, getattr(e, name), int(bindings.get(name, False)))
        self._dev.write(e.EV_ABS, e.ABS_RZ, 0)
        self._dev.syn()

    def send_menu(self, action: Any) -> None:
        from .timed_input import ControlInput
        self.send_buttons(ControlInput(action.name, stick_x=action.stick_x,
                                       stick_y=action.stick_y, a=action.confirm,
                                       start=action.skip, hold_frames=action.hold_frames))

    def neutral_menu(self) -> None:
        self.neutral()

    def close(self) -> None:
        self.neutral()
        self._dev.close()


def open_vgamepad_actuator(*, settle_seconds: float = UINPUT_SETTLE_SECONDS,
                          backend: str | None = None) -> Actuator:
    """Lazy virtual Xbox output: vgamepad on Windows, evdev/uinput on Linux.

    ``backend`` can explicitly select ``vgamepad`` or ``uinput``.

    This is the backend an actual training session drives the game with. It
    requires ``/dev/uinput`` access (root, or the ``uinput`` group on most
    distros) and is not exercised by the test suite for the same reason
    ``capture.open_screen_source`` isn't: there is no hardware in CI. What
    *is* tested is the device signature and the write path, via
    ``build_uinput_kwargs`` and ``_UInputActuator`` with a fake device.

    Under Wine/Proton the pad must exist *before* Cuphead enumerates
    controllers -- open the actuator, then launch the game.
    """
    backend = backend or ("vgamepad" if platform.system() == "Windows" else "uinput")
    if backend not in {"vgamepad", "uinput"}:
        raise ValueError("backend must be 'vgamepad' or 'uinput'")
    if backend == "vgamepad":
        from .windows_gamepad import open_windows_actuator
        actuator = open_windows_actuator()
        if settle_seconds > 0:
            time.sleep(settle_seconds)
        return actuator
    try:
        from evdev import AbsInfo, UInput, ecodes  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "evdev is not installed. Run `pip install -r requirements.txt` on the "
            "machine actually running Cuphead before using the real actuator."
        ) from exc

    try:
        dev = UInput(**build_uinput_kwargs(ecodes, AbsInfo))
    except OSError as exc:
        raise RuntimeError(
            "could not create the virtual Xbox 360 pad via /dev/uinput. "
            "Load the uinput module (`sudo modprobe uinput`) and grant this user "
            "read/write access to /dev/uinput (usually by adding it to the `input` "
            "or `uinput` group, then logging in again)."
        ) from exc
    actuator = _UInputActuator(dev, ecodes)
    if settle_seconds > 0:
        time.sleep(settle_seconds)
    return actuator
