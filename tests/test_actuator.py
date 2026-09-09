"""The in-memory actuator, and the virtual pad's XInput device signature.

The signature tests are the reason this file matters beyond bookkeeping.
`latency_canary.py --real` failed with "no visible response within 600 frames
of actuation" because the uinput device Wine saw was a four-button generic
joystick, not an Xbox pad, so the game never had it in an XInput slot. That
failure is invisible to any test that only checks the `Actuator` protocol.
So: assert the *device signature* Wine/SDL/udev classify on, against a list
spelled out here independently of the implementation, with a fake `ecodes`
and a fake device -- no /dev/uinput, no root, no evdev installed.
"""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.control.action_space import Action, enumerate_actions  # noqa: E402
from cuphead.control.actuator import (  # noqa: E402
    XBOX360_SIGNATURE,
    RecordingActuator,
    _UInputActuator,
    build_capabilities,
    build_uinput_kwargs,
)


class FakeEcodes:
    """Stands in for ``evdev.ecodes``: any symbol resolves to a stable int."""

    def __init__(self) -> None:
        self._codes = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._codes.setdefault(name, len(self._codes) + 1)

    def name_of(self, code):
        for name, value in self._codes.items():
            if value == code:
                return name
        raise KeyError(code)


class FakeAbsInfo:
    """Stands in for ``evdev.AbsInfo`` -- records the kwargs it was built with."""

    def __init__(self, value, min, max, fuzz, flat, resolution):  # noqa: A002
        self.value = value
        self.min = min
        self.max = max
        self.fuzz = fuzz
        self.flat = flat
        self.resolution = resolution


class FakeUInputDevice:
    """Records every write/syn/close instead of touching a real device."""

    def __init__(self) -> None:
        self.writes = []
        self.syns = 0
        self.closed = False

    def write(self, etype, code, value):
        self.writes.append((etype, code, value))

    def syn(self):
        self.syns += 1

    def close(self):
        self.closed = True


class TestRecordingActuator(unittest.TestCase):
    def test_send_records_the_action_and_a_timestamp(self):
        act = RecordingActuator()
        result = act.send(Action(move="right"))
        self.assertEqual(result.action, Action(move="right"))
        self.assertGreater(result.t_sent, 0.0)

    def test_history_accumulates_in_order(self):
        act = RecordingActuator()
        act.send(Action(move="left"))
        act.send(Action(move="right"))
        self.assertEqual([h.action.move for h in act.history], ["left", "right"])

    def test_timestamps_are_non_decreasing(self):
        act = RecordingActuator()
        for _ in range(20):
            act.send(Action())
        stamps = [h.t_sent for h in act.history]
        self.assertEqual(stamps, sorted(stamps))

    def test_close_is_a_no_op(self):
        act = RecordingActuator()
        act.send(Action())
        act.close()  # must not raise
        self.assertEqual(len(act.history), 1)


class TestXInputDeviceSignature(unittest.TestCase):
    """Everything Wine's XInput layer inspects before exposing the pad."""

    # Spelled out here rather than imported from the module: a test that reads
    # the same tuple the code writes would pass on an empty tuple.
    REQUIRED_KEYS = {
        "BTN_SOUTH",  # A
        "BTN_EAST",  # B
        "BTN_NORTH",  # Y
        "BTN_WEST",  # X
        "BTN_TL",  # LB
        "BTN_TR",  # RB
        "BTN_SELECT",
        "BTN_START",
        "BTN_THUMBL",
        "BTN_THUMBR",
    }
    REQUIRED_AXES = {
        "ABS_X",  # left stick
        "ABS_Y",
        "ABS_RX",  # right stick
        "ABS_RY",
        "ABS_Z",  # left trigger
        "ABS_RZ",  # right trigger
        "ABS_HAT0X",  # d-pad
        "ABS_HAT0Y",
    }

    def test_identifies_as_an_xbox_360_wired_pad(self):
        # SDL composes its controller GUID from bustype+vendor+product+version
        # and only maps devices it recognises; 045e:028e on BUS_USB is the
        # Xbox 360 wired pad, which is in SDL's built-in mapping database.
        self.assertEqual(XBOX360_SIGNATURE.vendor, 0x045E)
        self.assertEqual(XBOX360_SIGNATURE.product, 0x028E)
        self.assertEqual(XBOX360_SIGNATURE.bustype, 0x03)  # BUS_USB
        self.assertNotEqual(XBOX360_SIGNATURE.version, 0)

    def test_name_is_string_matchable_as_an_xbox_360_controller(self):
        self.assertIn("X-Box 360", XBOX360_SIGNATURE.name)

    def test_every_required_button_is_declared(self):
        missing = self.REQUIRED_KEYS - set(XBOX360_SIGNATURE.keys)
        self.assertEqual(missing, set(), f"XInput button block incomplete: {missing}")

    def test_every_required_axis_is_declared(self):
        missing = self.REQUIRED_AXES - {a.code for a in XBOX360_SIGNATURE.axes}
        self.assertEqual(missing, set(), f"XInput axis set incomplete: {missing}")

    def test_thumbsticks_are_signed_16_bit(self):
        for code in ("ABS_X", "ABS_Y", "ABS_RX", "ABS_RY"):
            spec = XBOX360_SIGNATURE.axis(code)
            self.assertEqual((spec.minimum, spec.maximum), (-32768, 32767), code)

    def test_triggers_are_unsigned_bytes_like_xpad(self):
        for code in ("ABS_Z", "ABS_RZ"):
            spec = XBOX360_SIGNATURE.axis(code)
            self.assertEqual((spec.minimum, spec.maximum), (0, 255), code)

    def test_dpad_hat_is_a_three_state_axis(self):
        for code in ("ABS_HAT0X", "ABS_HAT0Y"):
            spec = XBOX360_SIGNATURE.axis(code)
            self.assertEqual((spec.minimum, spec.maximum), (-1, 1), code)

    def test_every_axis_documents_why_it_exists(self):
        for spec in XBOX360_SIGNATURE.axes:
            self.assertTrue(spec.why.strip(), f"{spec.code} has no rationale")


class TestBuildUInputCall(unittest.TestCase):
    """The kwargs actually handed to ``evdev.UInput``."""

    def setUp(self):
        self.ecodes = FakeEcodes()
        self.kwargs = build_uinput_kwargs(self.ecodes, FakeAbsInfo)

    def test_capabilities_have_exactly_the_key_and_abs_event_types(self):
        caps = build_capabilities(self.ecodes, FakeAbsInfo)
        self.assertEqual(set(caps), {self.ecodes.EV_KEY, self.ecodes.EV_ABS})

    def test_capability_dict_resolves_every_declared_code(self):
        caps = self.kwargs["events"]
        key_names = {self.ecodes.name_of(c) for c in caps[self.ecodes.EV_KEY]}
        abs_names = {self.ecodes.name_of(c) for c, _ in caps[self.ecodes.EV_ABS]}
        self.assertEqual(key_names, set(XBOX360_SIGNATURE.keys))
        self.assertEqual(abs_names, {a.code for a in XBOX360_SIGNATURE.axes})

    def test_abs_info_carries_the_declared_ranges(self):
        caps = self.kwargs["events"]
        by_name = {self.ecodes.name_of(c): info for c, info in caps[self.ecodes.EV_ABS]}
        self.assertEqual((by_name["ABS_X"].min, by_name["ABS_X"].max), (-32768, 32767))
        self.assertEqual((by_name["ABS_RZ"].min, by_name["ABS_RZ"].max), (0, 255))
        self.assertEqual((by_name["ABS_HAT0Y"].min, by_name["ABS_HAT0Y"].max), (-1, 1))
        for info in by_name.values():
            self.assertEqual(info.value, 0)

    def test_usb_identity_is_passed_to_uinput_not_left_at_evdev_defaults(self):
        # evdev defaults to vendor=product=version=1, which composes an SDL
        # GUID matching nothing; the pad then enumerates but never maps.
        self.assertEqual(self.kwargs["vendor"], 0x045E)
        self.assertEqual(self.kwargs["product"], 0x028E)
        self.assertEqual(self.kwargs["bustype"], 0x03)
        self.assertEqual(self.kwargs["version"], XBOX360_SIGNATURE.version)
        self.assertEqual(self.kwargs["name"], XBOX360_SIGNATURE.name)

    def test_no_input_props_are_set(self):
        # Every INPUT_PROP_* describes a pointer/touchpad; tagging a gamepad
        # with one misclassifies it in udev.
        self.assertNotIn("input_props", self.kwargs)


class TestUInputWritePath(unittest.TestCase):
    """Where an actual mapping bug would live, exercised with a fake device."""

    def setUp(self):
        self.ecodes = FakeEcodes()
        self.dev = FakeUInputDevice()
        self.act = _UInputActuator(self.dev, self.ecodes)
        self.dev.writes.clear()  # drop the constructor's neutral state
        self.dev.syns = 0

    def written(self):
        return {self.ecodes.name_of(code): value for _, code, value in self.dev.writes}

    def test_construction_publishes_a_neutral_state_for_every_control(self):
        dev = FakeUInputDevice()
        _UInputActuator(dev, self.ecodes)
        names = {self.ecodes.name_of(c) for _, c, _ in dev.writes}
        self.assertEqual(
            names,
            set(XBOX360_SIGNATURE.keys) | {a.code for a in XBOX360_SIGNATURE.axes},
        )
        self.assertTrue(all(v == 0 for _, _, v in dev.writes))
        self.assertEqual(dev.syns, 1)

    def test_every_send_ends_in_a_syn(self):
        self.act.send(Action(move="right"))
        self.assertEqual(self.dev.syns, 1)

    def test_move_right_drives_the_left_stick_to_the_positive_rail(self):
        self.act.send(Action(move="right"))
        self.assertEqual(self.written()["ABS_X"], 32767)

    def test_duck_drives_abs_y_positive_because_evdev_y_is_positive_down(self):
        # `to_buttons` uses y-positive-up; evdev is y-positive-down. Getting
        # this wrong actuates a duck as an up-input and mirrors every locked
        # aim -- and looks perfectly healthy in every synthetic test.
        self.act.send(Action(vert="duck"))
        self.assertEqual(self.written()["ABS_Y"], 32767)

    def test_locked_north_aim_drives_abs_y_negative(self):
        self.act.send(Action(lock=True, shoot=True, aim="n"))
        self.assertEqual(self.written()["ABS_Y"], -32767)

    def test_face_buttons_map_to_the_xbox_layout(self):
        self.act.send(Action(vert="jump", shoot=True))
        w = self.written()
        self.assertEqual(w["BTN_SOUTH"], 1)  # A = jump
        self.assertEqual(w["BTN_WEST"], 1)  # X = shoot
        self.assertEqual(w["BTN_EAST"], 0)  # B = dash, not held

    def test_dash_holds_b(self):
        self.act.send(Action(dash=True))
        self.assertEqual(self.written()["BTN_EAST"], 1)

    def test_lock_holds_both_the_shoulder_and_the_analog_right_trigger(self):
        self.act.send(Action(lock=True, shoot=True, aim="e"))
        w = self.written()
        self.assertEqual(w["BTN_TR"], 1)
        self.assertEqual(w["ABS_RZ"], 255)

    def test_unlocked_action_releases_the_trigger(self):
        self.act.send(Action(move="left"))
        w = self.written()
        self.assertEqual(w["BTN_TR"], 0)
        self.assertEqual(w["ABS_RZ"], 0)

    def test_axis_values_stay_in_range_for_every_legal_action(self):
        limits = {a.code: (a.minimum, a.maximum) for a in XBOX360_SIGNATURE.axes}
        for action in enumerate_actions():
            dev = FakeUInputDevice()
            actuator = _UInputActuator(dev, self.ecodes)
            dev.writes.clear()
            actuator.send(action)
            for etype, code, value in dev.writes:
                name = self.ecodes.name_of(code)
                if etype == self.ecodes.EV_ABS:
                    lo, hi = limits[name]
                    self.assertTrue(lo <= value <= hi, f"{action} {name}={value}")
                else:
                    self.assertIn(value, (0, 1), f"{action} {name}={value}")

    def test_close_returns_the_pad_to_neutral_before_closing(self):
        self.act.send(Action(move="right", shoot=True))
        self.dev.writes.clear()
        self.act.close()
        self.assertTrue(self.dev.closed)
        self.assertTrue(all(v == 0 for _, _, v in self.dev.writes))
        # A pad closed mid-hold leaves the game with a stuck input.
        self.assertEqual(self.written()["ABS_X"], 0)


if __name__ == "__main__":
    unittest.main()
