"""Lazy Windows XInput output using vgamepad and the ViGEmBus driver."""
from __future__ import annotations

import time

from .actuator import Actuation
from .timed_input import ControlInput


class WindowsGamepadActuator:
    def __init__(self, pad, buttons):
        self._pad = pad
        self._buttons = buttons
        self.neutral()

    def _require_open(self):
        if self._pad is None:
            raise RuntimeError("controller is closed")

    def neutral(self):
        self._require_open()
        self._pad.reset()
        self._pad.update()

    def _publish(self, state, *, rt=False):
        self._require_open()
        self._pad.reset()
        self._pad.left_joystick_float(x_value_float=state.stick_x,
                                     y_value_float=state.stick_y)
        for field, button in (("a", "A"), ("b", "B"), ("x", "X"),
                              ("y", "Y"), ("rb", "RIGHT_SHOULDER"),
                              ("start", "START")):
            if getattr(state, field):
                self._pad.press_button(button=getattr(self._buttons, "XUSB_GAMEPAD_" + button))
        self._pad.right_trigger_float(value_float=float(rt))
        self._pad.update()

    def send(self, action):
        b = action.to_buttons()
        self._publish(ControlInput("combat", stick_x=b["stick_x"], stick_y=b["stick_y"],
                                   a=b["a"], b=b["b"], x=b["x"], rb=b["rt"]), rt=b["rt"])
        return Actuation(action, time.perf_counter())

    def send_buttons(self, state):
        self._publish(state)

    def send_menu(self, action):
        self.send_buttons(ControlInput(action.name, stick_x=action.stick_x,
                                       stick_y=action.stick_y, a=action.confirm,
                                       start=action.skip, hold_frames=action.hold_frames))

    def neutral_menu(self):
        self.neutral()

    def close(self):
        if self._pad is not None:
            try:
                self.neutral()
            finally:
                self._pad = None


def open_windows_actuator():
    try:
        import vgamepad
    except ImportError as exc:
        raise RuntimeError("Install vgamepad with `pip install -r requirements.txt` on Windows.") from exc
    try:
        pad = vgamepad.VX360Gamepad()
    except Exception as exc:
        raise RuntimeError("Could not create an XInput controller. Install the ViGEmBus driver required by vgamepad.") from exc
    return WindowsGamepadActuator(pad, vgamepad.XUSB_BUTTON)
