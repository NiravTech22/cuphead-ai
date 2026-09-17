"""Read held Cuphead keys through the same X11/XWayland server as capture.

No keyboard grab, event injection, privileged input-device access, or reader
thread is needed. Each poll queries current state, including keys already held
at startup; snapshot() returns that exact sample without querying again.
"""
from __future__ import annotations

import time

from .action_space import Action

# X keysym names, independent of the server's numeric keycodes. Preserve menu
# and EX/weapon controls even though the existing pruned Action omits them.
KEYBOARD_BINDINGS = {
    "left": "Left", "right": "Right", "up": "Up", "down": "Down",
    "jump": "z", "shoot": "x", "dash": "Shift_L", "lock": "c",
    "ex_super": "v", "switch_weapon": "Tab",
    "confirm": "Return", "pause_back": "Escape", "backspace": "BackSpace",
}
KEYBOARD_KEYS = tuple(KEYBOARD_BINDINGS.values())


def keyboard_action(keys: dict) -> Action:
    """Apply the SAME normalization/pruning as physical gamepad recordings.

    Opposite arrows cancel. Up is positive, matching Action.from_raw. This
    intentionally retains the existing schema's lossy combat-only semantics.
    """
    x = float(bool(keys["Right"])) - float(bool(keys["Left"]))
    y = float(bool(keys["Up"])) - float(bool(keys["Down"]))
    return Action.from_raw(stick_x=x, stick_y=y, jump=bool(keys["z"]),
                           duck=y < 0, shoot=bool(keys["x"]),
                           dash=bool(keys["Shift_L"]), lock=bool(keys["c"]))


class KeyboardFocusLost(RuntimeError):
    """A segment boundary: do not log keys sent to a different application."""


class X11KeyboardSource:
    def __init__(self, *, window_id: int):
        from Xlib import XK, display

        self._display = display.Display()
        self._window_id = window_id
        self._keys = None
        self._closed = False
        try:
            self._codes = {key: self._display.keysym_to_keycode(XK.string_to_keysym(key))
                           for key in KEYBOARD_KEYS}
            if not all(self._codes.values()):
                raise RuntimeError("keyboard layout lacks a required Cuphead default key")
        except BaseException:
            self.close()
            raise

    def is_focused(self) -> bool:
        focus = self._display.get_input_focus().focus
        # Wine may focus a child of the captured drawable.
        while hasattr(focus, "id"):
            if focus.id == self._window_id:
                return True
            focus = focus.query_tree().parent
        return False

    def wait_for_focus(self, timeout: float = 60.0) -> None:
        deadline = time.perf_counter() + timeout
        while not self.is_focused():
            if time.perf_counter() >= deadline:
                raise TimeoutError("focus the Cuphead window before recording (60s timeout)")
            time.sleep(0.05)

    def poll(self) -> Action:
        if self._closed:
            raise RuntimeError("poll() on a closed keyboard source")
        if not self.is_focused():
            raise KeyboardFocusLost("Cuphead lost keyboard focus")
        state = self._display.query_keymap()
        if not self.is_focused():
            raise KeyboardFocusLost("Cuphead lost focus during keyboard sample")
        self._keys = {key: int(bool(state[code // 8] & (1 << (code % 8))))
                      for key, code in self._codes.items()}
        return keyboard_action(self._keys)

    def snapshot(self) -> dict:
        if self._keys is None:
            raise RuntimeError("keyboard snapshot requires a successful poll")
        return {"backend": "x11_keyboard", "device": "x11_keyboard",
                "keys": dict(self._keys), "axes": {}, "focused": True}

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._display.close()
