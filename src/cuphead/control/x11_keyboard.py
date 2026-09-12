"""Explicit keyboard fallback for Wine installations that fail XInput routing."""

from __future__ import annotations


class X11KeyboardActuator:
    def __init__(self, title: str = "Cuphead"):
        from Xlib import XK, X
        from Xlib.ext import xtest

        from ..perception.window_capture import X11WindowSource

        self._source = X11WindowSource(title)
        self._display = self._source._display
        self._window = self._source._window
        self._X, self._XK, self._xtest = X, XK, xtest
        self._held = set()
        self._window.set_input_focus(X.RevertToParent, X.CurrentTime)
        self._display.sync()

    def _set(self, desired):
        # Never send input if the game window no longer exists.
        self._window.get_geometry()
        self._window.set_input_focus(self._X.RevertToParent, self._X.CurrentTime)
        for key in self._held - desired:
            self._xtest.fake_input(self._display, self._X.KeyRelease, key)
        for key in desired - self._held:
            self._xtest.fake_input(self._display, self._X.KeyPress, key)
        self._display.sync()
        self._held = desired

    def send_buttons(self, state):
        names = []
        if state.stick_x < -0.35:
            names.append("Left")
        if state.stick_x > 0.35:
            names.append("Right")
        if state.stick_y < -0.35:
            names.append("Down")
        if state.stick_y > 0.35:
            names.append("Up")
        for active, name in (
            (state.a, "z"),
            (state.b, "c"),
            (state.x, "x"),
            (state.y, "Tab"),
            (state.rb, "Shift_L"),
            (state.start, "Return"),
        ):
            if active:
                names.append(name)
        self._set(
            {
                self._display.keysym_to_keycode(self._XK.string_to_keysym(name))
                for name in names
            }
        )

    def neutral(self):
        # Release without changing focus, including when the game exited.
        for key in self._held:
            self._xtest.fake_input(self._display, self._X.KeyRelease, key)
        self._display.sync()
        self._held.clear()

    def close(self):
        try:
            self.neutral()
        finally:
            self._source.close()
