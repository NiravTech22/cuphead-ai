"""Keyboard held-state capture and compatibility with gamepad action records."""
import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cuphead.control.action_space import Action
from cuphead.control.keyboard_input import (
    KEYBOARD_KEYS, KeyboardFocusLost, X11KeyboardSource, keyboard_action,
)


def keys(*held):
    return {key: int(key in held) for key in KEYBOARD_KEYS}


class KeyboardActionTests(unittest.TestCase):
    def test_defaults_and_gamepad_equivalence_for_all_combat_combinations(self):
        combat = ("Left", "Right", "Up", "Down", "z", "x", "Shift_L", "c")
        for bits in itertools.product((0, 1), repeat=len(combat)):
            state = keys(*(key for key, held in zip(combat, bits) if held))
            sx = float(state["Right"] - state["Left"])
            sy = float(state["Up"] - state["Down"])
            pad = Action.from_raw(stick_x=sx, stick_y=sy, jump=bool(state["z"]),
                shoot=bool(state["x"]), dash=bool(state["Shift_L"]),
                lock=bool(state["c"]), duck=sy < 0)
            action = keyboard_action(state)
            self.assertEqual(action, pad)
            self.assertTrue(action.is_legal())
            self.assertEqual(action.to_buttons(), pad.to_buttons())
            self.assertEqual(set(action.to_buttons()), {"stick_x", "stick_y", "a", "b", "x", "rt"})

    def test_known_actions_and_opposites(self):
        self.assertEqual(keyboard_action(keys("Right", "z", "x")).to_buttons(),
            {"stick_x": 1.0, "stick_y": 0.0, "a": True, "x": True, "b": False, "rt": False})
        self.assertEqual(keyboard_action(keys("Left", "Right", "Up", "Down")), Action())
        self.assertEqual(keyboard_action(keys("Up", "Right", "c", "x")).aim, "ne")
        self.assertTrue(keyboard_action(keys("Shift_L")).dash)
        self.assertEqual(keyboard_action(keys("Down")).vert, "duck")

    def test_extra_controls_do_not_expand_predictor_schema(self):
        self.assertEqual(keyboard_action(keys("v", "Tab", "Return", "Escape")), Action())


class KeyboardSourceTests(unittest.TestCase):
    def source(self):
        codes = {key: 8 + i for i, key in enumerate(KEYBOARD_KEYS)}
        class Display:
            def __init__(self):
                self.focus = SimpleNamespace(id=99)
                self.state = [0] * 32
                self.queries = 0
                self.closed = False
                self.lose_after_query = False
            def keysym_to_keycode(self, key):
                return codes[key]
            def get_input_focus(self):
                return SimpleNamespace(focus=self.focus)
            def query_keymap(self):
                self.queries += 1
                if self.lose_after_query:
                    self.focus = 0
                return self.state[:]
            def close(self):
                self.closed = True
            def held(self, *held):
                self.state = [0] * 32
                for key in held:
                    code = codes[key]
                    self.state[code // 8] |= 1 << (code % 8)
        display = Display()
        xlib = SimpleNamespace(XK=SimpleNamespace(string_to_keysym=lambda key: key),
                               display=SimpleNamespace(Display=lambda: display))
        with patch.dict(sys.modules, Xlib=xlib):
            source = X11KeyboardSource(window_id=99)
        self.addCleanup(source.close)
        return source, display

    def test_initial_held_release_chord_and_cached_snapshot(self):
        source, display = self.source()
        with self.assertRaises(RuntimeError):
            source.snapshot()
        display.held("Right", "z", "x", "v", "Tab")
        self.assertEqual(source.poll(), keyboard_action(keys("Right", "z", "x", "v", "Tab")))
        display.held()
        snapshot = source.snapshot()
        self.assertEqual(snapshot["keys"], keys("Right", "z", "x", "v", "Tab"))
        self.assertEqual(display.queries, 1)
        snapshot["keys"]["z"] = 0
        self.assertEqual(source.snapshot()["keys"]["z"], 1)
        self.assertEqual(source.poll(), Action())
        self.assertEqual(source.snapshot()["keys"], keys())

    def test_focus_loss_before_and_during_poll_stops_capture(self):
        source, display = self.source()
        display.focus = 0
        with self.assertRaises(KeyboardFocusLost):
            source.poll()
        self.assertEqual(display.queries, 0)
        display.focus = SimpleNamespace(id=99)
        display.lose_after_query = True
        with self.assertRaises(KeyboardFocusLost):
            source.poll()
        with self.assertRaises(RuntimeError):
            source.snapshot()

    def test_child_window_focus_and_timeout(self):
        source, display = self.source()
        display.focus = SimpleNamespace(id=100, query_tree=lambda: SimpleNamespace(parent=SimpleNamespace(id=99)))
        self.assertTrue(source.is_focused())
        source.wait_for_focus(timeout=0)
        display.focus = 0
        with self.assertRaises(TimeoutError):
            source.wait_for_focus(timeout=0)

    def test_closed_source_and_server_failure_do_not_return_stale_input(self):
        source, display = self.source()
        source.poll()
        with patch.object(display, "query_keymap", side_effect=OSError("disconnected")):
            with self.assertRaises(OSError):
                source.poll()
        source.close()
        self.assertTrue(display.closed)
        with self.assertRaises(RuntimeError):
            source.poll()


if __name__ == "__main__":
    unittest.main()
