"""Action.from_raw normalization and the human-input sources."""

import sys
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.control.action_space import Action, STICK_DEADZONE, enumerate_actions  # noqa: E402
from cuphead.control.human_input import (  # noqa: E402
    ScriptedInputSource,
    _InputsInputSource,
    open_gamepad_source,
)


class TestFromRawProducesLegalActions(unittest.TestCase):
    def test_every_combination_of_raw_flags_is_legal(self):
        """from_raw is the recorder's half of the round trip into the pruned
        space -- if it can ever emit an illegal action, a human demonstration
        becomes an unusable imitation-learning target."""
        legal = set(enumerate_actions())
        stick_samples = [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.7, 0.7), (0.0, 0.0)]
        bools = [False, True]
        count = 0
        for sx, sy in stick_samples:
            for jump in bools:
                for duck in bools:
                    for dash in bools:
                        for shoot in bools:
                            for lock in bools:
                                a = Action.from_raw(
                                    stick_x=sx, stick_y=sy, jump=jump, duck=duck,
                                    dash=dash, shoot=shoot, lock=lock,
                                )
                                self.assertIn(a, legal, (sx, sy, jump, duck, dash, shoot, lock))
                                count += 1
        self.assertEqual(count, 6 * 32)

    def test_duck_beats_jump(self):
        a = Action.from_raw(jump=True, duck=True)
        self.assertEqual(a.vert, "duck")

    def test_dash_drops_a_simultaneous_duck(self):
        a = Action.from_raw(duck=True, dash=True)
        self.assertEqual(a.vert, "none")
        self.assertTrue(a.dash)

    def test_dash_cancels_shoot(self):
        a = Action.from_raw(dash=True, shoot=True)
        self.assertTrue(a.dash)
        self.assertFalse(a.shoot)

    def test_idle_lock_is_dropped(self):
        a = Action.from_raw(lock=True)
        self.assertFalse(a.lock)

    def test_lock_with_shoot_is_kept_and_repurposes_the_stick_for_aim(self):
        a = Action.from_raw(stick_x=1.0, stick_y=0.0, lock=True, shoot=True)
        self.assertTrue(a.lock)
        self.assertEqual(a.move, "none")
        self.assertEqual(a.aim, "e")

    def test_lock_while_ducking_is_not_reachable(self):
        a = Action.from_raw(duck=True, lock=True, shoot=True)
        self.assertFalse(a.lock)
        self.assertEqual(a.vert, "duck")

    def test_unlocked_stick_quantizes_to_left_or_right(self):
        self.assertEqual(Action.from_raw(stick_x=-1.0).move, "left")
        self.assertEqual(Action.from_raw(stick_x=1.0).move, "right")
        self.assertEqual(Action.from_raw(stick_x=0.0).move, "none")

    def test_stick_below_deadzone_reads_as_centered(self):
        a = Action.from_raw(stick_x=STICK_DEADZONE * 0.5)
        self.assertEqual(a.move, "none")

    def test_locked_aim_quantizes_to_the_nearest_of_eight_directions(self):
        north = Action.from_raw(stick_x=0.0, stick_y=1.0, lock=True, shoot=True)
        self.assertEqual(north.aim, "n")
        west = Action.from_raw(stick_x=-1.0, stick_y=0.0, lock=True, shoot=True)
        self.assertEqual(west.aim, "w")

    def test_centered_stick_while_locked_defaults_to_east(self):
        a = Action.from_raw(stick_x=0.0, stick_y=0.0, lock=True, shoot=True)
        self.assertEqual(a.aim, "e")


class TestScriptedInputSource(unittest.TestCase):
    def test_replays_actions_in_order(self):
        actions = [Action(move="left"), Action(move="right"), Action(vert="jump")]
        src = ScriptedInputSource(actions)
        self.assertEqual([src.poll() for _ in range(3)], actions)

    def test_raises_when_exhausted_rather_than_looping(self):
        src = ScriptedInputSource([Action()])
        src.poll()
        with self.assertRaises(StopIteration):
            src.poll()

    def test_empty_sequence_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            ScriptedInputSource([])

    def test_closed_source_rejects_further_polls(self):
        src = ScriptedInputSource([Action(), Action()])
        src.close()
        with self.assertRaises(RuntimeError):
            src.poll()


class FakeEvent:
    """One ``inputs``-shaped event: ev_type/code/state, as the package emits."""

    def __init__(self, ev_type, code, state):
        self.ev_type = ev_type
        self.code = code
        self.state = state


class FakeGamepad:
    """Stands in for ``inputs.devices.gamepads[0]``.

    ``read()`` returns queued batches and then blocks like the real one does,
    so the reader-thread contract is exercised rather than assumed.
    """

    def __init__(self, batches=(), then_raise=None):
        self._batches = list(batches)
        self._then_raise = then_raise
        self.closed = False

    def read(self):
        if self._batches:
            return self._batches.pop(0)
        if self._then_raise is not None:
            raise self._then_raise
        time.sleep(0.01)  # mimic inputs' blocking read
        return []

    def close(self):
        self.closed = True


class TestInputsBackendNormalization(unittest.TestCase):
    """The Windows backend must land on the same 56-action space as evdev.

    Events are folded in directly (reader thread off) so these assertions are
    deterministic; the thread itself is covered separately below.
    """

    def _source(self):
        return _InputsInputSource(FakeGamepad(), start_reader=False)

    def feed(self, src, *events):
        for event in events:
            src._apply(event)
        return src.poll()

    def test_idle_state_polls_as_the_neutral_action(self):
        self.assertEqual(self._source().poll(), Action())

    def test_every_polled_action_is_legal(self):
        """Same guarantee the evdev path leans on: a recorded demonstration is
        only an imitation target if it is inside the pruned space."""
        legal = set(enumerate_actions())
        src = self._source()
        axis_values = [-32768, -20000, 0, 20000, 32767]
        buttons = ["BTN_SOUTH", "BTN_WEST", "BTN_EAST", "BTN_TR"]
        for x in axis_values:
            for y in axis_values:
                for mask in range(16):
                    src._apply(FakeEvent("Absolute", "ABS_X", x))
                    src._apply(FakeEvent("Absolute", "ABS_Y", y))
                    for bit, code in enumerate(buttons):
                        src._apply(FakeEvent("Key", code, (mask >> bit) & 1))
                    self.assertIn(src.poll(), legal, (x, y, mask))

    def test_stick_right_reads_as_move_right(self):
        a = self.feed(self._source(), FakeEvent("Absolute", "ABS_X", 32767))
        self.assertEqual(a.move, "right")

    def test_stick_left_reads_as_move_left(self):
        a = self.feed(self._source(), FakeEvent("Absolute", "ABS_X", -32768))
        self.assertEqual(a.move, "left")

    def test_xinput_y_is_positive_up_and_is_not_negated(self):
        """XInput already reports up as positive, unlike evdev. If this backend
        negated it too, every locked aim recorded on Windows would be vertically
        mirrored relative to Linux -- silently, in a shared buffer."""
        src = self._source()
        a = self.feed(
            src,
            FakeEvent("Key", "BTN_TR", 1),
            FakeEvent("Key", "BTN_WEST", 1),
            FakeEvent("Absolute", "ABS_Y", 32767),
        )
        self.assertEqual(a.aim, "n")

    def test_buttons_map_to_the_same_flags_as_the_evdev_backend(self):
        self.assertEqual(self.feed(self._source(), FakeEvent("Key", "BTN_SOUTH", 1)).vert, "jump")
        self.assertTrue(self.feed(self._source(), FakeEvent("Key", "BTN_WEST", 1)).shoot)
        self.assertTrue(self.feed(self._source(), FakeEvent("Key", "BTN_EAST", 1)).dash)

    def test_button_release_clears_the_held_flag(self):
        src = self._source()
        self.feed(src, FakeEvent("Key", "BTN_SOUTH", 1))
        a = self.feed(src, FakeEvent("Key", "BTN_SOUTH", 0))
        self.assertEqual(a.vert, "none")

    def test_held_state_persists_across_polls_with_no_new_events(self):
        """Controller state is held, not edge-triggered: two polls inside one
        button hold must both report the button down."""
        src = self._source()
        self.feed(src, FakeEvent("Key", "BTN_SOUTH", 1))
        self.assertEqual(src.poll().vert, "jump")
        self.assertEqual(src.poll().vert, "jump")

    def test_dpad_hat_axis_drives_movement(self):
        a = self.feed(self._source(), FakeEvent("Absolute", "ABS_HAT0X", 1))
        self.assertEqual(a.move, "right")

    def test_dpad_buttons_drive_movement(self):
        a = self.feed(self._source(), FakeEvent("Key", "BTN_DPAD_LEFT", 1))
        self.assertEqual(a.move, "left")

    def test_unmapped_events_are_ignored_not_fatal(self):
        """A trigger pull or sync event must not kill a recording session."""
        a = self.feed(
            self._source(),
            FakeEvent("Absolute", "ABS_Z", 255),
            FakeEvent("Sync", "SYN_REPORT", 0),
            FakeEvent("Key", "BTN_THUMBL", 1),
        )
        self.assertEqual(a, Action())

    def test_precedence_rules_apply_through_from_raw(self):
        a = self.feed(
            self._source(), FakeEvent("Key", "BTN_EAST", 1), FakeEvent("Key", "BTN_WEST", 1)
        )
        self.assertTrue(a.dash)
        self.assertFalse(a.shoot)


class TestInputsBackendReaderThread(unittest.TestCase):
    def test_reader_thread_makes_events_visible_to_poll(self):
        pad = FakeGamepad(batches=[[FakeEvent("Key", "BTN_SOUTH", 1)]])
        src = _InputsInputSource(pad)
        self.addCleanup(src.close)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if src.poll().vert == "jump":
                break
            time.sleep(0.005)
        self.assertEqual(src.poll().vert, "jump")

    def test_poll_does_not_block_when_the_device_is_silent(self):
        """The recorder samples on a frame clock. A blocking poll would slide
        actions onto later frames than they were made on -- the exact
        frame-action misalignment the replay format exists to prevent."""
        src = _InputsInputSource(FakeGamepad())
        self.addCleanup(src.close)
        start = time.monotonic()
        for _ in range(50):
            src.poll()
        self.assertLess(time.monotonic() - start, 0.5)

    def test_a_dead_reader_fails_loudly_instead_of_serving_stale_state(self):
        pad = FakeGamepad(
            batches=[[FakeEvent("Key", "BTN_SOUTH", 1)]], then_raise=OSError("unplugged")
        )
        src = _InputsInputSource(pad)
        self.addCleanup(src.close)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and src._error is None:
            time.sleep(0.005)
        with self.assertRaises(RuntimeError):
            src.poll()

    def test_closed_source_rejects_further_polls(self):
        src = _InputsInputSource(FakeGamepad(), start_reader=False)
        src.close()
        with self.assertRaises(RuntimeError):
            src.poll()

    def test_close_closes_the_underlying_device(self):
        pad = FakeGamepad()
        src = _InputsInputSource(pad, start_reader=False)
        src.close()
        self.assertTrue(pad.closed)


class TestBackendSelection(unittest.TestCase):
    def setUp(self):
        from cuphead.control import human_input

        self.mod = human_input
        self.calls = []
        self._real_win = human_input._open_inputs_gamepad_source
        self._real_lin = human_input._open_evdev_gamepad_source
        human_input._open_inputs_gamepad_source = lambda: self.calls.append("windows")
        human_input._open_evdev_gamepad_source = lambda **kw: self.calls.append("linux")

    def tearDown(self):
        self.mod._open_inputs_gamepad_source = self._real_win
        self.mod._open_evdev_gamepad_source = self._real_lin

    def _with_platform(self, name):
        """Swap the module's view of ``platform``, not the stdlib module itself."""
        real = self.mod.platform
        self.mod.platform = type("FakePlatform", (), {"system": staticmethod(lambda: name)})
        self.addCleanup(lambda: setattr(self.mod, "platform", real))

    def test_windows_selects_the_inputs_backend(self):
        self._with_platform("Windows")
        open_gamepad_source()
        self.assertEqual(self.calls, ["windows"])

    def test_linux_selects_the_evdev_backend(self):
        self._with_platform("Linux")
        open_gamepad_source()
        self.assertEqual(self.calls, ["linux"])

    def test_explicit_backend_overrides_the_platform_sniff(self):
        self._with_platform("Linux")
        open_gamepad_source(backend="windows")
        self.assertEqual(self.calls, ["windows"])

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            open_gamepad_source(backend="playstation")


class TestModuleImportsWithoutPlatformPackages(unittest.TestCase):
    def test_neither_evdev_nor_inputs_is_bound_at_module_level(self):
        """The lazy-import contract: importing control.human_input on a machine
        with neither package installed must work, or the stdlib-only test suite
        and preflight stop running. Both names must stay local to their own
        factory function -- this test is the tripwire on that.
        """
        from cuphead.control import human_input

        self.assertFalse(hasattr(human_input, "evdev"))
        self.assertFalse(hasattr(human_input, "inputs"))
        self.assertNotIn(
            "evdev", human_input._open_inputs_gamepad_source.__code__.co_names
        )
        self.assertIn("inputs", human_input._open_inputs_gamepad_source.__code__.co_names)


if __name__ == "__main__":
    unittest.main()
