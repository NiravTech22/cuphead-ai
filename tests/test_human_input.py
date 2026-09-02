"""Action.from_raw normalization and the scripted human-input source."""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.control.action_space import Action, STICK_DEADZONE, enumerate_actions  # noqa: E402
from cuphead.control.human_input import ScriptedInputSource  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
