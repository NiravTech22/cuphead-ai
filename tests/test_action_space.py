"""Action space pruning and reflex override arbitration."""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.control.action_space import (  # noqa: E402
    ACTION_REPEAT,
    DECISION_HZ,
    Action,
    ReflexOverride,
    action_count,
    enumerate_actions,
    index_of,
)


class TestActionSpace(unittest.TestCase):
    def test_pruning_is_a_large_reduction_of_the_raw_product(self):
        raw = 3 * 3 * 2 * 2 * 2 * 8  # 576
        self.assertLess(action_count(), raw / 4)
        self.assertGreater(action_count(), 20)

    def test_every_enumerated_action_is_legal(self):
        self.assertTrue(all(a.is_legal() for a in enumerate_actions()))

    def test_enumeration_is_stable_and_unique(self):
        actions = enumerate_actions()
        self.assertEqual(len(set(actions)), len(actions))
        self.assertEqual(enumerate_actions(), actions)

    def test_index_round_trips(self):
        a = enumerate_actions()[7]
        self.assertIs(enumerate_actions()[index_of(a)], a)

    def test_dash_while_ducking_is_pruned(self):
        self.assertFalse(Action(dash=True, vert="duck").is_legal())

    def test_air_dash_is_kept(self):
        self.assertTrue(Action(dash=True, vert="jump").is_legal())

    def test_aim_without_lock_is_pruned(self):
        self.assertFalse(Action(lock=False, aim="n").is_legal())
        self.assertTrue(Action(lock=True, shoot=True, aim="n").is_legal())

    def test_aim_lock_roots_the_player(self):
        self.assertFalse(Action(move="left", lock=True, shoot=True).is_legal())

    def test_dash_cancels_the_shot(self):
        self.assertFalse(Action(dash=True, shoot=True).is_legal())

    def test_idle_aim_lock_is_pruned(self):
        """Holding lock without firing or dashing does nothing at all."""
        self.assertFalse(Action(lock=True, shoot=False, aim="n").is_legal())

    def test_invalid_field_values_are_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            Action(move="backwards")
        with self.assertRaises(ValueError):
            Action(aim="up")

    def test_decision_rate_matches_action_repeat(self):
        self.assertEqual(ACTION_REPEAT, 4)
        self.assertAlmostEqual(DECISION_HZ, 15.0)


class TestButtonLowering(unittest.TestCase):
    def test_movement_maps_to_the_stick(self):
        self.assertEqual(Action(move="left").to_buttons()["stick_x"], -1.0)
        self.assertEqual(Action(move="right").to_buttons()["stick_x"], 1.0)

    def test_lock_overrides_the_stick_with_the_aim_vector(self):
        b = Action(move="left", lock=True, aim="n").to_buttons()
        self.assertEqual((b["stick_x"], b["stick_y"]), (0.0, 1.0))

    def test_jump_and_shoot_map_to_buttons(self):
        b = Action(vert="jump", shoot=True).to_buttons()
        self.assertTrue(b["a"])
        self.assertTrue(b["x"])


class TestReflexOverride(unittest.TestCase):
    def test_override_requires_confidence_above_threshold(self):
        self.assertTrue(ReflexOverride("parry", 0.95).applies(0.9))
        self.assertFalse(ReflexOverride("parry", 0.5).applies(0.9))

    def test_non_whitelisted_kinds_never_apply(self):
        self.assertFalse(ReflexOverride("fire_super", 1.0).applies(0.0))

    def test_parry_preserves_the_planner_movement_intent(self):
        """The reflex decides the next 90 ms; the planner decided where to be."""
        planned = Action(move="right", shoot=True)
        merged = ReflexOverride("parry", 0.99).merge(planned)
        self.assertEqual(merged.move, "right")
        self.assertEqual(merged.vert, "jump")
        self.assertTrue(merged.shoot)

    def test_overrides_drop_aim_lock_because_it_roots_the_player(self):
        planned = Action(move="none", shoot=True, lock=True, aim="ne")
        self.assertFalse(ReflexOverride("parry", 0.99).merge(planned).lock)
        self.assertFalse(ReflexOverride("dash", 0.99).merge(planned).lock)

    def test_dash_override_sets_dash_and_clears_vertical(self):
        merged = ReflexOverride("dash", 0.99).merge(Action(move="left", vert="jump"))
        self.assertTrue(merged.dash)
        self.assertEqual(merged.vert, "none")
        self.assertEqual(merged.move, "left")

    def test_merged_override_is_still_a_legal_action(self):
        for planned in enumerate_actions():
            for kind in ("parry", "dash"):
                self.assertTrue(ReflexOverride(kind, 1.0).merge(planned).is_legal())


if __name__ == "__main__":
    unittest.main()
