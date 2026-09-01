"""The risk-constrained cost function and its reward-hacking guards."""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.planner.objective import (  # noqa: E402
    LAMBDA_CRITICAL,
    LAMBDA_HEALTHY,
    PhasePrior,
    risk_weight,
    stalling_detected,
    step_reward,
    trajectory_value,
)


class TestRiskWeight(unittest.TestCase):
    def test_low_health_costs_more_than_full_health(self):
        self.assertGreater(risk_weight(1), risk_weight(3))

    def test_endpoints_match_the_declared_constants(self):
        self.assertAlmostEqual(risk_weight(3), LAMBDA_HEALTHY)
        self.assertAlmostEqual(risk_weight(0), LAMBDA_CRITICAL)

    def test_late_phase_raises_the_cost_of_the_same_hit(self):
        """A death at 90% through the fight throws away more invested time."""
        self.assertGreater(risk_weight(2, phase_progress=0.9), risk_weight(2, phase_progress=0.05))

    def test_monotonic_in_health(self):
        weights = [risk_weight(hp) for hp in range(4)]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_health_is_clamped_not_extrapolated(self):
        self.assertEqual(risk_weight(9, max_hp=3), risk_weight(3, max_hp=3))
        self.assertEqual(risk_weight(-4, max_hp=3), risk_weight(0, max_hp=3))

    def test_zero_max_hp_is_rejected(self):
        with self.assertRaises(ValueError):
            risk_weight(1, max_hp=0)


class TestStepReward(unittest.TestCase):
    def test_damage_is_rewarded_and_risk_is_penalized(self):
        safe_damage = step_reward(dps_uptime=1.0, p_hit=0.0, hp=3)
        risky_damage = step_reward(dps_uptime=1.0, p_hit=0.5, hp=3)
        self.assertGreater(safe_damage, risky_damage)

    def test_at_full_health_aggression_can_be_worth_a_hit_risk(self):
        """Survival is the constraint, not the objective."""
        aggressive = step_reward(dps_uptime=1.0, p_hit=0.15, hp=3)
        passive = step_reward(dps_uptime=0.0, p_hit=0.0, hp=3)
        self.assertGreater(aggressive, passive)

    def test_at_one_heart_the_same_risk_is_no_longer_worth_it(self):
        aggressive = step_reward(dps_uptime=1.0, p_hit=0.15, hp=1)
        passive = step_reward(dps_uptime=0.0, p_hit=0.0, hp=1)
        self.assertLess(aggressive, passive)

    def test_doing_nothing_is_always_negative(self):
        """The time penalty is what stops the planner from learning to wait."""
        self.assertLess(step_reward(dps_uptime=0.0, p_hit=0.0, hp=3), 0.0)

    def test_aggression_prior_is_clamped(self):
        wild = step_reward(1.0, 0.1, 3, aggression=1000.0)
        capped = step_reward(1.0, 0.1, 3, aggression=2.0)
        self.assertAlmostEqual(wild, capped)


class TestTrajectoryValue(unittest.TestCase):
    def test_mismatched_sequence_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            trajectory_value([1.0, 1.0], [0.0], hp=3)

    def test_discounting_reduces_the_weight_of_later_steps(self):
        front = trajectory_value([1.0, 0.0], [0.0, 0.0], hp=3)
        back = trajectory_value([0.0, 1.0], [0.0, 0.0], hp=3)
        self.assertGreater(front, back)

    def test_terminal_value_bootstraps_past_the_horizon(self):
        without = trajectory_value([0.5] * 12, [0.0] * 12, hp=3)
        with_v = trajectory_value([0.5] * 12, [0.0] * 12, hp=3, terminal_value=10.0)
        self.assertGreater(with_v, without)

    def test_hoarding_supers_is_penalized(self):
        spent = trajectory_value([1.0] * 5, [0.0] * 5, hp=3, unspent_cards=0)
        hoarded = trajectory_value([1.0] * 5, [0.0] * 5, hp=3, unspent_cards=3)
        self.assertGreater(spent, hoarded)


class TestStallingDetector(unittest.TestCase):
    def test_fires_on_low_damage_and_a_long_attempt(self):
        self.assertTrue(stalling_detected(dps_uptime=0.1, ttk_s=60.0, baseline_ttk_s=40.0))

    def test_does_not_fire_on_a_fast_aggressive_attempt(self):
        self.assertFalse(stalling_detected(dps_uptime=0.8, ttk_s=35.0, baseline_ttk_s=40.0))

    def test_low_uptime_alone_is_not_enough(self):
        """Either signal alone is noise; together they are the signature."""
        self.assertFalse(stalling_detected(dps_uptime=0.1, ttk_s=38.0, baseline_ttk_s=40.0))

    def test_a_long_attempt_with_good_uptime_is_not_stalling(self):
        self.assertFalse(stalling_detected(dps_uptime=0.9, ttk_s=70.0, baseline_ttk_s=40.0))


class TestPhasePrior(unittest.TestCase):
    def test_out_of_range_llm_output_is_clamped_not_trusted(self):
        clamped = PhasePrior(phase="p2", aggression=50.0, lambda_scale=0.0).clamped()
        self.assertLessEqual(clamped.aggression, 2.0)
        self.assertGreaterEqual(clamped.lambda_scale, 0.25)

    def test_nonsense_enum_values_fall_back_to_defaults(self):
        clamped = PhasePrior(phase="p2", preferred_lane="the_moon", parry_priority="maximum").clamped()
        self.assertEqual(clamped.preferred_lane, "none")
        self.assertEqual(clamped.parry_priority, "normal")

    def test_valid_prior_survives_clamping_unchanged(self):
        prior = PhasePrior(phase="p2", aggression=0.8, preferred_lane="left", parry_priority="high")
        self.assertEqual(prior.clamped(), prior)


if __name__ == "__main__":
    unittest.main()
