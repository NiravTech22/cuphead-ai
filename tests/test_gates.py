"""The statistical gate, tested against distributions with known separation."""

import random
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.orchestration.gates import (  # noqa: E402
    ArmResult,
    bootstrap_median_diff_ci,
    evaluate,
    format_verdict,
)


def arm(name, center, n=40, spread=1.5, deaths=0, seed=1):
    rng = random.Random(seed)
    return ArmResult(
        name=name,
        ttk_s=[rng.gauss(center, spread) for _ in range(n)],
        deaths=deaths,
        attempts=n + deaths,
    )


class TestBootstrap(unittest.TestCase):
    def test_clear_improvement_gives_a_ci_below_zero(self):
        lo, hi = bootstrap_median_diff_ci(
            arm("b", 40.0, seed=1).ttk_s, arm("c", 35.0, seed=2).ttk_s, samples=2000
        )
        self.assertLess(hi, 0.0)
        self.assertLess(lo, hi)

    def test_identical_distributions_give_a_ci_spanning_zero(self):
        lo, hi = bootstrap_median_diff_ci(
            arm("b", 40.0, seed=3).ttk_s, arm("c", 40.0, seed=4).ttk_s, samples=2000
        )
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_is_deterministic_for_a_fixed_seed(self):
        a, b = arm("b", 40.0, seed=5).ttk_s, arm("c", 36.0, seed=6).ttk_s
        first = bootstrap_median_diff_ci(a, b, samples=500, seed=7)
        second = bootstrap_median_diff_ci(a, b, samples=500, seed=7)
        self.assertEqual(first, second)

    def test_empty_arm_is_an_error_not_a_silent_pass(self):
        with self.assertRaises(ValueError):
            bootstrap_median_diff_ci([], [1.0, 2.0])


class TestGate(unittest.TestCase):
    def test_real_improvement_passes(self):
        v = evaluate(arm("base", 42.0, seed=11), arm("cand", 36.0, seed=12), bootstrap_samples=2000)
        self.assertTrue(v.passed, v.reasons)
        self.assertEqual(v.verdict, "SUCCESSFUL")

    def test_noise_does_not_pass(self):
        v = evaluate(arm("base", 40.0, seed=13), arm("cand", 39.9, seed=14), bootstrap_samples=2000)
        self.assertFalse(v.passed)

    def test_underpowered_comparison_is_rejected(self):
        v = evaluate(
            arm("base", 42.0, n=8, seed=15),
            arm("cand", 30.0, n=8, seed=16),
            bootstrap_samples=1000,
        )
        self.assertFalse(v.passed)
        self.assertTrue(any("below the minimum" in r for r in v.reasons))

    def test_death_rate_regression_fails_a_faster_agent(self):
        """Speed bought with deaths is not an improvement."""
        v = evaluate(
            arm("base", 42.0, seed=17, deaths=1),
            arm("cand", 34.0, seed=18, deaths=20),
            bootstrap_samples=2000,
        )
        self.assertFalse(v.passed)
        self.assertTrue(any("death rate regressed" in r for r in v.reasons))

    def test_stalling_fails_regardless_of_ttk(self):
        v = evaluate(
            arm("base", 42.0, seed=19),
            arm("cand", 34.0, seed=20),
            bootstrap_samples=2000,
            stalling_detected=True,
        )
        self.assertFalse(v.passed)
        self.assertTrue(any("stalling" in r for r in v.reasons))

    def test_latency_overrun_fails_even_when_faster(self):
        cand = arm("cand", 34.0, seed=21)
        cand.latency_p99_ms = 18.0
        v = evaluate(
            arm("base", 42.0, seed=22), cand, bootstrap_samples=2000, latency_budget_ms=10.0
        )
        self.assertFalse(v.passed)
        self.assertTrue(any("latency canary" in r for r in v.reasons))

    def test_all_failures_are_reported_not_just_the_first(self):
        v = evaluate(
            arm("base", 40.0, n=5, seed=23),
            arm("cand", 41.0, n=5, seed=24),
            bootstrap_samples=500,
        )
        self.assertGreaterEqual(len(v.reasons), 2)

    def test_format_verdict_includes_n_and_ci(self):
        v = evaluate(arm("base", 42.0, seed=25), arm("cand", 36.0, seed=26), bootstrap_samples=1000)
        text = format_verdict(v)
        self.assertIn("median TTK", text)
        self.assertIn("bootstrap CI", text)
        self.assertIn("n ", text)


class TestArmResult(unittest.TestCase):
    def test_death_rate_counts_deaths_against_all_attempts(self):
        a = ArmResult("a", ttk_s=[10.0] * 8, deaths=2, attempts=10)
        self.assertAlmostEqual(a.death_rate, 0.2)

    def test_p10_is_the_fast_tail(self):
        a = ArmResult("a", ttk_s=[float(x) for x in range(1, 11)], attempts=10)
        self.assertLessEqual(a.p10_ttk, a.median_ttk)

    def test_empty_arm_has_infinite_median(self):
        self.assertEqual(ArmResult("a").median_ttk, float("inf"))


if __name__ == "__main__":
    unittest.main()
