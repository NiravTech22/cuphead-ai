"""Round-trip latency measurement, validated against a known injected delay."""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.control.action_space import Action  # noqa: E402
from cuphead.control.actuator import RecordingActuator  # noqa: E402
from cuphead.evaluation.latency import (  # noqa: E402
    FakeClock,
    LatencyTimeoutError,
    NotifyingActuator,
    RespondingFrameSource,
    measure_many,
    measure_once,
    percentile,
    synthetic_round_trip,
)
from cuphead.perception.capture import SyntheticFrameSource  # noqa: E402


class TestPercentile(unittest.TestCase):
    def test_median_of_an_odd_sample(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 50), 2.0)

    def test_p99_of_uniform_1_to_100(self):
        samples = [float(x) for x in range(1, 101)]
        self.assertAlmostEqual(percentile(samples, 99), 99.01, places=2)

    def test_single_sample_returns_itself_at_any_percentile(self):
        self.assertEqual(percentile([42.0], 0), 42.0)
        self.assertEqual(percentile([42.0], 100), 42.0)

    def test_empty_samples_is_an_error(self):
        with self.assertRaises(ValueError):
            percentile([], 50)

    def test_out_of_range_percentile_is_rejected(self):
        with self.assertRaises(ValueError):
            percentile([1.0, 2.0], 101)


class TestMeasureOnce(unittest.TestCase):
    def test_recovers_a_known_injected_delay_in_frame_counts(self):
        """The core self-test: a fixed 3-frame response delay at 60 fps
        must be recovered as ~50 ms (3 * 1/60 s), not some arbitrary value."""
        actuator, source = synthetic_round_trip(respond_after_frames=3, fps=60.0)
        latency_ms = measure_once(actuator, source, Action(move="right"))
        self.assertAlmostEqual(latency_ms, 3 * (1000.0 / 60.0), delta=0.01)

    def test_one_frame_delay_is_the_minimum_measurable_latency(self):
        actuator, source = synthetic_round_trip(respond_after_frames=1, fps=60.0)
        latency_ms = measure_once(actuator, source, Action())
        self.assertAlmostEqual(latency_ms, 1000.0 / 60.0, delta=0.01)

    def test_times_out_when_the_source_never_responds(self):
        clock = FakeClock(dt=1.0 / 60.0)
        source = RespondingFrameSource(respond_after_frames=1_000_000, clock=clock)
        actuator = NotifyingActuator(source, clock)
        with self.assertRaises(LatencyTimeoutError):
            measure_once(actuator, source, Action(), max_frames=5)

    def test_a_source_that_never_changes_at_all_also_times_out(self):
        """RecordingActuator paired with a source that ignores it entirely --
        the exact failure mode of a misconfigured real backend."""
        actuator = RecordingActuator()
        source = SyntheticFrameSource(duplicate_at=frozenset(range(1, 20)))
        with self.assertRaises(LatencyTimeoutError):
            measure_once(actuator, source, Action(), max_frames=10)


class TestMeasureMany(unittest.TestCase):
    def test_reports_percentiles_matching_the_fixed_injected_delay(self):
        actuator, source = synthetic_round_trip(respond_after_frames=2, fps=60.0)
        actions = [Action(move="left"), Action(move="right")] * 100
        report = measure_many(actuator, source, actions)
        expected = 2 * (1000.0 / 60.0)
        self.assertEqual(report.n, 200)
        self.assertAlmostEqual(report.p50_ms, expected, delta=0.01)
        self.assertAlmostEqual(report.p99_ms, expected, delta=0.01)
        self.assertEqual(len(report.samples_ms), 200)

    def test_within_budget_is_strict_on_both_bounds(self):
        actuator, source = synthetic_round_trip(respond_after_frames=1, fps=60.0)
        report = measure_many(actuator, source, [Action()] * 30)
        self.assertTrue(report.within_budget(p50_ms=30.0, p99_ms=50.0))
        self.assertFalse(report.within_budget(p50_ms=0.001, p99_ms=50.0))

    def test_to_dict_round_trips_through_json(self):
        import json

        actuator, source = synthetic_round_trip(respond_after_frames=1, fps=60.0)
        report = measure_many(actuator, source, [Action()] * 5)
        parsed = json.loads(json.dumps(report.to_dict()))
        self.assertEqual(parsed["n"], 5)
        self.assertEqual(len(parsed["samples_ms"]), 5)


if __name__ == "__main__":
    unittest.main()
