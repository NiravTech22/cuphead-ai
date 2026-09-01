"""Event traces and the metrics derived from them."""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.evaluation.metrics import AttemptMetrics, aggregate, per_boss  # noqa: E402
from cuphead.events.schema import Event, EventKind, EventTrace  # noqa: E402


def ev(kind, frame, **data):
    return Event(kind=kind, frame=frame, data=data)


def knockout(boss="goopy_le_grande", ttk_frames=2400, hits=1):
    events = [ev(EventKind.PHASE_ENTER, 0, phase="p1")]
    events += [ev(EventKind.HIT_TAKEN, 300 + i * 100, hp_before=3 - i, hp_after=2 - i) for i in range(hits)]
    events += [
        ev(EventKind.DPS_WINDOW, 100, state="open"),
        ev(EventKind.DPS_WINDOW, ttk_frames - 100, state="close"),
        ev(EventKind.KNOCKOUT, ttk_frames, time_s=ttk_frames / 60.0),
    ]
    return EventTrace(boss=boss, events=events)


def death(boss="goopy_le_grande", at=900):
    return EventTrace(
        boss=boss,
        events=[ev(EventKind.PHASE_ENTER, 0, phase="p1"), ev(EventKind.DEATH, at, phase_id="p1")],
    )


class TestEventTrace(unittest.TestCase):
    def test_ttk_is_the_knockout_frame_in_seconds(self):
        self.assertAlmostEqual(knockout(ttk_frames=1800).ttk_s, 30.0)

    def test_a_death_has_no_finite_ttk(self):
        self.assertEqual(death().ttk_s, float("inf"))
        self.assertEqual(death().outcome, "DEATH")

    def test_incomplete_trace_is_reported_as_such(self):
        self.assertEqual(EventTrace(boss="b", events=[ev(EventKind.PHASE_ENTER, 0)]).outcome, "INCOMPLETE")

    def test_hits_taken_counts_hit_events(self):
        self.assertEqual(knockout(hits=2).hits_taken, 2)

    def test_dps_uptime_is_a_fraction_of_the_attempt(self):
        uptime = knockout(ttk_frames=2400).dps_uptime()
        self.assertGreater(uptime, 0.8)
        self.assertLessEqual(uptime, 1.0)

    def test_unclosed_dps_window_runs_to_the_last_frame(self):
        """A window still open at the knockout was open until the knockout."""
        trace = EventTrace(
            boss="b",
            events=[ev(EventKind.DPS_WINDOW, 0, state="open"), ev(EventKind.KNOCKOUT, 600)],
        )
        self.assertAlmostEqual(trace.dps_uptime(), 1.0)

    def test_no_dps_windows_means_zero_uptime(self):
        self.assertEqual(death().dps_uptime(), 0.0)

    def test_parry_conversion_is_successes_over_opportunities(self):
        trace = EventTrace(
            boss="b",
            events=[
                ev(EventKind.PARRY, 10, result="success"),
                ev(EventKind.PARRY, 20, result="miss"),
                ev(EventKind.PARRY, 30, result="success"),
                ev(EventKind.KNOCKOUT, 100),
            ],
        )
        self.assertAlmostEqual(trace.parry_conversion, 2 / 3)

    def test_trace_round_trips_through_dict(self):
        original = knockout(hits=2)
        restored = EventTrace.from_dict(original.to_dict())
        self.assertEqual(restored.boss, original.boss)
        self.assertEqual(len(restored.events), len(original.events))
        self.assertAlmostEqual(restored.ttk_s, original.ttk_s)


class TestAggregation(unittest.TestCase):
    def test_deaths_are_counted_but_excluded_from_the_ttk_sample(self):
        """Imputing a TTK for a death would corrupt the median the gate rests on."""
        traces = [knockout(ttk_frames=2400) for _ in range(4)] + [death()]
        arm = aggregate("a", traces)
        self.assertEqual(arm.attempts, 5)
        self.assertEqual(arm.deaths, 1)
        self.assertEqual(len(arm.ttk_s), 4)
        self.assertAlmostEqual(arm.death_rate, 0.2)
        self.assertTrue(all(t != float("inf") for t in arm.ttk_s))

    def test_median_ignores_the_death(self):
        arm = aggregate("a", [knockout(ttk_frames=2400), knockout(ttk_frames=2400), death()])
        self.assertAlmostEqual(arm.median_ttk, 40.0)

    def test_per_boss_split_keeps_bosses_separate(self):
        traces = [knockout(boss="goopy"), knockout(boss="ribby"), death(boss="ribby")]
        split = per_boss(traces)
        self.assertEqual(sorted(split), ["goopy", "ribby"])
        self.assertEqual(split["goopy"].attempts, 1)
        self.assertEqual(split["ribby"].deaths, 1)

    def test_attempt_metrics_from_trace(self):
        m = AttemptMetrics.from_trace(knockout(ttk_frames=1800, hits=2))
        self.assertEqual(m.outcome, "KNOCKOUT")
        self.assertEqual(m.hits_taken, 2)
        self.assertAlmostEqual(m.ttk_s, 30.0)

    def test_empty_arm_does_not_crash(self):
        arm = aggregate("empty", [])
        self.assertEqual(arm.attempts, 0)
        self.assertEqual(arm.death_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
