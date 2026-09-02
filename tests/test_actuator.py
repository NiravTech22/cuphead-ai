"""The in-memory actuator used by tests and the latency canary's synthetic mode."""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.control.action_space import Action  # noqa: E402
from cuphead.control.actuator import RecordingActuator  # noqa: E402


class TestRecordingActuator(unittest.TestCase):
    def test_send_records_the_action_and_a_timestamp(self):
        act = RecordingActuator()
        result = act.send(Action(move="right"))
        self.assertEqual(result.action, Action(move="right"))
        self.assertGreater(result.t_sent, 0.0)

    def test_history_accumulates_in_order(self):
        act = RecordingActuator()
        act.send(Action(move="left"))
        act.send(Action(move="right"))
        self.assertEqual([h.action.move for h in act.history], ["left", "right"])

    def test_timestamps_are_non_decreasing(self):
        act = RecordingActuator()
        for _ in range(20):
            act.send(Action())
        stamps = [h.t_sent for h in act.history]
        self.assertEqual(stamps, sorted(stamps))

    def test_close_is_a_no_op(self):
        act = RecordingActuator()
        act.send(Action())
        act.close()  # must not raise
        self.assertEqual(len(act.history), 1)


if __name__ == "__main__":
    unittest.main()
