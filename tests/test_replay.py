"""The replay format: round-trip fidelity and frame-action alignment.

Directly exercises the acceptance criterion for task `harness-replay-format`:
a 1000-frame round trip with zero differences, and a rejected 1-frame
misalignment.
"""

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.control.action_space import Action, enumerate_actions  # noqa: E402
from cuphead.events.schema import Event, EventKind, EventTrace  # noqa: E402
from cuphead.memory.replay import (  # noqa: E402
    ReplayAlignmentError,
    ReplayError,
    ReplayReader,
    ReplayWriter,
    read_replay,
    write_replay,
)


def sample_events(frame_count: int) -> EventTrace:
    return EventTrace(
        boss="goopy_le_grande",
        events=[
            Event(EventKind.PHASE_ENTER, 0, {"phase": "p1"}),
            Event(EventKind.HIT_TAKEN, frame_count // 2, {"hp_before": 3, "hp_after": 2}),
            Event(EventKind.KNOCKOUT, frame_count - 1, {}),
        ],
    )


class TestRoundTrip(unittest.TestCase):
    def test_1000_frame_round_trip_has_zero_differences(self):
        actions = enumerate_actions()
        n = 1000
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frames_and_actions = [(i, actions[i % len(actions)].to_buttons()) for i in range(n)]
            hud = [(i, {"hp": 3, "super_cards": 1.5, "weapon": "peashooter"}) for i in range(0, n, 100)]
            write_replay(
                root,
                run_id="run_00001",
                boss="goopy_le_grande",
                frames_and_actions=frames_and_actions,
                events=sample_events(n),
                outcome="KNOCKOUT",
                loadout={"weapon": "peashooter", "charm": "smoke_bomb"},
                game_build="1.3.4",
                fps=60.0,
                hud_samples=hud,
            )

            replay = read_replay(root, "run_00001")

            self.assertEqual(replay.meta.boss, "goopy_le_grande")
            self.assertEqual(replay.meta.loadout, {"weapon": "peashooter", "charm": "smoke_bomb"})
            self.assertEqual(replay.meta.game_build, "1.3.4")
            self.assertEqual(replay.meta.outcome, "KNOCKOUT")
            self.assertEqual(replay.meta.frame_count, n)

            self.assertEqual(len(replay.actions), n)
            for i, entry in enumerate(replay.actions):
                self.assertEqual(entry["frame"], i)
                self.assertEqual(entry["action"], actions[i % len(actions)].to_buttons())

            self.assertEqual(len(replay.hud), len(hud))
            self.assertEqual(replay.events.boss, "goopy_le_grande")
            self.assertEqual(replay.events.outcome, "KNOCKOUT")
            self.assertEqual(replay.events.hits_taken, 1)

    def test_writer_and_reader_agree_without_the_convenience_wrapper(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            w = ReplayWriter(root, run_id="r1", boss="ribby_and_croaks", fps=60.0)
            for i in range(50):
                w.log_action(i, Action().to_buttons())
            w.finalize(events=sample_events(50), outcome="KNOCKOUT", frame_count=50)

            replay = ReplayReader(root, "r1").read()
            self.assertEqual(len(replay.actions), 50)


class TestAlignment(unittest.TestCase):
    def test_a_missing_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frames_and_actions = [(i, Action().to_buttons()) for i in range(50) if i != 17]
            with self.assertRaises(ReplayAlignmentError):
                write_replay(
                    root,
                    run_id="bad",
                    boss="b",
                    frames_and_actions=frames_and_actions,
                    events=sample_events(50),
                    outcome="KNOCKOUT",
                )

    def test_a_duplicated_frame_within_one_writer_session_is_rejected_immediately(self):
        with tempfile.TemporaryDirectory() as d:
            w = ReplayWriter(Path(d), run_id="dup", boss="b")
            w.log_action(0, Action().to_buttons())
            with self.assertRaises(ReplayAlignmentError):
                w.log_action(0, Action().to_buttons())
            w.abandon()

    def test_a_one_frame_misalignment_injected_after_writing_is_caught_on_read(self):
        """The exact scenario named in the task's acceptance criterion:
        a 1-frame action misalignment must be caught by the loader."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_replay(
                root,
                run_id="shift",
                boss="b",
                frames_and_actions=[(i, Action().to_buttons()) for i in range(20)],
                events=sample_events(20),
                outcome="KNOCKOUT",
            )
            # Corrupt the on-disk log: shift every action's frame index by 1,
            # simulating the exact failure mode the alignment check exists for.
            actions_path = root / "shift" / "actions.jsonl"
            import json

            lines = [json.loads(l) for l in actions_path.read_text().splitlines()]
            for entry in lines:
                entry["frame"] += 1
            actions_path.write_text("\n".join(json.dumps(e) for e in lines) + "\n")

            with self.assertRaises(ReplayAlignmentError):
                read_replay(root, "shift")

    def test_reading_an_unfinalized_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            w = ReplayWriter(Path(d), run_id="partial", boss="b")
            w.log_action(0, Action().to_buttons())
            w.abandon()
            with self.assertRaises(ReplayError):
                read_replay(Path(d), "partial")

    def test_zero_frame_replay_is_not_treated_as_misaligned(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_replay(
                root, run_id="empty", boss="b", frames_and_actions=[],
                events=EventTrace(boss="b", events=[]), outcome="INCOMPLETE",
            )
            replay = read_replay(root, "empty")
            self.assertEqual(replay.meta.frame_count, 0)
            self.assertEqual(replay.actions, [])

    def test_finalizing_twice_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            w = ReplayWriter(Path(d), run_id="r", boss="b")
            w.finalize(events=sample_events(0), outcome="INCOMPLETE", frame_count=0)
            with self.assertRaises(ReplayError):
                w.finalize(events=sample_events(0), outcome="INCOMPLETE", frame_count=0)


if __name__ == "__main__":
    unittest.main()
