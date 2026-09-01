"""Task selection: dependency gating and the hard phase order."""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.orchestration.tasks import (  # noqa: E402
    highest_unlocked_phase,
    load_tasks,
    refresh_blocked,
    select_next,
    set_status,
    unblocked,
)


def task(tid, *, phase=0, priority=50, status="OPEN", depends_on=()):
    return {
        "id": tid,
        "title": tid,
        "owner": "vision",
        "priority": priority,
        "status": status,
        "acceptance": "n >= 1",
        "phase": phase,
        "depends_on": list(depends_on),
    }


def state(*phase_statuses):
    return {"phases": [{"id": i, "status": s} for i, s in enumerate(phase_statuses)]}


class TestSelection(unittest.TestCase):
    def test_dependency_blocks_a_task(self):
        tasks = load_tasks({"tasks": [task("a"), task("b", depends_on=["a"])]})
        ready = [t.id for t in unblocked(tasks)]
        self.assertEqual(ready, ["a"])

    def test_closed_dependency_unblocks(self):
        tasks = load_tasks(
            {"tasks": [task("a", status="VALIDATED"), task("b", depends_on=["a"])]}
        )
        self.assertEqual([t.id for t in unblocked(tasks)], ["b"])

    def test_priority_then_id_ordering_is_deterministic(self):
        tasks = load_tasks(
            {"tasks": [task("z", priority=10), task("m", priority=90), task("a", priority=90)]}
        )
        self.assertEqual([t.id for t in unblocked(tasks)], ["a", "m", "z"])

    def test_in_progress_tasks_are_not_reselected(self):
        tasks = load_tasks({"tasks": [task("a", status="IN_PROGRESS"), task("b")]})
        self.assertEqual([t.id for t in unblocked(tasks)], ["b"])


class TestPhaseOrder(unittest.TestCase):
    def test_highest_unlocked_phase_is_the_first_unfinished(self):
        self.assertEqual(highest_unlocked_phase(state("VALIDATED", "PENDING", "PENDING")), 1)
        self.assertEqual(highest_unlocked_phase(state("PENDING", "PENDING")), 0)

    def test_later_phase_blocked_even_with_satisfied_dependencies(self):
        """The build order is a hard constraint, not a hint."""
        tasks = load_tasks({"tasks": [task("early", phase=0), task("late", phase=4, priority=99)]})
        ready = [t.id for t in unblocked(tasks, max_phase=highest_unlocked_phase(state("PENDING"))) ]
        self.assertEqual(ready, ["early"])

    def test_advancing_a_phase_unlocks_the_next(self):
        doc = {"tasks": [task("late", phase=1, priority=99)]}
        self.assertIsNone(select_next(doc, state("PENDING", "PENDING")))
        self.assertEqual(select_next(doc, state("VALIDATED", "PENDING")).id, "late")

    def test_all_phases_done_does_not_crash(self):
        self.assertEqual(highest_unlocked_phase(state("VALIDATED", "SUCCESSFUL")), 1)


class TestMutation(unittest.TestCase):
    def test_set_status_rejects_unknown_status(self):
        doc = {"tasks": [task("a")]}
        with self.assertRaises(ValueError):
            set_status(doc, "a", "MOSTLY_FINE")

    def test_set_status_rejects_unknown_task(self):
        with self.assertRaises(KeyError):
            set_status({"tasks": []}, "ghost", "OPEN")

    def test_refresh_blocked_marks_the_queue_honestly(self):
        doc = {"tasks": [task("a"), task("b", depends_on=["a"], status="OPEN")]}
        refresh_blocked(doc, state("PENDING"))
        by_id = {t["id"]: t["status"] for t in doc["tasks"]}
        self.assertEqual(by_id, {"a": "OPEN", "b": "BLOCKED"})

    def test_refresh_blocked_leaves_closed_tasks_alone(self):
        doc = {"tasks": [task("a", status="VALIDATED")]}
        refresh_blocked(doc, state("PENDING"))
        self.assertEqual(doc["tasks"][0]["status"], "VALIDATED")


class TestShippedQueue(unittest.TestCase):
    """The checked-in queue must actually be runnable."""

    def test_next_task_is_the_harness_latency_canary(self):
        import json

        tasks_doc = json.loads((REPO / ".ai" / "tasks.json").read_text())
        project = json.loads((REPO / ".ai" / "project_state.json").read_text())
        nxt = select_next(tasks_doc, project)
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt.id, "harness-capture-latency")


if __name__ == "__main__":
    unittest.main()
