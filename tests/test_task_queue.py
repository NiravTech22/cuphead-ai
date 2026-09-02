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

    def test_implemented_tasks_are_not_reselected(self):
        """IMPLEMENTED means done and waiting on qa -- not "hand to a worker
        again." Regression test for a bug where a loop restart between the
        worker finishing and qa running would silently re-dispatch the task."""
        tasks = load_tasks({"tasks": [task("a", status="IMPLEMENTED"), task("b")]})
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

    def test_refresh_blocked_leaves_implemented_tasks_alone(self):
        """A task awaiting qa must survive refresh_blocked untouched, or a
        loop restart mid-pipeline silently reverts it to OPEN and it gets
        re-implemented instead of validated."""
        doc = {"tasks": [task("a", status="IMPLEMENTED")]}
        refresh_blocked(doc, state("PENDING"))
        self.assertEqual(doc["tasks"][0]["status"], "IMPLEMENTED")

    def test_refresh_blocked_leaves_in_progress_tasks_alone(self):
        doc = {"tasks": [task("a", status="IN_PROGRESS")]}
        refresh_blocked(doc, state("PENDING"))
        self.assertEqual(doc["tasks"][0]["status"], "IN_PROGRESS")


class TestShippedQueue(unittest.TestCase):
    """The checked-in queue must actually be runnable."""

    def _load(self):
        import json

        tasks_doc = json.loads((REPO / ".ai" / "tasks.json").read_text())
        project = json.loads((REPO / ".ai" / "project_state.json").read_text())
        return tasks_doc, project

    def test_no_task_is_selectable_while_all_phase_0_harness_work_awaits_qa(self):
        """As shipped, every phase-0 harness task is IMPLEMENTED (code done,
        awaiting qa) and nothing downstream is unblocked yet -- so the queue
        correctly has no work for a worker to pick up right now. This is the
        real current state, not a bug: the next actor is qa, not a worker."""
        tasks_doc, project = self._load()
        self.assertIsNone(select_next(tasks_doc, project))

    def test_every_phase_0_harness_task_is_at_least_implemented(self):
        tasks_doc, _ = self._load()
        harness_ids = {
            "harness-capture-latency",
            "harness-frame-integrity",
            "harness-replay-format",
            "harness-human-demo-recorder",
        }
        by_id = {t["id"]: t["status"] for t in tasks_doc["tasks"]}
        for tid in harness_ids:
            self.assertIn(by_id[tid], ("IMPLEMENTED", "VALIDATED", "SUCCESSFUL"), tid)


if __name__ == "__main__":
    unittest.main()
