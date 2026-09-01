"""State store: atomicity, validation, and the real .ai/ documents."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.orchestration.state_store import StateError, StateStore  # noqa: E402


def make_store(tmp: Path) -> StateStore:
    (tmp / ".ai").mkdir()
    docs = {
        "project_state.json": {
            "project": "t",
            "current_phase": {"id": 0},
            "phases": [{"id": 0, "status": "PENDING"}],
            "components": {},
            "gate": {},
        },
        "tasks.json": {"tasks": []},
        "agent_status.json": {"agents": {"vision": {"state": "IDLE"}}, "loop": {}},
        "failures.json": {"failures": []},
    }
    for name, body in docs.items():
        (tmp / ".ai" / name).write_text(json.dumps(body), encoding="utf-8")
    return StateStore(tmp)


class TestRealStateFiles(unittest.TestCase):
    """The checked-in .ai/ documents must always be healthy."""

    def setUp(self):
        self.store = StateStore(REPO)

    def test_all_documents_parse(self):
        for name in ("project_state.json", "tasks.json", "agent_status.json", "failures.json"):
            self.assertIsInstance(self.store.read(name), dict, name)

    def test_validate_reports_no_problems(self):
        self.assertEqual(self.store.validate(), [])

    def test_every_task_owner_is_a_real_agent(self):
        agents = set(self.store.agent_status()["agents"])
        for task in self.store.tasks()["tasks"]:
            self.assertIn(task["owner"], agents, task["id"])

    def test_every_task_has_a_measurable_acceptance(self):
        for task in self.store.tasks()["tasks"]:
            self.assertTrue(task["acceptance"].strip(), task["id"])
            # An acceptance criterion without a number or a comparison is an
            # adjective in disguise, and QA cannot run it.
            self.assertTrue(
                any(ch.isdigit() for ch in task["acceptance"]),
                f"{task['id']} acceptance has no measurable threshold",
            )


class TestStateStoreMechanics(unittest.TestCase):
    def test_write_is_atomic_and_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            store = make_store(Path(d))
            store.write("tasks.json", {"tasks": [{"id": "a"}]})
            self.assertEqual(store.tasks()["tasks"][0]["id"], "a")
            # No temp files left behind.
            leftovers = [p.name for p in (Path(d) / ".ai").iterdir() if p.name.endswith(".tmp")]
            self.assertEqual(leftovers, [])

    def test_unknown_state_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            store = make_store(Path(d))
            with self.assertRaises(StateError):
                store.read("secrets.json")

    def test_malformed_json_is_reported_not_raised_bare(self):
        with tempfile.TemporaryDirectory() as d:
            store = make_store(Path(d))
            (Path(d) / ".ai" / "tasks.json").write_text("{not json", encoding="utf-8")
            problems = store.validate()
            self.assertTrue(any("not valid JSON" in p for p in problems))

    def test_validate_catches_unknown_dependency(self):
        with tempfile.TemporaryDirectory() as d:
            store = make_store(Path(d))
            store.write(
                "tasks.json",
                {
                    "tasks": [
                        {
                            "id": "a",
                            "title": "t",
                            "owner": "vision",
                            "priority": 1,
                            "status": "OPEN",
                            "acceptance": "n >= 1",
                            "depends_on": ["ghost"],
                        }
                    ]
                },
            )
            self.assertTrue(any("unknown task" in p for p in store.validate()))

    def test_validate_catches_unknown_owner(self):
        with tempfile.TemporaryDirectory() as d:
            store = make_store(Path(d))
            store.write(
                "tasks.json",
                {
                    "tasks": [
                        {
                            "id": "a",
                            "title": "t",
                            "owner": "nobody",
                            "priority": 1,
                            "status": "OPEN",
                            "acceptance": "n >= 1",
                        }
                    ]
                },
            )
            self.assertTrue(any("unknown agent" in p for p in store.validate()))

    def test_failures_are_appended_never_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            store = make_store(Path(d))
            store.record_failure("t1", "vision", "first")
            store.record_failure("t2", "vision", "second")
            failures = store.failures()["failures"]
            self.assertEqual([f["task_id"] for f in failures], ["t1", "t2"])
            self.assertFalse(failures[0]["tested"])


if __name__ == "__main__":
    unittest.main()
