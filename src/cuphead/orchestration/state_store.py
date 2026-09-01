"""Atomic, schema-checked access to the ``.ai/`` persistent state files.

Every read and write of project state goes through here. Direct ``json.load`` on
``.ai/*.json`` elsewhere in the codebase is a bug: the loop relies on atomic
replacement so a crash mid-write cannot leave a truncated state file behind.

Stdlib only, by design -- the orchestration layer must run before any dependency
is installed.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

STATE_FILES = ("project_state.json", "tasks.json", "agent_status.json", "failures.json")


class StateError(RuntimeError):
    """Raised when the persistent state is missing, malformed, or inconsistent."""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repo_root(start: Path | None = None) -> Path:
    """Walk upward to the directory containing ``.ai/``."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".ai").is_dir():
            return candidate
    raise StateError("could not locate repository root (no .ai/ directory found)")


@dataclass
class StateStore:
    """Reads and writes the four ``.ai/`` documents."""

    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "StateStore":
        return cls(root=repo_root(start))

    @property
    def ai_dir(self) -> Path:
        return self.root / ".ai"

    def path(self, name: str) -> Path:
        if name not in STATE_FILES:
            raise StateError(f"unknown state file: {name!r}")
        return self.ai_dir / name

    # -- io -------------------------------------------------------------

    def read(self, name: str) -> Dict[str, Any]:
        p = self.path(name)
        if not p.exists():
            raise StateError(f"missing state file: {p}")
        try:
            with p.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise StateError(f"{p} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise StateError(f"{p} must contain a JSON object, got {type(data).__name__}")
        return data

    def write(self, name: str, data: Dict[str, Any]) -> None:
        """Atomically replace a state file.

        Writes to a temporary file in the same directory and ``os.replace``s it,
        so a reader never observes a partially written document.
        """
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=False)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- convenience ----------------------------------------------------

    def project_state(self) -> Dict[str, Any]:
        return self.read("project_state.json")

    def tasks(self) -> Dict[str, Any]:
        return self.read("tasks.json")

    def agent_status(self) -> Dict[str, Any]:
        return self.read("agent_status.json")

    def failures(self) -> Dict[str, Any]:
        return self.read("failures.json")

    def touch_project_state(self, **updates: Any) -> Dict[str, Any]:
        state = self.project_state()
        state.update(updates)
        state["last_updated"] = utcnow()
        self.write("project_state.json", state)
        return state

    def record_failure(
        self,
        task_id: str,
        agent: str,
        reason: str,
        hypothesis: str = "",
        evidence: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Append a failure record. Failures are appended, never overwritten."""
        doc = self.failures()
        entry = {
            "task_id": task_id,
            "agent": agent,
            "reason": reason,
            "hypothesis": hypothesis,
            "evidence": evidence or {},
            "recorded_at": utcnow(),
            "tested": False,
        }
        doc.setdefault("failures", []).append(entry)
        self.write("failures.json", doc)
        return entry

    def set_agent(self, agent: str, **updates: Any) -> None:
        doc = self.agent_status()
        agents = doc.setdefault("agents", {})
        if agent not in agents:
            raise StateError(f"unknown agent: {agent!r}")
        agents[agent].update(updates)
        self.write("agent_status.json", doc)

    def set_loop(self, **updates: Any) -> None:
        doc = self.agent_status()
        doc.setdefault("loop", {}).update(updates)
        self.write("agent_status.json", doc)

    def validate(self) -> list[str]:
        """Return a list of problems with the persistent state. Empty means healthy."""
        problems: list[str] = []
        for name in STATE_FILES:
            try:
                self.read(name)
            except StateError as exc:
                problems.append(str(exc))

        if problems:
            return problems

        state = self.project_state()
        for key in ("project", "current_phase", "phases", "components", "gate"):
            if key not in state:
                problems.append(f"project_state.json missing required key {key!r}")

        tasks_doc = self.tasks()
        tasks = tasks_doc.get("tasks")
        if not isinstance(tasks, list):
            problems.append("tasks.json must contain a 'tasks' list")
            return problems

        seen: set[str] = set()
        known_agents = set(self.agent_status().get("agents", {}))
        for task in tasks:
            tid = task.get("id")
            if not tid:
                problems.append("a task has no id")
                continue
            if tid in seen:
                problems.append(f"duplicate task id: {tid}")
            seen.add(tid)
            for key in ("title", "owner", "priority", "status", "acceptance"):
                if key not in task:
                    problems.append(f"task {tid} missing required field {key!r}")
            owner = task.get("owner")
            if owner and owner not in known_agents:
                problems.append(f"task {tid} owned by unknown agent {owner!r}")
            if not str(task.get("acceptance", "")).strip():
                problems.append(f"task {tid} has an empty acceptance criterion")

        for task in tasks:
            for dep in task.get("depends_on", []):
                if dep not in seen:
                    problems.append(f"task {task.get('id')} depends on unknown task {dep!r}")

        return problems
