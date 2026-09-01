"""Task queue semantics for the babysitter loop.

The queue is a flat list with explicit dependencies and a phase number. Two rules
govern selection and they are enforced here rather than in the loop, so they are
testable without invoking anything:

1. A task is *unblocked* only when every task in ``depends_on`` is closed.
2. A task in phase N cannot run until every phase below N has been VALIDATED --
   the build order in ``docs/SPEEDRUN_PLAN.md`` section 9 is a hard constraint,
   not a suggestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

OPEN_STATUSES = {"OPEN", "BLOCKED", "IN_PROGRESS", "IMPLEMENTED"}
CLOSED_STATUSES = {"VALIDATED", "SUCCESSFUL", "ABANDONED"}
ALL_STATUSES = OPEN_STATUSES | CLOSED_STATUSES | {"FAILED"}

#: Statuses that mean "already claimed by a stage of the pipeline, do not
#: reselect and do not recompute OPEN/BLOCKED for it." IN_PROGRESS is a worker
#: mid-implementation; IMPLEMENTED is done and waiting on qa. Both must survive
#: a `refresh_blocked` call untouched -- otherwise a loop restart between "the
#: worker finished" and "qa ran" would silently bounce the task back to OPEN
#: and hand it to a worker a second time instead of routing it to qa.
IN_FLIGHT_STATUSES = {"IN_PROGRESS", "IMPLEMENTED"}

#: Phase statuses that count as "this phase is done".
PHASE_DONE = {"VALIDATED", "SUCCESSFUL"}


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    owner: str
    priority: int
    status: str
    acceptance: str
    layer: str = ""
    phase: int = 0
    depends_on: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Task":
        return cls(
            id=raw["id"],
            title=raw.get("title", ""),
            owner=raw.get("owner", ""),
            priority=int(raw.get("priority", 0)),
            status=raw.get("status", "OPEN"),
            acceptance=raw.get("acceptance", ""),
            layer=raw.get("layer", ""),
            phase=int(raw.get("phase", 0)),
            depends_on=tuple(raw.get("depends_on", ())),
            notes=raw.get("notes", ""),
        )

    @property
    def is_closed(self) -> bool:
        return self.status in CLOSED_STATUSES


def load_tasks(tasks_doc: Dict[str, Any]) -> List[Task]:
    return [Task.from_dict(raw) for raw in tasks_doc.get("tasks", [])]


def highest_unlocked_phase(project_state: Dict[str, Any]) -> int:
    """The highest phase id that may currently be worked on.

    Phases must complete in order, so this is (lowest phase not yet done). A task
    in a higher phase is blocked even if all of its explicit dependencies are met.
    """
    phases = project_state.get("phases", [])
    for phase in sorted(phases, key=lambda p: int(p.get("id", 0))):
        if phase.get("status") not in PHASE_DONE:
            return int(phase.get("id", 0))
    # Everything is done; allow the last phase.
    return max((int(p.get("id", 0)) for p in phases), default=0)


def unblocked(tasks: Iterable[Task], max_phase: Optional[int] = None) -> List[Task]:
    """Tasks that are runnable right now, in selection order."""
    task_list = list(tasks)
    closed = {t.id for t in task_list if t.is_closed}

    ready: List[Task] = []
    for task in task_list:
        if task.is_closed or task.status in IN_FLIGHT_STATUSES:
            continue
        if max_phase is not None and task.phase > max_phase:
            continue
        if any(dep not in closed for dep in task.depends_on):
            continue
        ready.append(task)

    # Deterministic ordering: failures with untested hypotheses come first (the
    # loop passes those in separately), then priority desc, then id asc.
    ready.sort(key=lambda t: (-t.priority, t.id))
    return ready


def select_next(
    tasks_doc: Dict[str, Any],
    project_state: Dict[str, Any],
) -> Optional[Task]:
    """The single task the babysitter should work on next, or None."""
    tasks = load_tasks(tasks_doc)
    candidates = unblocked(tasks, max_phase=highest_unlocked_phase(project_state))
    return candidates[0] if candidates else None


def set_status(tasks_doc: Dict[str, Any], task_id: str, status: str, **extra: Any) -> Dict[str, Any]:
    """Return a copy of ``tasks_doc`` with one task's status updated."""
    if status not in ALL_STATUSES:
        raise ValueError(f"unknown task status: {status!r}")
    found = False
    for raw in tasks_doc.get("tasks", []):
        if raw.get("id") == task_id:
            raw["status"] = status
            raw.update(extra)
            found = True
            break
    if not found:
        raise KeyError(f"no such task: {task_id!r}")
    return tasks_doc


def refresh_blocked(tasks_doc: Dict[str, Any], project_state: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute OPEN/BLOCKED for every task that is not closed or in-flight.

    Keeps the on-disk queue honest so a human reading ``tasks.json`` sees the same
    picture the loop does.
    """
    tasks = load_tasks(tasks_doc)
    ready = {t.id for t in unblocked(tasks, max_phase=highest_unlocked_phase(project_state))}
    for raw in tasks_doc.get("tasks", []):
        status = raw.get("status")
        if status in CLOSED_STATUSES or status in IN_FLIGHT_STATUSES:
            continue
        raw["status"] = "OPEN" if raw.get("id") in ready else "BLOCKED"
    return tasks_doc
