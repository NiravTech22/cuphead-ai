"""Orchestration layer: persistent state, task queue, and the statistical gate.

Stdlib only. This layer must import and run on a machine with nothing installed,
because it is what tells you whether the rest of the project is safe to start.
"""

from .gates import ArmResult, GateVerdict, evaluate, format_verdict
from .state_store import StateError, StateStore, utcnow
from .tasks import Task, refresh_blocked, select_next, set_status, unblocked

__all__ = [
    "ArmResult",
    "GateVerdict",
    "StateError",
    "StateStore",
    "Task",
    "evaluate",
    "format_verdict",
    "refresh_blocked",
    "select_next",
    "set_status",
    "unblocked",
    "utcnow",
]
