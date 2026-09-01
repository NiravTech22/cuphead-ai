"""Event detection — the symbolic layer.

Owner: the ``vision`` agent. See ``docs/SPEEDRUN_PLAN.md`` section 3.4.

Events are extracted from HUD deltas, motion segmentation and a learned phase
classifier. They are simultaneously the supervision signal for the world model's
hit head, the unit of measurement for the evaluation harness, and the only thing
the LLM strategist ever reads.

Acceptance: F1 >= 0.95 against hand-labelled traces, reported per boss *and* per
event class.
"""

from .schema import Event, EventKind, EventTrace

__all__ = ["Event", "EventKind", "EventTrace"]
