"""Controller — virtual gamepad, action space, frame timing.

Owner: the ``architect`` agent. See ``docs/SPEEDRUN_PLAN.md`` section 2.

Virtual XInput (uinput/evdev on Linux, ViGEm on Windows) rather than keyboard:
lower latency and unambiguous state. An action is a *held* button state for the
duration of one decision window, not an event.

Every actuation is logged against the frame index it was intended for. Misaligned
(state, action) pairs are the most common silent killer of a world model — the
loss curve looks healthy and hit-prediction AUC sits at chance forever.
"""

from .action_space import (
    ACTION_REPEAT,
    DECISION_HZ,
    REFLEX_WHITELIST,
    Action,
    ReflexOverride,
    action_count,
    enumerate_actions,
    index_of,
)

__all__ = [
    "ACTION_REPEAT",
    "DECISION_HZ",
    "REFLEX_WHITELIST",
    "Action",
    "ReflexOverride",
    "action_count",
    "enumerate_actions",
    "index_of",
]
