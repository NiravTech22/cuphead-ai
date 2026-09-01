"""Reflex layer — the 60 Hz reactive policy.

Owner: the ``world-model`` agent. See ``docs/SPEEDRUN_PLAN.md`` section 6.

Parry windows are roughly 5-6 frames (~90 ms). A 66 ms tactical loop plus model
error plus actuation latency cannot reliably hit that, so parry and i-frame dash
timing get a dedicated small convnet running every frame on the pink mask and a
short frame stack.

Arbitration: the reflex head may preempt the planner, but only within the
whitelist in ``control.action_space.REFLEX_WHITELIST`` and only above a per-boss
confidence threshold. Every override is logged with its outcome — override
precision is a gated metric (>= 0.90).

The planner decides where to be. The reflex decides what to do in the next 90 ms.
"""
