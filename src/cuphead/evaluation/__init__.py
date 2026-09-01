"""Evaluation — metrics, the eval harness, and the regression suite.

Owner: the ``qa`` agent. See ``docs/SPEEDRUN_PLAN.md`` section 1.2.

Evaluation reads event traces, never pixels, so the whole regression suite runs
offline against frozen replays at no cost. The statistical gate itself lives in
``orchestration.gates`` because the babysitter loop needs it before any ML
dependency exists.
"""

from .metrics import AttemptMetrics, aggregate, per_boss

__all__ = ["AttemptMetrics", "aggregate", "per_boss"]
