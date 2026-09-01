"""Planner — CEM/MPPI model-predictive control over latent rollouts.

Owner: the ``world-model`` agent. See ``docs/SPEEDRUN_PLAN.md`` section 5.

N = 512 sequences, <= 3 CEM iterations, horizon 12-20 steps at 15 Hz (~1 second,
matched to the duration of a projectile wave). Seeded from the policy prior rather
than uniformly, which is what makes three iterations enough.

Hard real-time: total planning budget <= 10 ms. A planner that is 20% better and
15 ms is a failed change, not a tradeoff.
"""

from .objective import (
    PhasePrior,
    risk_weight,
    stalling_detected,
    step_reward,
    trajectory_value,
)

__all__ = [
    "PhasePrior",
    "risk_weight",
    "stalling_detected",
    "step_reward",
    "trajectory_value",
]
