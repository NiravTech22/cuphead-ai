"""The risk-constrained cost function.

    J = sum_t  gamma^t [ dps_uptime(z_t, a_t)  -  lambda(hp, phase) * P_hit(z_t, a_t) ]
                        - time_penalty  - stall_penalty  - unspent_super_penalty

Survival is the *constraint*; damage is the objective. Cuphead bosses change phase
on HP thresholds rather than timers, so out-damaging a phase skips its attack
patterns entirely -- time saved is superlinear in damage dealt. An agent tuned to
survive will beat an agent tuned to win, and lose the run.

Pure Python and array-free: the shapes here are the reference semantics that the
batched GPU implementation must reproduce. Keeping a readable version means the
vectorized one can be tested against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

#: Risk weight at full health, and at zero. lambda interpolates between them.
#:
#: The ~20x spread is deliberate and load-bearing. Decisions happen at 15 Hz, so a
#: per-step hit probability of 0.15 means being hit within about a second. At full
#: health that is an acceptable price for a second of uninterrupted damage; at one
#: heart it ends the attempt and throws away everything invested in it. A gentler
#: ratio produces an agent that trades its last heart for damage it will never
#: get to use.
LAMBDA_HEALTHY = 0.6
LAMBDA_CRITICAL = 12.0

#: Per-step penalty that makes stalling expensive. Without this the planner learns
#: that standing in a corner scores better than fighting.
TIME_PENALTY = 0.05

#: Cards are worthless once the attempt ends. Charge for hoarding them.
UNSPENT_SUPER_PENALTY = 0.4


def risk_weight(hp: int, max_hp: int = 3, phase_progress: float = 0.0) -> float:
    """lambda(hp, phase): how much a hit costs relative to damage dealt.

    Two effects compound. Low HP raises the cost of a hit because death is closer.
    Late phase progress raises it again because a death later in the fight throws
    away more invested time -- the same hit is cheap at 5% and expensive at 90%.
    """
    if max_hp <= 0:
        raise ValueError("max_hp must be positive")
    hp = max(0, min(hp, max_hp))
    frac = hp / max_hp
    base = LAMBDA_CRITICAL + (LAMBDA_HEALTHY - LAMBDA_CRITICAL) * frac
    progress = max(0.0, min(1.0, phase_progress))
    return base * (1.0 + progress)


def step_reward(
    dps_uptime: float,
    p_hit: float,
    hp: int,
    *,
    max_hp: int = 3,
    phase_progress: float = 0.0,
    aggression: float = 1.0,
) -> float:
    """Reward for a single planned step.

    ``aggression`` is the strategist's per-phase prior, clamped to a safe range so
    a bad LLM suggestion cannot turn the planner suicidal.
    """
    aggression = max(0.1, min(2.0, aggression))
    lam = risk_weight(hp, max_hp, phase_progress)
    return aggression * dps_uptime - lam * p_hit - TIME_PENALTY


def trajectory_value(
    dps_uptimes: Sequence[float],
    p_hits: Sequence[float],
    hp: int,
    *,
    gamma: float = 0.97,
    max_hp: int = 3,
    phase_progress: float = 0.0,
    aggression: float = 1.0,
    terminal_value: float = 0.0,
    unspent_cards: int = 0,
) -> float:
    """Discounted return of one candidate action sequence.

    ``terminal_value`` is the world model's value head bootstrapping past the
    horizon, which is what keeps a 1-second lookahead from being myopic about a
    phase transition just beyond it.
    """
    if len(dps_uptimes) != len(p_hits):
        raise ValueError("dps_uptimes and p_hits must be the same length")

    total = 0.0
    for t, (dps, ph) in enumerate(zip(dps_uptimes, p_hits)):
        total += (gamma**t) * step_reward(
            dps, ph, hp, max_hp=max_hp, phase_progress=phase_progress, aggression=aggression
        )
    total += (gamma ** len(dps_uptimes)) * terminal_value
    total -= UNSPENT_SUPER_PENALTY * max(0, unspent_cards)
    return total


@dataclass(frozen=True)
class PhasePrior:
    """Structured strategist output. Validated and clamped before it reaches the planner.

    An LLM proposal is a hypothesis, never a configuration change. This type is the
    boundary where prose stops and numbers start.
    """

    phase: str
    aggression: float = 1.0
    preferred_lane: str = "none"
    lambda_scale: float = 1.0
    parry_priority: str = "normal"
    super_policy: str = "spend_when_safe"
    forbid: tuple[str, ...] = ()

    def clamped(self) -> "PhasePrior":
        return PhasePrior(
            phase=self.phase,
            aggression=max(0.1, min(2.0, self.aggression)),
            preferred_lane=self.preferred_lane if self.preferred_lane in ("none", "left", "right", "center") else "none",
            lambda_scale=max(0.25, min(4.0, self.lambda_scale)),
            parry_priority=self.parry_priority if self.parry_priority in ("low", "normal", "high") else "normal",
            super_policy=self.super_policy,
            forbid=self.forbid,
        )


def stalling_detected(
    dps_uptime: float,
    ttk_s: float,
    baseline_ttk_s: float,
    *,
    min_uptime: float = 0.35,
    ttk_tolerance: float = 1.15,
) -> bool:
    """Reward-hacking check: is the agent buying survival with time?

    Fires when DPS uptime collapses *and* the attempt runs long relative to the
    baseline. Either alone is noise; together they are the signature of an agent
    that discovered not-fighting.
    """
    return dps_uptime < min_uptime and ttk_s > baseline_ttk_s * ttk_tolerance
