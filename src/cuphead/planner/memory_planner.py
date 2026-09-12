"""Consequence scoring with local, action-specific cold-start exploration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..memory.latent_bank import Latent, distance
from ..world_model.knn_dynamics import Consequence, KNNDynamics


@dataclass(frozen=True)
class MemoryDecision:
    action: str
    reason: str
    consequence: Consequence | None
    score: float | None


class MemoryPlanner:
    def __init__(
        self,
        dynamics: KNNDynamics,
        *,
        hit_cost: float = 5.0,
        uncertainty_cost: float = 0.25,
        max_hit_probability: float = 0.5,
    ):
        if (
            not math.isfinite(hit_cost)
            or not math.isfinite(uncertainty_cost)
            or hit_cost < 0
            or uncertainty_cost < 0
            or not 0 <= max_hit_probability <= 1
        ):
            raise ValueError("invalid consequence costs")
        self.dynamics = dynamics
        self.hit_cost, self.uncertainty_cost = hit_cost, uncertainty_cost
        self.max_hit_probability = max_hit_probability

    def choose(
        self,
        state: Latent,
        candidates: Sequence[str],
        scope: str,
        *,
        goal: Latent | None = None,
        designated: str | None = None,
    ) -> MemoryDecision:
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("nonempty unique candidate actions required")
        if designated is not None and designated not in candidates:
            raise ValueError("designated action must be a candidate")
        if goal is not None:
            distance(state, goal)  # Validate before any cold-start early return.
        predictions = {a: self.dynamics.predict(state, a, scope) for a in candidates}

        def score(p: Consequence) -> float:
            progress = (
                distance(state, goal) - distance(p.next_state, goal)
                if goal is not None
                else 0
            )
            return (
                p.reward
                + progress
                - self.hit_cost * p.hit_probability
                - self.uncertainty_cost * p.uncertainty
            )

        if designated is not None:
            known = predictions[designated]
            if (
                known is not None
                and known.hit_probability <= self.max_hit_probability
                and (goal is None or score(known) > 0)
            ):
                return MemoryDecision(
                    designated, "designated_memory", known, score(known)
                )
        untried = [a for a in candidates if predictions[a] is None]
        if untried:
            # The caller orders valid exploration actions. Prefer the designated action
            # only if it too is untried in this local neighborhood and scenario.
            action = designated if designated in untried else untried[0]
            return MemoryDecision(action, "cold_start", None, None)
        safe = [
            a
            for a in candidates
            if predictions[a].hit_probability <= self.max_hit_probability
        ]
        choices = safe or list(candidates)
        best = max(choices, key=lambda a: score(predictions[a]))
        return MemoryDecision(
            best, "predicted_consequence", predictions[best], score(predictions[best])
        )
