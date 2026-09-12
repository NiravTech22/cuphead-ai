"""Predict consequences of a candidate action from locally observed transitions."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..memory.latent_bank import Latent, LatentBank, distance


@dataclass(frozen=True)
class Consequence:
    next_state: Latent
    reward: float
    hit_probability: float
    terminal_probability: float
    uncertainty: float
    neighbors: int


class KNNDynamics:
    def __init__(self, bank: LatentBank, *, k: int = 5, radius: float = 0.25):
        if k < 1 or not math.isfinite(radius) or radius < 0:
            raise ValueError("invalid kNN parameters")
        self.bank, self.k, self.radius = bank, k, radius

    def predict(self, state: Latent, action: str, scope: str) -> Consequence | None:
        neighbors = self.bank.lookup(state, action, scope, k=self.k, radius=self.radius)
        if not neighbors:
            return (
                None  # Unknown is never represented as a confident identity transition.
            )
        # Exact matches dominate; distance and ensemble disagreement express uncertainty.
        raw = [1 / max(n.distance, 1e-6) for n in neighbors]
        weights = [w / sum(raw) for w in raw]
        predictions = [
            tuple(
                z + b - a
                for z, a, b in zip(state, n.experience.state, n.experience.next_state)
            )
            for n in neighbors
        ]
        mean = tuple(
            sum(w * p[i] for w, p in zip(weights, predictions))
            for i in range(len(state))
        )
        spread = (
            sum(w * distance(p, mean) ** 2 for w, p in zip(weights, predictions)) ** 0.5
        )
        return Consequence(
            mean,
            sum(w * n.experience.reward for w, n in zip(weights, neighbors)),
            sum(w * n.experience.hit for w, n in zip(weights, neighbors)),
            sum(w * n.experience.terminal for w, n in zip(weights, neighbors)),
            spread + sum(w * n.distance for w, n in zip(weights, neighbors)),
            len(neighbors),
        )
