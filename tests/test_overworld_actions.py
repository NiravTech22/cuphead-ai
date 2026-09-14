"""Regression for a recorded live run that paused during map exploration."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cuphead.control.timed_input import OVERWORLD_INPUTS
from cuphead.memory.latent_bank import Experience, LatentBank
from cuphead.planner.memory_planner import MemoryPlanner
from cuphead.world_model.knn_dynamics import KNNDynamics


class OverworldActionsTests(unittest.TestCase):
    def test_exploration_cannot_pause_after_unhelpful_known_actions(self):
        bank = LatentBank(2, "fixture")
        actions = [a.key for a in OVERWORLD_INPUTS]
        # Exhaust all candidates: the former second cold-start choice was Start.
        for _ in actions:
            decision = MemoryPlanner(KNNDynamics(bank)).choose((0., 0.), actions, "map")
            selected = next(a for a in OVERWORLD_INPUTS if a.key == decision.action)
            self.assertFalse(selected.start)
            bank.add(Experience((0., 0.), selected.key, (0., 0.), "map", reward=-.01))


if __name__ == "__main__":
    unittest.main()
