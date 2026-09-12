"""End-to-end behavior, terminal verification, and rejection of unobserved outcomes."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cuphead.control.timed_input import TimedExecutor
from cuphead.evaluation.synthetic_agent import (
    FixtureEncoder,
    SyntheticGame,
    run_synthetic,
)
from cuphead.memory.latent_bank import LatentBank
from cuphead.orchestration.memory_agent import MemoryAgent
from cuphead.planner.memory_planner import MemoryPlanner
from cuphead.world_model.knn_dynamics import KNNDynamics


class RuntimeTests(unittest.TestCase):
    def test_entire_fixture_including_cold_start_death_retry_and_reload(self):
        first, bank = run_synthetic()
        self.assertTrue(first["fixture_completed"])
        self.assertEqual(first["deaths"], 1)
        self.assertGreater(first["cold_starts"], 0)
        second, _ = run_synthetic(bank=bank)
        self.assertTrue(second["fixture_completed"])
        self.assertEqual(second["deaths"], 0)
        self.assertGreater(second["memory_retrievals"], 0)
        self.assertFalse(second["cuphead_victory_verified"])

    def agent(self):
        game = SyntheticGame()
        encoder = FixtureEncoder()
        bank = LatentBank(encoder.output_dim, encoder.fingerprint)
        executor = TimedExecutor(game, sleep=lambda _: None)
        agent = MemoryAgent(
            source=game,
            encoder=encoder,
            verifier=game,
            executor=executor,
            bank=bank,
            planner=MemoryPlanner(KNNDynamics(bank)),
            settle_seconds=0,
        )
        return game, agent, bank

    def test_win_requires_repeated_visual_evidence_and_sends_no_actions(self):
        game, agent, bank = self.agent()
        game.state = "victory"
        self.assertEqual(agent.step().status, "verifying")
        self.assertEqual(agent.step().status, "verifying")
        self.assertEqual(agent.step().status, "won")
        self.assertIsNone(game.active)
        self.assertEqual(len(bank), 0)

    def test_unknown_screen_releases_and_does_not_corrupt_memory(self):
        game, agent, bank = self.agent()
        game.active = "shoot"
        game.classify = lambda _: None
        self.assertEqual(agent.step().status, "unknown")
        self.assertIsNone(game.active)
        self.assertEqual(len(bank), 0)

    def test_unverified_transition_cannot_become_a_training_example(self):
        game, agent, bank = self.agent()
        classify = game.classify
        game.classify = lambda frame: (
            classify(frame) if frame.payload == "title" else None
        )
        self.assertEqual(agent.step().status, "unverified_transition")
        self.assertEqual(len(bank), 0)
        self.assertIsNone(game.active)

    def test_unknown_frame_interrupts_terminal_confirmation(self):
        game, agent, _bank = self.agent()
        game.state = "victory"
        self.assertEqual(agent.step().status, "verifying")
        original = game.classify
        game.classify = lambda _: None
        self.assertEqual(agent.step().status, "unknown")
        game.classify = original
        self.assertEqual(agent.step().status, "verifying")
        self.assertEqual(agent.step().status, "verifying")
        self.assertEqual(agent.step().status, "won")


if __name__ == "__main__":
    unittest.main()
