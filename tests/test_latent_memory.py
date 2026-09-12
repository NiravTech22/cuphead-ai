"""Behavioral tests for action-local memory, consequence planning, and persistence."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cuphead.control.timed_input import ControlInput, TimedExecutor
from cuphead.memory.latent_bank import Experience, LatentBank
from cuphead.planner.memory_planner import MemoryPlanner
from cuphead.strategist.landmark_route import LandmarkRoute, RouteEdge
from cuphead.world_model.knn_dynamics import KNNDynamics


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.bank = LatentBank(2, "frozen-weights", capacity=8)
        self.model = KNNDynamics(self.bank, radius=0.1, k=3)
        self.planner = MemoryPlanner(self.model)

    def add(self, a="shoot", state=(0.0, 0.0), after=(1.0, 0.0), scope="boss", **kw):
        self.bank.add(Experience(state, a, after, scope, **kw))

    def test_no_cross_action_or_scenario_or_distant_neighbors(self):
        self.add()
        self.assertIsNone(self.model.predict((0.0, 0.0), "jump", "boss"))
        self.assertIsNone(self.model.predict((0.0, 0.0), "shoot", "menu"))
        self.assertIsNone(self.model.predict((1.0, 1.0), "shoot", "boss"))
        self.assertIsNotNone(self.model.predict((0.0, 0.0), "shoot", "boss"))

    def test_cold_start_exhausts_untried_actions_locally(self):
        actions = ["shoot", "jump", "duck"]
        picked = []
        for _ in actions:
            d = self.planner.choose((0.0, 0.0), actions, "boss")
            picked.append(d.action)
            self.assertEqual(d.reason, "cold_start")
            self.add(a=d.action, after=(0.0, 0.0))
        self.assertEqual(picked, actions)
        self.assertEqual(
            self.planner.choose((0.0, 0.0), actions, "boss").reason,
            "predicted_consequence",
        )
        self.assertEqual(
            self.planner.choose((1.0, 1.0), actions, "boss").reason, "cold_start"
        )

    def test_designated_action_uses_only_its_own_evidence(self):
        self.add(a="jump")
        d = self.planner.choose(
            (0.0, 0.0), ["jump", "shoot"], "boss", designated="shoot"
        )
        self.assertEqual((d.action, d.reason), ("shoot", "cold_start"))
        self.add(a="shoot", reward=1)
        d = self.planner.choose(
            (0.0, 0.0), ["jump", "shoot"], "boss", designated="shoot"
        )
        self.assertEqual(d.reason, "designated_memory")

    def test_hit_consequences_override_dangerous_designated_action(self):
        self.add(a="shoot", reward=0.5, hit=True)
        self.add(a="duck", reward=0.1)
        d = self.planner.choose(
            (0.0, 0.0), ["shoot", "duck"], "boss", designated="shoot"
        )
        self.assertEqual(d.action, "duck")

    def test_predicts_delta_at_query_not_stored_state(self):
        self.add(state=(0.0, 0.0), after=(0.5, 0.0))
        p = self.model.predict((0.05, 0.0), "shoot", "boss")
        self.assertAlmostEqual(p.next_state[0], 0.55)
        self.assertAlmostEqual(p.uncertainty, 0.05)

    def test_disagreeing_consequences_raise_uncertainty(self):
        self.add(after=(1.0, 0.0))
        self.add(after=(-1.0, 0.0), hit=True)
        p = self.model.predict((0.0, 0.0), "shoot", "boss")
        self.assertEqual(p.next_state, (0.0, 0.0))
        self.assertAlmostEqual(p.hit_probability, 0.5)
        self.assertGreater(p.uncertainty, 0.9)

    def test_goal_scores_consequences_not_action_names(self):
        self.add(a="left", after=(1.0, 0.0))
        self.add(a="right", after=(-1.0, 0.0))
        d = self.planner.choose((0.0, 0.0), ["right", "left"], "boss", goal=(1.0, 0.0))
        self.assertEqual(d.action, "left")

    def test_memory_reload_and_fingerprint_rejection(self):
        self.add(hit=True, terminal=True)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bank.json"
            self.bank.save(p)
            b = LatentBank.load(p, encoder_id="frozen-weights")
            self.assertTrue(b.lookup((0.0, 0.0), "shoot", "boss")[0].experience.hit)
            with self.assertRaises(ValueError):
                LatentBank.load(p, encoder_id="different-weights")

    def test_bounded_eviction_and_mutable_inputs(self):
        state = [0.0, 0.0]
        self.add(state=state)
        state[0] = 9
        self.assertEqual(self.bank.lookup((0.0, 0.0), "shoot", "boss")[0].distance, 0)
        for i in range(10):
            self.add(a=str(i))
        self.assertEqual(len(self.bank), 8)
        self.assertFalse(self.bank.lookup((0.0, 0.0), "shoot", "boss"))

    def test_invalid_transition_and_query_rejected(self):
        for state in ((0.0,), (math.nan, 0.0), (math.inf, 0.0)):
            with self.assertRaises(ValueError):
                self.add(state=state)
        with self.assertRaises(ValueError):
            self.add(frame=2, next_frame=1)
        with self.assertRaises(ValueError):
            self.add(elapsed=0)
        with self.assertRaises(ValueError):
            self.bank.lookup((0.0, 0.0), "a", "b", radius=math.nan)
        with self.assertRaises(ValueError):
            self.planner.choose((0.0, 0.0), ["a"], "boss", goal=(0.0,))


class NavigationTests(unittest.TestCase):
    def test_route_relocalizes_after_reset_without_advancing_on_noop(self):
        route = LandmarkRoute(
            [
                RouteEdge("title", "map", "confirm"),
                RouteEdge("map", "boss", "right"),
                RouteEdge("death", "boss", "confirm"),
                RouteEdge("boss", "win", "shoot"),
            ],
            "win",
        )
        self.assertEqual(route.next_edge("title").action, "confirm")
        self.assertFalse(route.observe("title", "confirm", "title"))
        self.assertEqual(route.confirmed, {})
        self.assertEqual(route.next_edge("death").target, "boss")
        self.assertTrue(route.observe("title", "confirm", "map"))
        self.assertEqual(route.next_edge("map").target, "boss")
        self.assertIsNone(route.next_edge("win"))

    def test_cycles_and_unreachable_goal_terminate(self):
        route = LandmarkRoute(
            [RouteEdge("a", "b", "right"), RouteEdge("b", "a", "left")], "win"
        )
        self.assertIsNone(route.next_edge("a"))

    def test_timed_executor_releases_after_capture_exception(self):
        class Pad:
            active = False

            def send_buttons(self, a):
                self.active = True

            def neutral(self):
                self.active = False

        pad = Pad()
        timer = [0.0]

        def sleep(t):
            timer[0] += t

        executor = TimedExecutor(pad, clock=lambda: timer[0], sleep=sleep)

        def fail():
            raise RuntimeError("lost screen")

        with self.assertRaises(RuntimeError):
            executor.execute(ControlInput("jump", a=True), fail)
        self.assertFalse(pad.active)

    def test_real_duration_and_release_are_part_of_action_identity(self):
        a = ControlInput("confirm", a=True, hold_frames=2)
        b = ControlInput("confirm", a=True, hold_frames=4)
        self.assertNotEqual(a.key, b.key)
        timer = [0.0]
        samples = []

        class Pad:
            def send_buttons(self, a):
                pass

            def neutral(self):
                pass

        def sleep(t):
            timer[0] += t

        ex = TimedExecutor(Pad(), clock=lambda: timer[0], sleep=sleep)
        ex.execute(a, lambda: samples.append(timer[0]))
        self.assertAlmostEqual(timer[0], 4 / 60)
        self.assertEqual(len(samples), 3)


if __name__ == "__main__":
    unittest.main()
