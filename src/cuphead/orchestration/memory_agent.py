"""Capture -> frozen latent -> action-conditioned memory -> execute -> observe.

The runtime owns orchestration; perception, memory, world model, planning and
actuation remain separate. Unknown observations release all input and wait.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from ..control.timed_input import COMBAT_INPUTS, NAVIGATION_INPUTS, OVERWORLD_INPUTS
from ..memory.latent_bank import Experience, LatentBank, distance
from ..planner.memory_planner import MemoryPlanner
from ..strategist.landmark_route import LandmarkRoute


@dataclass(frozen=True)
class AgentStep:
    status: str
    scene: str | None
    next_scene: str | None = None
    action: str | None = None
    reason: str | None = None
    reward: float | None = None
    prediction_error: float | None = None
    elapsed: float | None = None
    predicted_score: float | None = None
    predicted_hit_probability: float | None = None
    predicted_uncertainty: float | None = None


class MemoryAgent:
    def __init__(
        self,
        *,
        source,
        encoder,
        verifier,
        executor,
        bank: LatentBank,
        planner: MemoryPlanner,
        route: LandmarkRoute | None = None,
        emit: Callable[[dict], None] | None = None,
        confirmations: int = 3,
        navigation_only: bool = False,
        settle_seconds: float = 5.0,
    ):
        if confirmations < 2:
            raise ValueError("terminal verification needs at least two observations")
        self.source, self.encoder, self.verifier = source, encoder, verifier
        self.executor, self.bank, self.planner, self.route = (
            executor,
            bank,
            planner,
            route,
        )
        self.emit = emit or (lambda record: None)
        self.confirmations = confirmations
        self.navigation_only = navigation_only
        self.settle_seconds = settle_seconds
        self._last_mode = None
        self._terminal_label = None
        self._terminal_count = 0
        self._goals = {}
        self.last_frame = None

    def _record(self, result: AgentStep) -> AgentStep:
        self.emit(asdict(result))
        return result

    def step(self) -> AgentStep:
        before_frame = self.source.read()
        self.last_frame = before_frame
        scene = self.verifier.classify(before_frame)
        if scene is None:
            self.executor.actuator.neutral()
            self.encoder.reset()
            self._terminal_count = 0
            return self._record(AgentStep("unknown", None))
        at_navigation_goal = (
            self.navigation_only and self.route and scene.label == self.route.goal
        )
        if scene.won or at_navigation_goal:
            self.executor.actuator.neutral()
            if self._terminal_label == scene.label:
                self._terminal_count += 1
            else:
                self._terminal_label, self._terminal_count = scene.label, 1
            goal_matches = self.route is None or scene.label == self.route.goal
            status = (
                "won"
                if self._terminal_count >= self.confirmations and goal_matches
                else "verifying"
            )
            if status == "won" and not scene.won:
                status = "goal_reached"
            return self._record(AgentStep(status, scene.label))
        self._terminal_count = 0
        if scene.mode != self._last_mode or scene.terminal:
            self.encoder.reset()
        self._last_mode = scene.mode
        low_detail = scene.mode != "combat"
        before = self.encoder.encode(before_frame, low_detail=low_detail)
        actions = (
            COMBAT_INPUTS
            if scene.mode == "combat"
            else OVERWORLD_INPUTS
            if scene.mode == "overworld"
            else NAVIGATION_INPUTS
        )
        edge = self.route.next_edge(scene.label) if self.route else None
        designated, goal = None, None
        if edge:
            designated = next((a.key for a in actions if a.name == edge.action), None)
            # Static goal embeddings only guide static navigation, not motion clips.
            if low_detail and edge.target in self.verifier.references:
                if edge.target not in self._goals:
                    self.encoder.reset()
                    self._goals[edge.target] = self.encoder.encode(
                        self.verifier.references[edge.target]
                    )
                    self.encoder.reset()
                goal = self._goals[edge.target]
        # Include preprocessing mode in the exact filter; 128px and video latents differ.
        scope = f"{scene.label}:{'image128' if low_detail else 'video256'}"
        decision = self.planner.choose(
            before, [a.key for a in actions], scope, goal=goal, designated=designated
        )
        action = next(a for a in actions if a.key == decision.action)
        started = time.perf_counter()
        observed = self.executor.execute(action, self.source.read)
        after_frame = observed[-1]
        self.last_frame = after_frame
        after_scene = self.verifier.classify(after_frame)
        settle_deadline = time.perf_counter() + self.settle_seconds
        while time.perf_counter() < settle_deadline and (
            after_scene is None
            or (
                scene.mode == "menu"
                and edge is not None
                and after_scene.label == scene.label
            )
        ):
            time.sleep(0.05)
            after_frame = self.source.read()
            self.last_frame = after_frame
            observed = [*observed[-3:], after_frame]
            after_scene = self.verifier.classify(after_frame)
        if after_scene is None:
            # Loading/transitional imagery is not silently paired with a later action.
            self.encoder.reset()
            return self._record(
                AgentStep(
                    "unverified_transition",
                    scene.label,
                    action=action.name,
                    reason=decision.reason,
                )
            )
        if not low_detail and hasattr(self.encoder, "encode_clip"):
            after = self.encoder.encode_clip(observed, low_detail=False)
        else:
            after = self.encoder.encode(after_frame, low_detail=low_detail)
        hit = (
            scene.hp is not None
            and after_scene.hp is not None
            and after_scene.hp < scene.hp
        )
        hit = hit or after_scene.mode == "death"
        terminal = after_scene.terminal
        progress = (
            distance(before, goal) - distance(after, goal) if goal is not None else 0.0
        )
        reward = progress - 0.01 - 5 * hit + 20 * after_scene.won
        if (
            self.route
            and edge
            and after_scene.label == edge.target
            and scene.label != after_scene.label
        ):
            reward += 1
        elapsed = time.perf_counter() - started
        self.bank.add(
            Experience(
                before,
                action.key,
                after,
                scope,
                reward,
                hit,
                terminal,
                before_frame.index,
                after_frame.index,
                elapsed,
            )
        )
        if self.route:
            self.route.observe(scene.label, action.name, after_scene.label)
        error = (
            distance(decision.consequence.next_state, after)
            if decision.consequence
            else None
        )
        return self._record(
            AgentStep(
                "observed",
                scene.label,
                after_scene.label,
                action.name,
                decision.reason,
                reward,
                error,
                elapsed,
                predicted_score=decision.score,
                predicted_hit_probability=decision.consequence.hit_probability if decision.consequence else None,
                predicted_uncertainty=decision.consequence.uncertainty if decision.consequence else None,
            )
        )
