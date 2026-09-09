"""Closed-loop latent world model for deterministic menu navigation.

A frozen visual encoder produces state and goal latents. Only the small
action-conditioned predictor learns online from observed menu transitions.
Template classification remains a verifier when the predictor is surprised.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from ..perception.capture import Frame, FrameSource
from ..perception.latent_encoder import LatentEncoder


Latent = tuple[float, ...]


def latent_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"latent dimensions differ: {len(left)} != {len(right)}")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


@dataclass(frozen=True)
class MenuInput:
    """One held virtual-gamepad state suitable for a menu transition."""

    name: str
    stick_x: float = 0.0
    stick_y: float = 0.0
    confirm: bool = False
    skip: bool = False
    hold_frames: int = 4

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("menu action needs a name")
        if not -1.0 <= self.stick_x <= 1.0 or not -1.0 <= self.stick_y <= 1.0:
            raise ValueError("menu stick components must be in [-1, 1]")
        if self.hold_frames < 1:
            raise ValueError("hold_frames must be positive")

    def vector(self) -> tuple[float, float, float, float]:
        return (self.stick_x, self.stick_y, float(self.confirm), float(self.skip))


DEFAULT_MENU_ACTIONS = (
    MenuInput("confirm", confirm=True, hold_frames=2),
    MenuInput("skip", skip=True, hold_frames=2),
    MenuInput("left", stick_x=-1.0),
    MenuInput("right", stick_x=1.0),
    MenuInput("up", stick_y=1.0),
    MenuInput("down", stick_y=-1.0),
)


class MenuActuator(Protocol):
    def send_menu(self, action: MenuInput) -> Any: ...

    def neutral_menu(self) -> None: ...


class FallbackClassifier(Protocol):
    """The old template classifier, kept only as a low-confidence check."""

    def classify(self, frame: Frame) -> str | None: ...


class ActionConditionedPredictor(Protocol):
    @property
    def ready(self) -> bool: ...

    def predict(self, latent: Latent, action: MenuInput) -> Latent: ...

    def observe(self, latent: Latent, action: MenuInput, next_latent: Latent) -> None: ...

    def fit(self, *, epochs: int = 1) -> float | None: ...


@dataclass
class GoalLatents:
    """Reference frames embedded once, rather than pixel-template matched."""

    values: dict[str, Latent] = field(default_factory=dict)

    def capture(self, label: str, frame: Frame, encoder: LatentEncoder) -> Latent:
        if not label:
            raise ValueError("goal label must not be empty")
        latent = encoder.encode(frame)
        self.values[label] = latent
        return latent

    def require(self, label: str) -> Latent:
        try:
            return self.values[label]
        except KeyError as exc:
            raise KeyError(f"no goal latent registered for {label!r}") from exc


@dataclass(frozen=True)
class TransitionRecord:
    frame_t: int
    action: MenuInput
    frame_t1: int
    latent_t: Latent
    latent_t1: Latent
    predicted_t1: Latent
    prediction_error: float


@dataclass
class TransitionLog:
    """In-memory online data, serializable as JSONL after the observed run."""

    records: list[TransitionRecord] = field(default_factory=list)

    def append(self, record: TransitionRecord) -> None:
        self.records.append(record)

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in self.records),
            encoding="utf-8",
        )


class TorchMLPPredictor:
    """Small trainable latent dynamics head; the visual encoder stays frozen."""

    def __init__(
        self,
        latent_dim: int,
        *,
        hidden_dim: int = 128,
        learning_rate: float = 3e-3,
        min_transitions: int = 4,
        device: str | None = None,
    ) -> None:
        if latent_dim < 1 or hidden_dim < 1 or min_transitions < 1:
            raise ValueError("latent_dim, hidden_dim and min_transitions must be positive")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TorchMLPPredictor requires torch") from exc
        self._torch = torch
        self.latent_dim = latent_dim
        self.min_transitions = min_transitions
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self._examples: list[tuple[Latent, MenuInput, Latent]] = []

    @property
    def ready(self) -> bool:
        return len(self._examples) >= self.min_transitions

    def _input(self, latent: Latent, action: MenuInput) -> Any:
        if len(latent) != self.latent_dim:
            raise ValueError(f"expected latent dimension {self.latent_dim}, got {len(latent)}")
        return self._torch.tensor(
            [*latent, *action.vector()], dtype=self._torch.float32, device=self.device
        )

    def predict(self, latent: Latent, action: MenuInput) -> Latent:
        self.model.eval()
        with self._torch.inference_mode():
            delta = self.model(self._input(latent, action))
        return tuple(float(a + b) for a, b in zip(latent, delta.detach().cpu().tolist()))

    def observe(self, latent: Latent, action: MenuInput, next_latent: Latent) -> None:
        if len(next_latent) != self.latent_dim:
            raise ValueError(f"expected latent dimension {self.latent_dim}, got {len(next_latent)}")
        self._examples.append((latent, action, next_latent))

    def fit(self, *, epochs: int = 1) -> float | None:
        if not self._examples:
            return None
        if epochs < 1:
            raise ValueError("epochs must be positive")
        inputs = self._torch.stack([self._input(latent, action) for latent, action, _ in self._examples])
        current = self._torch.tensor(
            [latent for latent, _, _ in self._examples], dtype=self._torch.float32, device=self.device
        )
        targets = self._torch.tensor(
            [next_latent for _, _, next_latent in self._examples],
            dtype=self._torch.float32,
            device=self.device,
        )
        loss_value = 0.0
        self.model.train()
        for _ in range(epochs):
            predicted = current + self.model(inputs)
            loss = self._torch.nn.functional.mse_loss(predicted, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.detach().cpu())
        return loss_value


@dataclass(frozen=True)
class PlannedAction:
    action: MenuInput
    predicted_latent: Latent
    predicted_goal_distance: float


class MiniCEMPlanner:
    """Exhaustively score a small discrete candidate set in latent space."""

    def __init__(self, predictor: ActionConditionedPredictor, *, candidate_budget: int = 32) -> None:
        if candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        self.predictor = predictor
        self.candidate_budget = candidate_budget

    def plan(
        self, current: Latent, goal: Latent, candidates: Sequence[MenuInput]
    ) -> PlannedAction:
        if not candidates:
            raise ValueError("at least one candidate action is required")
        scored = []
        for action in candidates[: self.candidate_budget]:
            predicted = self.predictor.predict(current, action)
            scored.append((latent_distance(predicted, goal), action.name, action, predicted))
        distance, _, action, predicted = min(scored)
        return PlannedAction(action, predicted, distance)


@dataclass(frozen=True)
class NavigationStep:
    goal_label: str
    action: MenuInput
    before_distance: float
    after_distance: float
    prediction_error: float
    progressed: bool
    fallback_label: str | None
    fallback_confirmed: bool
    training_loss: float | None


class ClosedLoopMenuNavigator:
    """Execute one predicted menu step, observe it, learn, then verify it."""

    def __init__(
        self,
        *,
        encoder: LatentEncoder,
        goals: GoalLatents,
        predictor: ActionConditionedPredictor,
        actuator: MenuActuator,
        source: FrameSource,
        transitions: TransitionLog | None = None,
        fallback: FallbackClassifier | None = None,
        goal_threshold: float = 0.35,
        min_distance_decrease: float = 1e-4,
        max_prediction_error: float = 0.5,
    ) -> None:
        if goal_threshold < 0 or min_distance_decrease < 0 or max_prediction_error < 0:
            raise ValueError("navigation thresholds must be non-negative")
        self.encoder = encoder
        self.goals = goals
        self.predictor = predictor
        self.actuator = actuator
        self.source = source
        self.transitions = transitions or TransitionLog()
        self.fallback = fallback
        self.goal_threshold = goal_threshold
        self.min_distance_decrease = min_distance_decrease
        self.max_prediction_error = max_prediction_error
        self.planner = MiniCEMPlanner(predictor)

    def step(self, goal_label: str, candidates: Sequence[MenuInput] = DEFAULT_MENU_ACTIONS) -> NavigationStep:
        before_frame = self.source.read()
        before = self.encoder.encode(before_frame)
        goal = self.goals.require(goal_label)
        planned = self.planner.plan(before, goal, candidates)

        self.actuator.send_menu(planned.action)
        after_frame = before_frame
        for _ in range(planned.action.hold_frames):
            after_frame = self.source.read()
        self.actuator.neutral_menu()

        after = self.encoder.encode(after_frame)
        prediction_error = latent_distance(planned.predicted_latent, after)
        record = TransitionRecord(
            before_frame.index, planned.action, after_frame.index, before, after,
            planned.predicted_latent, prediction_error,
        )
        self.transitions.append(record)
        self.predictor.observe(before, planned.action, after)
        loss = self.predictor.fit()

        before_distance = latent_distance(before, goal)
        after_distance = latent_distance(after, goal)
        fallback_label = None
        fallback_confirmed = False
        if self.fallback is not None and prediction_error > self.max_prediction_error:
            fallback_label = self.fallback.classify(after_frame)
            fallback_confirmed = fallback_label == goal_label

        progressed = (
            after_distance <= self.goal_threshold
            or after_distance + self.min_distance_decrease < before_distance
        )
        if fallback_confirmed:
            progressed = True
        return NavigationStep(
            goal_label, planned.action, before_distance, after_distance, prediction_error,
            progressed, fallback_label, fallback_confirmed, loss,
        )

