"""Deterministic integration fixture, explicitly not an actual Cuphead victory."""

from __future__ import annotations

from ..control.timed_input import TimedExecutor
from ..memory.latent_bank import LatentBank
from ..orchestration.memory_agent import MemoryAgent
from ..perception.capture import Frame
from ..perception.landmarks import Scene
from ..planner.memory_planner import MemoryPlanner
from ..strategist.landmark_route import LandmarkRoute, RouteEdge
from ..world_model.knn_dynamics import KNNDynamics


class SyntheticGame:
    labels = ("title", "overworld", "entrance", "boss", "death", "victory")

    def __init__(self):
        self.state, self.index = "title", 0
        self.active = None
        self.deaths = 0
        self.references = {label: label for label in self.labels}

    def read(self):
        self.index += 1
        return Frame(
            self.index, float(self.index), self.state, self.labels.index(self.state)
        )

    def classify(self, frame):
        label = frame.payload
        mode = {
            "title": "menu",
            "entrance": "menu",
            "overworld": "overworld",
            "boss": "combat",
            "death": "death",
            "victory": "victory",
        }[label]
        return Scene(
            label,
            mode,
            label in {"death", "victory"},
            0 if label == "death" else 3,
            label == "victory",
        )

    def send_buttons(self, action):
        self.active = action.name
        expected = {
            "title": "confirm",
            "overworld": "right",
            "entrance": "confirm",
            "death": "confirm",
            "boss": "jump_shoot",
        }
        if self.state == "boss" and action.name == "shoot":
            self.state = "death"
            self.deaths += 1
        elif action.name == expected.get(self.state):
            self.state = {
                "title": "overworld",
                "overworld": "entrance",
                "entrance": "boss",
                "death": "boss",
                "boss": "victory",
            }[self.state]

    def neutral(self):
        self.active = None

    def close(self):
        self.neutral()


class FixtureEncoder:
    fingerprint = "synthetic-onehot-v1-NOT-PRETRAINED"
    output_dim = len(SyntheticGame.labels)

    def encode(self, value, *, low_detail=True):
        label = getattr(value, "payload", value)
        return tuple(float(label == item) for item in SyntheticGame.labels)

    def reset(self):
        pass


def run_synthetic(*, bank=None, emit=None, max_steps=50):
    game, encoder = SyntheticGame(), FixtureEncoder()
    bank = (
        bank
        if bank is not None
        else LatentBank(encoder.output_dim, encoder.fingerprint)
    )
    planner = MemoryPlanner(KNNDynamics(bank, radius=0.1))
    route = LandmarkRoute(
        [
            RouteEdge("title", "overworld", "confirm"),
            RouteEdge("overworld", "entrance", "right"),
            RouteEdge("entrance", "boss", "confirm"),
            RouteEdge("death", "boss", "confirm"),
            RouteEdge("boss", "victory", "shoot"),
        ],
        "victory",
    )
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    executor = TimedExecutor(game, clock=lambda: clock[0], sleep=sleep)
    agent = MemoryAgent(
        source=game,
        encoder=encoder,
        verifier=game,
        executor=executor,
        bank=bank,
        planner=planner,
        route=route,
        emit=emit,
    )
    trace = []
    for _ in range(max_steps):
        result = agent.step()
        trace.append(result)
        if result.status == "won":
            break
    return {
        "environment": "synthetic_fixture",
        "cuphead_victory_verified": False,
        "fixture_completed": trace[-1].status == "won",
        "steps": len(trace),
        "deaths": game.deaths,
        "transitions": len(bank),
        "cold_starts": sum(row.reason == "cold_start" for row in trace),
        "memory_retrievals": sum(row.reason == "designated_memory" for row in trace),
    }, bank
