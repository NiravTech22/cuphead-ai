"""The factorized, pruned action space.

Cuphead's raw input product space is 3 x 3 x 2 x 2 x 2 x 8 = 576 combinations, most
of which are illegal or useless. Sampling that space in CEM wastes the sample
budget on nonsense. Pruning it to the ~60 combinations that can actually occur is
worth more planner quality than any amount of extra rollouts.

An action is a *held button state* for the duration of one decision window
(4 frames at 60 Hz), not an event.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

MOVE = ("none", "left", "right")
VERT = ("none", "jump", "duck")
AIM8 = ("n", "ne", "e", "se", "s", "sw", "w", "nw")

#: Frames of input held per planner decision.
ACTION_REPEAT = 4
#: Planner decision rate, Hz.
DECISION_HZ = 60.0 / ACTION_REPEAT


@dataclass(frozen=True)
class Action:
    """One held-input state."""

    move: str = "none"
    vert: str = "none"
    dash: bool = False
    shoot: bool = False
    lock: bool = False
    aim: str = "e"

    def __post_init__(self) -> None:
        if self.move not in MOVE:
            raise ValueError(f"bad move: {self.move!r}")
        if self.vert not in VERT:
            raise ValueError(f"bad vert: {self.vert!r}")
        if self.aim not in AIM8:
            raise ValueError(f"bad aim: {self.aim!r}")

    def is_legal(self) -> bool:
        """Reject combinations the game cannot express or that never help.

        Each rule below is a fact about Cuphead, not a heuristic:

        * Dashing while ducking is not a distinct state -- the duck is dropped.
        * Dashing while jumping *is* an air-dash, is legal, and matters.
        * Aim-lock roots the player, so a lock with a movement direction is not a
          reachable state.
        * The dash cancels the shot, so dash+shoot collapses to dash.
        * Ducking overrides the aim stance, so lock+duck is not distinct.
        * Holding aim-lock without firing and without dashing does nothing at all.
        * Aim direction only matters under lock; without it the field duplicates
          ``move`` and multiplies the space by eight for free.
        """
        if self.dash and self.vert == "duck":
            return False
        if self.dash and self.shoot:
            return False
        if not self.lock and self.aim != "e":
            return False
        if self.lock and self.move != "none":
            return False
        if self.lock and self.vert == "duck":
            return False
        if self.lock and not self.shoot and not self.dash:
            return False
        return True

    def to_buttons(self) -> dict:
        """Lower to the virtual-gamepad button/axis state."""
        dx = {"none": 0.0, "left": -1.0, "right": 1.0}[self.move]
        dy = {"none": 0.0, "jump": 0.0, "duck": -1.0}[self.vert]
        if self.lock:
            dx, dy = _aim_vector(self.aim)
        return {
            "stick_x": dx,
            "stick_y": dy,
            "a": self.vert == "jump",
            "x": self.shoot,
            "b": self.dash,
            "rt": self.lock,
        }


def _aim_vector(aim: str) -> Tuple[float, float]:
    s = 0.7071
    return {
        "n": (0.0, 1.0),
        "ne": (s, s),
        "e": (1.0, 0.0),
        "se": (s, -s),
        "s": (0.0, -1.0),
        "sw": (-s, -s),
        "w": (-1.0, 0.0),
        "nw": (-s, s),
    }[aim]


@lru_cache(maxsize=1)
def enumerate_actions() -> Tuple[Action, ...]:
    """Every legal action, in a stable order. The planner indexes into this."""
    actions: List[Action] = []
    for move in MOVE:
        for vert in VERT:
            for dash in (False, True):
                for shoot in (False, True):
                    for lock in (False, True):
                        for aim in (AIM8 if lock else ("e",)):
                            a = Action(move, vert, dash, shoot, lock, aim)
                            if a.is_legal():
                                actions.append(a)
    return tuple(actions)


def action_count() -> int:
    return len(enumerate_actions())


def index_of(action: Action) -> int:
    return enumerate_actions().index(action)


#: The reflex layer may only override the planner with these.
REFLEX_WHITELIST = ("parry", "dash")


@dataclass(frozen=True)
class ReflexOverride:
    """A reflex-layer preemption of the planner's action.

    Arbitration rule: the override applies only if ``kind`` is whitelisted and
    ``confidence`` clears the per-boss threshold. Every override is logged with
    its outcome, because override precision is itself a gated metric.
    """

    kind: str
    confidence: float

    def applies(self, threshold: float) -> bool:
        return self.kind in REFLEX_WHITELIST and self.confidence >= threshold

    def merge(self, planned: Action) -> Action:
        """Apply the override on top of the planner's action, preserving intent.

        The reflex layer decides *what to do in the next 90 ms*; the planner
        decided *where to be*. So the movement direction survives -- a parry that
        also abandons positioning costs more than it saves.

        Aim-lock does not survive either override, because both are physical
        evasions and aim-lock roots the player. The result is always a legal
        action, which the planner's executor relies on.
        """
        if self.kind == "parry":
            return Action(planned.move, "jump", False, planned.shoot, False, "e")
        if self.kind == "dash":
            return Action(planned.move, "none", True, False, False, "e")
        return planned
