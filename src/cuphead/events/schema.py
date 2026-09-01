"""The discrete event vocabulary.

Every replay reduces to a few hundred of these. That compression is what makes the
LLM strategist viable: it reads event traces, never pixels, and a hundred traces
fit comfortably in context where a hundred videos do not.

Events are also the supervision signal for the world model's hit-probability head
and the unit of measurement for the evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class EventKind(str, Enum):
    PHASE_ENTER = "PHASE_ENTER"
    HIT_TAKEN = "HIT_TAKEN"
    PARRY = "PARRY"
    SUPER_FIRED = "SUPER_FIRED"
    PROJECTILE_WAVE = "PROJECTILE_WAVE"
    DPS_WINDOW = "DPS_WINDOW"
    DEATH = "DEATH"
    KNOCKOUT = "KNOCKOUT"
    REFLEX_OVERRIDE = "REFLEX_OVERRIDE"


@dataclass(frozen=True)
class Event:
    kind: EventKind
    frame: int
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "frame": self.frame, **self.data}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Event":
        data = {k: v for k, v in raw.items() if k not in ("kind", "frame")}
        return cls(kind=EventKind(raw["kind"]), frame=int(raw["frame"]), data=data)


@dataclass
class EventTrace:
    """The full event stream of one attempt."""

    boss: str
    loadout: Dict[str, str] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)
    fps: float = 60.0

    def of_kind(self, kind: EventKind) -> List[Event]:
        return [e for e in self.events if e.kind is kind]

    @property
    def outcome(self) -> str:
        if self.of_kind(EventKind.KNOCKOUT):
            return "KNOCKOUT"
        if self.of_kind(EventKind.DEATH):
            return "DEATH"
        return "INCOMPLETE"

    @property
    def ttk_s(self) -> float:
        """Time to kill, seconds. ``inf`` when the attempt did not end in a knockout."""
        ko = self.of_kind(EventKind.KNOCKOUT)
        if not ko:
            return float("inf")
        return ko[-1].frame / self.fps

    @property
    def hits_taken(self) -> int:
        return len(self.of_kind(EventKind.HIT_TAKEN))

    @property
    def parry_conversion(self) -> float:
        """Successful parries / parryable objects presented."""
        parries = self.of_kind(EventKind.PARRY)
        if not parries:
            return 0.0
        good = sum(1 for e in parries if e.data.get("result") == "success")
        return good / len(parries)

    def dps_uptime(self) -> float:
        """Fraction of the attempt spent inside an open DPS window.

        Windows are emitted as paired ``DPS_WINDOW(open)`` / ``DPS_WINDOW(close)``
        events. An unclosed window at the end of the trace runs to the last frame,
        which is the correct reading for a knockout.
        """
        windows = self.of_kind(EventKind.DPS_WINDOW)
        if not windows or not self.events:
            return 0.0
        last_frame = max(e.frame for e in self.events)
        if last_frame <= 0:
            return 0.0

        total = 0
        open_at: int | None = None
        for e in windows:
            state = e.data.get("state")
            if state == "open" and open_at is None:
                open_at = e.frame
            elif state == "close" and open_at is not None:
                total += e.frame - open_at
                open_at = None
        if open_at is not None:
            total += last_frame - open_at
        return total / last_frame

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boss": self.boss,
            "loadout": self.loadout,
            "fps": self.fps,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EventTrace":
        return cls(
            boss=raw["boss"],
            loadout=raw.get("loadout", {}),
            fps=float(raw.get("fps", 60.0)),
            events=[Event.from_dict(e) for e in raw.get("events", [])],
        )
