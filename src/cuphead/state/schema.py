"""The fused agent state: symbolic facts + learned latent.

Deliberately hybrid. The symbolic half is exact (the HUD is deterministic pixels)
and is what the strategist and the evaluation harness read. The latent half is
learned and is what the world model rolls forward. Neither alone is sufficient:
pure latents are undebuggable, pure symbols are brittle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class SymbolicState:
    """Exactly-known facts, read from the HUD and the pink mask."""

    frame: int
    hp: int
    max_hp: int
    super_cards: float
    weapon: str
    boss: str
    phase_id: str
    parryable_count: int = 0
    player_xy: Optional[tuple[float, float]] = None

    @property
    def health_fraction(self) -> float:
        return self.hp / self.max_hp if self.max_hp else 0.0

    @property
    def can_super(self) -> bool:
        return self.super_cards >= 1.0


@dataclass
class AgentState:
    """What the planner sees each decision step."""

    symbolic: SymbolicState
    latent: Sequence[float] = field(default_factory=tuple)
    phase_progress: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        s = self.symbolic
        return (
            f"f{s.frame} {s.boss}/{s.phase_id} hp={s.hp}/{s.max_hp} "
            f"cards={s.super_cards:.1f} parryable={s.parryable_count} "
            f"progress={self.phase_progress:.2f}"
        )
