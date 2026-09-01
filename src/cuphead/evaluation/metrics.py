"""Per-attempt and per-arm metrics, computed from event traces.

The evaluation harness never looks at pixels. It reads event traces, which means
regression testing runs offline against frozen replays and costs nothing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..events.schema import EventKind, EventTrace
from ..orchestration.gates import ArmResult


@dataclass
class AttemptMetrics:
    boss: str
    outcome: str
    ttk_s: float
    hits_taken: int
    dps_uptime: float
    parry_conversion: float
    supers_fired: int
    unspent_cards: float = 0.0

    @classmethod
    def from_trace(cls, trace: EventTrace) -> "AttemptMetrics":
        return cls(
            boss=trace.boss,
            outcome=trace.outcome,
            ttk_s=trace.ttk_s,
            hits_taken=trace.hits_taken,
            dps_uptime=trace.dps_uptime(),
            parry_conversion=trace.parry_conversion,
            supers_fired=len(trace.of_kind(EventKind.SUPER_FIRED)),
        )


def aggregate(name: str, traces: Sequence[EventTrace]) -> ArmResult:
    """Fold a set of attempts into one arm of an A/B comparison.

    Deaths contribute to ``deaths`` and are excluded from the TTK sample -- a
    death has no time-to-kill, and imputing one (as infinity, or as the timeout)
    would corrupt the median that the gate is built on. The death rate carries
    that information instead, which is why the gate checks both.
    """
    metrics = [AttemptMetrics.from_trace(t) for t in traces]
    kills = [m.ttk_s for m in metrics if m.outcome == "KNOCKOUT"]
    deaths = sum(1 for m in metrics if m.outcome == "DEATH")

    arm = ArmResult(name=name, ttk_s=kills, deaths=deaths, attempts=len(metrics))
    if metrics:
        arm.dps_uptime = statistics.fmean(m.dps_uptime for m in metrics)
        arm.parry_conversion = statistics.fmean(m.parry_conversion for m in metrics)
    return arm


def per_boss(traces: Sequence[EventTrace]) -> Dict[str, ArmResult]:
    """Split an arm by boss.

    Never accept an aggregate number that hides a per-boss regression: a model
    that gains 4 seconds on Goopy and loses 6 on Grim Matchstick reads as an
    improvement in the pooled median if Goopy is over-represented.
    """
    buckets: Dict[str, List[EventTrace]] = {}
    for t in traces:
        buckets.setdefault(t.boss, []).append(t)
    return {boss: aggregate(boss, group) for boss, group in sorted(buckets.items())}
