"""The statistical gate.

A behavioural change ships only when the numbers say so. This module is the
single implementation of that rule, in pure stdlib, so it runs everywhere --
including inside the babysitter loop before any ML dependency exists.

The primary metric is median time-to-kill, because deaths make the mean useless.
Significance is a bootstrap CI on the *difference in medians*, which needs no
distributional assumption and copes with the bimodality that a death rate
inevitably produces.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

MIN_ATTEMPTS_PER_ARM = 30
BOOTSTRAP_SAMPLES = 10_000


@dataclass
class ArmResult:
    """One arm of an A/B comparison: N attempts against a fixed configuration."""

    name: str
    ttk_s: List[float] = field(default_factory=list)
    deaths: int = 0
    attempts: int = 0
    dps_uptime: Optional[float] = None
    parry_conversion: Optional[float] = None
    latency_p99_ms: Optional[float] = None

    @property
    def n(self) -> int:
        return self.attempts or len(self.ttk_s)

    @property
    def death_rate(self) -> float:
        return (self.deaths / self.n) if self.n else 0.0

    @property
    def median_ttk(self) -> float:
        return statistics.median(self.ttk_s) if self.ttk_s else float("inf")

    @property
    def p10_ttk(self) -> float:
        if not self.ttk_s:
            return float("inf")
        ordered = sorted(self.ttk_s)
        idx = max(0, int(round(0.10 * (len(ordered) - 1))))
        return ordered[idx]


@dataclass
class GateVerdict:
    passed: bool
    reasons: List[str]
    metrics: Dict[str, float]

    @property
    def verdict(self) -> str:
        return "SUCCESSFUL" if self.passed else "FAILED"


def bootstrap_median_diff_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI on ``median(candidate) - median(baseline)``.

    Negative values mean the candidate is faster. A CI entirely below zero is the
    evidence the gate requires.
    """
    if not baseline or not candidate:
        raise ValueError("both arms need at least one observation")

    rng = random.Random(seed)
    base = list(baseline)
    cand = list(candidate)
    nb, nc = len(base), len(cand)

    diffs: List[float] = []
    for _ in range(samples):
        b = statistics.median([base[rng.randrange(nb)] for _ in range(nb)])
        c = statistics.median([cand[rng.randrange(nc)] for _ in range(nc)])
        diffs.append(c - b)

    diffs.sort()
    lo_idx = int((alpha / 2) * (samples - 1))
    hi_idx = int((1 - alpha / 2) * (samples - 1))
    return diffs[lo_idx], diffs[hi_idx]


def evaluate(
    baseline: ArmResult,
    candidate: ArmResult,
    *,
    min_attempts: int = MIN_ATTEMPTS_PER_ARM,
    death_rate_regression_bound: float = 0.05,
    latency_budget_ms: Optional[float] = None,
    stalling_detected: bool = False,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> GateVerdict:
    """Apply the full gate from ``CLAUDE.md``.

    Every criterion is checked and every failure is reported -- the caller gets
    the complete picture in one pass rather than the first thing that broke.
    """
    reasons: List[str] = []

    for arm in (baseline, candidate):
        if arm.n < min_attempts:
            reasons.append(f"arm {arm.name!r} has n={arm.n}, below the minimum {min_attempts}")

    if candidate.median_ttk >= baseline.median_ttk:
        reasons.append(
            f"median TTK did not improve: {baseline.median_ttk:.2f}s -> {candidate.median_ttk:.2f}s"
        )

    ci_lo = ci_hi = float("nan")
    if baseline.ttk_s and candidate.ttk_s:
        ci_lo, ci_hi = bootstrap_median_diff_ci(
            baseline.ttk_s, candidate.ttk_s, samples=bootstrap_samples, seed=seed
        )
        if not (ci_hi < 0):
            reasons.append(
                f"bootstrap CI on the median difference does not exclude zero: "
                f"[{ci_lo:.2f}, {ci_hi:.2f}]"
            )
    else:
        reasons.append("missing TTK observations in one or both arms")

    death_delta = candidate.death_rate - baseline.death_rate
    if death_delta > death_rate_regression_bound:
        reasons.append(
            f"death rate regressed by {death_delta:.3f}, above the bound "
            f"{death_rate_regression_bound:.3f}"
        )

    if stalling_detected:
        reasons.append("stalling detector fired: the agent is buying survival with time")

    if latency_budget_ms is not None and candidate.latency_p99_ms is not None:
        if candidate.latency_p99_ms > latency_budget_ms:
            reasons.append(
                f"latency canary out of budget: p99 {candidate.latency_p99_ms:.1f} ms > "
                f"{latency_budget_ms:.1f} ms"
            )

    metrics = {
        "baseline_median_ttk_s": baseline.median_ttk,
        "candidate_median_ttk_s": candidate.median_ttk,
        "baseline_p10_ttk_s": baseline.p10_ttk,
        "candidate_p10_ttk_s": candidate.p10_ttk,
        "median_delta_s": candidate.median_ttk - baseline.median_ttk,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "baseline_death_rate": baseline.death_rate,
        "candidate_death_rate": candidate.death_rate,
        "baseline_n": float(baseline.n),
        "candidate_n": float(candidate.n),
    }

    return GateVerdict(passed=not reasons, reasons=reasons, metrics=metrics)


def format_verdict(v: GateVerdict) -> str:
    """One-screen human summary. QA pastes this verbatim into the task record."""
    m = v.metrics
    lines = [
        f"VERDICT: {v.verdict}",
        f"  median TTK   {m['baseline_median_ttk_s']:.2f}s -> {m['candidate_median_ttk_s']:.2f}s "
        f"(delta {m['median_delta_s']:+.2f}s)",
        f"  p10 TTK      {m['baseline_p10_ttk_s']:.2f}s -> {m['candidate_p10_ttk_s']:.2f}s",
        f"  bootstrap CI [{m['ci_low']:.2f}, {m['ci_high']:.2f}]",
        f"  death rate   {m['baseline_death_rate']:.3f} -> {m['candidate_death_rate']:.3f}",
        f"  n            {int(m['baseline_n'])} vs {int(m['candidate_n'])}",
    ]
    for reason in v.reasons:
        lines.append(f"  FAIL: {reason}")
    return "\n".join(lines)
