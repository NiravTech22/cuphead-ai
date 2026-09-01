#!/usr/bin/env python3
"""Measure round-trip actuate -> observe latency and gate it against the harness budget.

Implements the acceptance criterion for task ``harness-capture-latency``:
p50 < 30 ms and p99 < 50 ms over 200 trials, written to ``experiments/`` as a
JSON record. See ``docs/SPEEDRUN_PLAN.md`` section 2.1.

Two modes:

    python3 scripts/latency_canary.py --synthetic
        Self-test. Runs against the synthetic actuator/frame-source pair in
        ``evaluation.latency`` with a known, injected response delay and no
        real hardware. This validates the *measurement code* -- it recovers
        the injected delay to the millisecond -- and exercises the full CLI
        and reporting path. It is NOT evidence that the real capture and
        actuation pipeline meets the budget; the record it writes is tagged
        ``"mode": "synthetic"`` so it can never be mistaken for that.

    python3 scripts/latency_canary.py --real
        The actual measurement, against ``perception.open_screen_source``
        and ``control.open_vgamepad_actuator``. Run this on the machine
        actually running Cuphead, with the game window visible and the
        virtual pad wired up as the game's active controller. This is what
        closes task ``harness-capture-latency`` for real.

Either way the script exits non-zero if the budget is missed, so it can gate
a training session the same way ``preflight.py`` gates the loop: don't start
collecting data, let alone training, on a harness that hasn't proven it can
keep up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.control.action_space import Action, enumerate_actions  # noqa: E402
from cuphead.evaluation.latency import (  # noqa: E402
    LatencyTimeoutError,
    measure_many,
    synthetic_round_trip,
)
from cuphead.orchestration.state_store import utcnow  # noqa: E402

DEFAULT_P50_BUDGET_MS = 30.0
DEFAULT_P99_BUDGET_MS = 50.0
DEFAULT_TRIALS = 200


def sample_actions(n: int) -> list[Action]:
    """A varied, deterministic sequence of legal actions to probe with.

    Cycling through the full legal action set (rather than repeating one
    action) means the canary also incidentally checks that every action the
    planner might choose actually produces a visible response -- an action
    that silently does nothing would otherwise hide behind a passing latency
    number measured only on, say, "move right."
    """
    actions = enumerate_actions()
    return [actions[i % len(actions)] for i in range(n)]


def run(mode: str, trials: int, p50_budget_ms: float, p99_budget_ms: float) -> dict:
    actions = sample_actions(trials)

    if mode == "synthetic":
        actuator, source = synthetic_round_trip(respond_after_frames=1, fps=60.0)
    else:
        from cuphead.control.actuator import open_vgamepad_actuator
        from cuphead.perception.capture import open_screen_source

        actuator = open_vgamepad_actuator()
        source = open_screen_source()

    try:
        report = measure_many(actuator, source, actions)
    except LatencyTimeoutError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print(
            "No visible response was detected. In --real mode this usually means the "
            "capture region doesn't cover the game window, or the virtual pad isn't the "
            "game's active input device.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    finally:
        actuator.close()
        source.close()

    passed = report.within_budget(p50_ms=p50_budget_ms, p99_ms=p99_budget_ms)

    record = {
        "task": "harness-capture-latency",
        "mode": mode,
        "recorded_at": utcnow(),
        "trials": trials,
        "budget_ms": {"p50": p50_budget_ms, "p99": p99_budget_ms},
        "passed": passed,
        **report.to_dict(),
    }
    if mode == "synthetic":
        record["caveat"] = (
            "Synthetic self-test only. Validates the measurement code against a known "
            "injected delay; is not evidence about the real capture/actuation pipeline. "
            "Re-run with --real on the machine running Cuphead before closing this task."
        )
    return record


def write_record(record: dict) -> Path:
    experiments_dir = REPO / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = experiments_dir / f"latency_canary_{record['mode']}_{stamp}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--synthetic", action="store_const", dest="mode", const="synthetic")
    mode_group.add_argument("--real", action="store_const", dest="mode", const="real")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--p50-budget-ms", type=float, default=DEFAULT_P50_BUDGET_MS)
    ap.add_argument("--p99-budget-ms", type=float, default=DEFAULT_P99_BUDGET_MS)
    args = ap.parse_args()

    record = run(args.mode, args.trials, args.p50_budget_ms, args.p99_budget_ms)
    path = write_record(record)

    print(f"mode={record['mode']} n={record['n']}")
    print(f"  p50 {record['p50_ms']:.2f} ms  (budget < {args.p50_budget_ms:.0f} ms)")
    print(f"  p99 {record['p99_ms']:.2f} ms  (budget < {args.p99_budget_ms:.0f} ms)")
    print(f"  mean {record['mean_ms']:.2f} ms   max {record['max_ms']:.2f} ms")
    print(f"  record written to {path.relative_to(REPO)}")
    if record["mode"] == "synthetic":
        print(f"  NOTE: {record['caveat']}")

    if not record["passed"]:
        print("FAIL: outside the harness latency budget", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
