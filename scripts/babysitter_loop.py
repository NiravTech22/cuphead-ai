#!/usr/bin/env python3
"""The outer autonomous loop.

    read state -> select task -> delegate to worker -> QA gate -> record -> commit

Design constraints, learned the hard way by everyone who has run one of these
overnight:

* **Preflight or nothing.** The loop refuses to start on a broken foundation or a
  dirty tree. A dirty tree means something upstream went wrong and committing on
  top of it destroys the evidence.
* **The worker never grades itself.** Implementation and acceptance are two
  separate ``claude`` invocations with two different agents. This is the entire
  reason the QA agent exists.
* **Three strikes.** Three consecutive failures on the same task halts the loop
  and escalates, rather than burning budget rediscovering the same wall.
* **One task per commit.** A commit closing three tasks cannot be reverted cleanly.
* **Budget caps.** Per-iteration wall-clock and dollar caps, because an unattended
  loop with neither is a way to lose a weekend and a credit limit.

Usage:

    python3 scripts/babysitter_loop.py --once --dry-run     # show the plan
    python3 scripts/babysitter_loop.py --once               # one supervised iteration
    python3 scripts/babysitter_loop.py --max-iterations 20  # autonomous run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cuphead.orchestration.state_store import StateStore, utcnow  # noqa: E402
from cuphead.orchestration.tasks import (  # noqa: E402
    Task,
    refresh_blocked,
    select_next,
    set_status,
)

WORKER_AGENTS = {"architect", "vision", "world-model", "data"}
MAX_CONSECUTIVE_FAILURES = 3


@dataclass
class LoopConfig:
    dry_run: bool = False
    max_iterations: int = 1
    model: str = "opus"
    permission_mode: str = "acceptEdits"
    timeout_s: int = 1800
    max_budget_usd: Optional[float] = None
    commit: bool = True


class Halt(RuntimeError):
    """Raised to stop the loop deliberately, with a reason worth reading."""


# -- shelling out to claude --------------------------------------------


def run_agent(agent: str, prompt: str, cfg: LoopConfig) -> str:
    """Invoke one agent non-interactively and return its text output.

    ``--agent`` selects the definition in ``.claude/agents/``. Output is parsed
    from ``--output-format json`` so a truncated or errored run is detectable
    rather than silently treated as a successful empty response.
    """
    cmd = [
        "claude",
        "--print",
        "--agent",
        agent,
        "--model",
        cfg.model,
        "--permission-mode",
        cfg.permission_mode,
        "--output-format",
        "json",
    ]
    if cfg.max_budget_usd is not None:
        cmd += ["--max-budget-usd", str(cfg.max_budget_usd)]
    cmd.append(prompt)

    if cfg.dry_run:
        return f"[dry-run] would invoke: {' '.join(cmd[:-1])} <prompt of {len(prompt)} chars>"

    proc = subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True, timeout=cfg.timeout_s, check=False
    )
    if proc.returncode != 0:
        raise Halt(f"agent {agent!r} exited {proc.returncode}: {proc.stderr.strip()[:500]}")

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return proc.stdout.strip()

    if isinstance(payload, dict):
        if payload.get("is_error"):
            raise Halt(f"agent {agent!r} reported an error: {payload.get('result')}")
        return str(payload.get("result", "")).strip()
    return proc.stdout.strip()


# -- git ----------------------------------------------------------------


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)


def tree_is_clean() -> bool:
    return not git("status", "--porcelain").stdout.strip()


def commit(message: str) -> Optional[str]:
    git("add", "-A")
    if not git("diff", "--cached", "--quiet").returncode:
        return None  # nothing staged
    res = git("commit", "-m", message)
    if res.returncode != 0:
        raise Halt(f"commit failed: {res.stderr.strip()[:300]}")
    return git("rev-parse", "HEAD").stdout.strip()[:12]


# -- prompts ------------------------------------------------------------


def worker_prompt(task: Task) -> str:
    return f"""Implement task `{task.id}` from `.ai/tasks.json`.

TITLE: {task.title}
LAYER: {task.layer}
PHASE: {task.phase}

ACCEPTANCE CRITERION (you do not get to grade this — QA does):
{task.acceptance}

{('NOTES: ' + task.notes) if task.notes else ''}

Rules:
- Read `CLAUDE.md` and the relevant section of `docs/SPEEDRUN_PLAN.md` first.
- Stay inside your layer. A change reaching across three layers goes to `architect`.
- Add tests that a fresh checkout can run with `python3 -m unittest discover -s tests`.
- Do not modify `.ai/*.json` — the loop owns those.
- Do not commit; the loop commits.

When you are done, end your reply with a single line:
STATUS: IMPLEMENTED   or   STATUS: BLOCKED <one-line reason>
"""


def qa_prompt(task: Task, worker_report: str) -> str:
    return f"""Evaluate task `{task.id}` against its acceptance criterion.

ACCEPTANCE CRITERION:
{task.acceptance}

The implementing agent reported:
---
{worker_report[-4000:]}
---

Treat that report as a claim, not as evidence. Run the checks yourself:
- `python3 -m unittest discover -s tests` must be green with no silent skips.
- Any numeric threshold in the criterion must be measured and reported with its n.
- Report per-boss numbers where the criterion is per-boss; never an aggregate that
  could hide a regression.

End your reply with exactly one line:
VERDICT: VALIDATED   or   VERDICT: SUCCESSFUL   or   VERDICT: FAILED <one-line reason>
"""


def parse_verdict(text: str) -> tuple[str, str]:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            rest = line.split(":", 1)[1].strip()
            word, _, reason = rest.partition(" ")
            return word.upper(), reason.strip()
    return "FAILED", "QA did not emit a VERDICT line"


# -- the loop -----------------------------------------------------------


def preflight(cfg: LoopConfig) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "preflight.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise Halt("preflight failed — fix the foundation before running the loop")
    if not cfg.dry_run and not tree_is_clean():
        raise Halt(
            "working tree is dirty at loop start; commit or stash first "
            "(a dirty tree means something upstream went wrong)"
        )


def iterate(store: StateStore, cfg: LoopConfig, iteration: int) -> bool:
    """Run one iteration. Returns False when there is no more work."""
    state = store.project_state()
    tasks_doc = store.tasks()
    tasks_doc = refresh_blocked(tasks_doc, state)
    store.write("tasks.json", tasks_doc)

    task = select_next(tasks_doc, state)
    if task is None:
        print("No unblocked tasks. Ask `architect` to decompose the next phase.")
        return False

    print(f"\n=== iteration {iteration} — {task.id} → {task.owner} ===")
    print(f"    {task.title}")
    print(f"    acceptance: {task.acceptance[:120]}")

    if task.owner not in WORKER_AGENTS:
        raise Halt(f"task {task.id} has non-worker owner {task.owner!r}")

    status_doc = store.agent_status()
    failures = status_doc["agents"][task.owner].get("consecutive_failures", 0)
    if failures >= MAX_CONSECUTIVE_FAILURES:
        raise Halt(
            f"agent {task.owner!r} has {failures} consecutive failures — escalating to a human"
        )

    if cfg.dry_run:
        print("\n--- worker prompt ---")
        print(worker_prompt(task))
        print("--- (dry run: stopping before invocation) ---")
        return False

    store.set_agent(task.owner, state="WORKING", current_task=task.id)
    store.write("tasks.json", set_status(store.tasks(), task.id, "IN_PROGRESS"))
    store.set_loop(running=True, iteration=iteration, last_iteration_at=utcnow())

    # 1. implement
    report = run_agent(task.owner, worker_prompt(task), cfg)
    print(f"\n[worker/{task.owner}] {report[-600:]}")

    if "STATUS: BLOCKED" in report.upper():
        reason = report.upper().split("STATUS: BLOCKED", 1)[1].strip()[:200]
        store.record_failure(task.id, task.owner, "worker reported BLOCKED", reason)
        store.write("tasks.json", set_status(store.tasks(), task.id, "BLOCKED"))
        store.set_agent(task.owner, state="IDLE", current_task=None, last_verdict="BLOCKED")
        return True

    store.write("tasks.json", set_status(store.tasks(), task.id, "IMPLEMENTED"))

    # 2. gate
    store.set_agent("qa", state="WORKING", current_task=task.id)
    verdict_text = run_agent("qa", qa_prompt(task, report), cfg)
    verdict, reason = parse_verdict(verdict_text)
    print(f"\n[qa] VERDICT: {verdict} {reason}")
    store.set_agent("qa", state="IDLE", current_task=None, last_verdict=verdict)

    # 3. record
    if verdict in ("VALIDATED", "SUCCESSFUL"):
        store.write(
            "tasks.json",
            set_status(store.tasks(), task.id, verdict, closed_at=utcnow(), evidence=verdict_text[-1500:]),
        )
        store.set_agent(task.owner, state="IDLE", current_task=None, last_verdict=verdict, consecutive_failures=0)
        sha = commit(f"{task.id}: {task.title}\n\nVerdict: {verdict}\n{reason}".strip()) if cfg.commit else None
        print(f"    committed {sha}" if sha else "    nothing to commit")
    else:
        store.record_failure(task.id, task.owner, reason or "QA failed the acceptance check",
                             hypothesis="", evidence={"qa": verdict_text[-1500:]})
        store.write("tasks.json", set_status(store.tasks(), task.id, "OPEN"))
        store.set_agent(
            task.owner,
            state="IDLE",
            current_task=None,
            last_verdict="FAILED",
            consecutive_failures=failures + 1,
        )
        print(f"    FAILED recorded ({failures + 1} consecutive for {task.owner})")

    store.touch_project_state(iteration=iteration)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Cuphead-AI autonomous orchestration loop.")
    ap.add_argument("--once", action="store_true", help="run a single iteration")
    ap.add_argument("--max-iterations", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, invoke nothing")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--permission-mode", default="acceptEdits")
    ap.add_argument("--timeout", type=int, default=1800, help="per-agent wall clock cap, seconds")
    ap.add_argument("--max-budget-usd", type=float, default=None)
    ap.add_argument("--no-commit", action="store_true")
    args = ap.parse_args()

    cfg = LoopConfig(
        dry_run=args.dry_run,
        max_iterations=1 if args.once else args.max_iterations,
        model=args.model,
        permission_mode=args.permission_mode,
        timeout_s=args.timeout,
        max_budget_usd=args.max_budget_usd,
        commit=not args.no_commit,
    )

    store = StateStore(REPO)
    try:
        preflight(cfg)
        store.set_loop(running=True, started_at=utcnow(), halt_reason=None)
        for i in range(1, cfg.max_iterations + 1):
            if not iterate(store, cfg, i):
                break
            time.sleep(1)
    except Halt as exc:
        print(f"\nHALT: {exc}", file=sys.stderr)
        store.set_loop(running=False, halt_reason=str(exc))
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        store.set_loop(running=False, halt_reason="interrupted by user")
        return 130

    store.set_loop(running=False, halt_reason=None)
    print("\nloop finished cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
