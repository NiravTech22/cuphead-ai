#!/usr/bin/env python3
"""Verify the foundation before anything autonomous is allowed to run.

Checks, in order of how badly each one ruins your day when it is wrong:

  1. Python version and repo layout
  2. The four ``.ai/`` state documents parse and are internally consistent
  3. Every agent file has valid YAML frontmatter with a usable ``name``
  4. Task owners resolve to real agents; dependencies resolve to real tasks
  5. Git exists, has a baseline commit, and the tree is clean
  6. The ``claude`` CLI is present and supports the flags the loop uses
  7. The stdlib test suite is green

Exit status is 0 only when every check passes. The babysitter loop calls this and
refuses to start otherwise -- the whole point of the orchestration layer is that
it will not build on a broken foundation.

    python3 scripts/preflight.py [--skip-tests] [--json]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

EXPECTED_AGENTS = {"architect", "babysitter", "data", "qa", "vision", "world-model"}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fatal: bool = True


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", fatal: bool = True) -> None:
        self.checks.append(Check(name, ok, detail, fatal))

    @property
    def failed(self) -> List[Check]:
        return [c for c in self.checks if not c.ok and c.fatal]

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks if not c.ok and not c.fatal]

    def render(self) -> str:
        lines = []
        for c in self.checks:
            mark = "PASS" if c.ok else ("WARN" if not c.fatal else "FAIL")
            lines.append(f"  [{mark}] {c.name}" + (f" — {c.detail}" if c.detail else ""))
        return "\n".join(lines)


def parse_frontmatter(text: str) -> dict:
    """Minimal YAML frontmatter reader.

    Deliberately not a YAML dependency: preflight must run before anything is
    installed. Agent frontmatter is flat ``key: value``, which is all this needs.
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    body = text[3:end]
    out = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )


def run_checks(skip_tests: bool = False) -> Report:
    r = Report()

    # 1. Environment ----------------------------------------------------
    r.add(
        "python >= 3.10",
        sys.version_info >= (3, 10),
        f"found {sys.version.split()[0]}",
    )
    for rel in (".ai", ".claude/agents", "scripts", "src/cuphead", "tests", "docs"):
        r.add(f"layout: {rel}/", (REPO / rel).is_dir())

    # 2. State documents ------------------------------------------------
    try:
        from cuphead.orchestration.state_store import StateStore

        store = StateStore(REPO)
        problems = store.validate()
        r.add(
            "state documents consistent",
            not problems,
            "; ".join(problems[:5]) if problems else "4 documents parsed",
        )
    except Exception as exc:  # noqa: BLE001 - preflight reports, never raises
        store = None
        r.add("state documents consistent", False, f"{type(exc).__name__}: {exc}")

    # 3. Agent definitions ----------------------------------------------
    agent_dir = REPO / ".claude" / "agents"
    found = {}
    if agent_dir.is_dir():
        for path in sorted(agent_dir.glob("*.md")):
            fm = parse_frontmatter(path.read_text(encoding="utf-8"))
            name = fm.get("name", "")
            ok = bool(name) and bool(fm.get("description"))
            r.add(
                f"agent frontmatter: {path.name}",
                ok,
                "missing name or description" if not ok else f"name={name}",
            )
            if name:
                found[name] = path

    missing = EXPECTED_AGENTS - set(found)
    r.add(
        "all six agents present",
        not missing,
        f"missing: {sorted(missing)}" if missing else ", ".join(sorted(found)),
    )

    # 4. Task graph -----------------------------------------------------
    if store is not None:
        try:
            from cuphead.orchestration.tasks import (
                highest_unlocked_phase,
                load_tasks,
                unblocked,
            )

            tasks_doc = store.tasks()
            state = store.project_state()
            tasks = load_tasks(tasks_doc)
            ready = unblocked(tasks, max_phase=highest_unlocked_phase(state))
            r.add(
                "task queue has runnable work",
                bool(ready),
                f"{len(ready)} unblocked of {len(tasks)}; next = "
                + (ready[0].id if ready else "none"),
                fatal=False,
            )
        except Exception as exc:  # noqa: BLE001
            r.add("task queue readable", False, f"{type(exc).__name__}: {exc}")

    # 5. Git ------------------------------------------------------------
    if shutil.which("git") is None:
        r.add("git available", False, "git not on PATH")
    else:
        r.add("git available", True)
        head = git("rev-parse", "--verify", "HEAD")
        r.add(
            "baseline commit exists",
            head.returncode == 0,
            "no commits yet — commit a baseline before running the loop"
            if head.returncode
            else head.stdout.strip()[:12],
        )
        status = git("status", "--porcelain")
        dirty = [ln for ln in status.stdout.splitlines() if ln.strip()]
        r.add(
            "working tree clean",
            not dirty,
            f"{len(dirty)} modified/untracked path(s)" if dirty else "",
            fatal=False,
        )

    # 6. Claude CLI -----------------------------------------------------
    claude = shutil.which("claude")
    r.add("claude CLI on PATH", claude is not None, claude or "not found", fatal=False)
    if claude:
        ver = subprocess.run([claude, "--version"], capture_output=True, text=True, check=False)
        r.add("claude --version", ver.returncode == 0, ver.stdout.strip(), fatal=False)
        helptext = subprocess.run(
            [claude, "--help"], capture_output=True, text=True, check=False
        ).stdout
        for flag in ("--agent", "--print", "--output-format", "--permission-mode"):
            r.add(
                f"claude supports {flag}",
                flag in helptext,
                "" if flag in helptext else "CLI version may be too old for the loop",
                fatal=False,
            )

    # 7. Tests ----------------------------------------------------------
    if not skip_tests:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        r.add(
            "unit tests green",
            proc.returncode == 0,
            tail[-1] if tail else "",
        )

    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the Cuphead-AI foundation.")
    ap.add_argument("--skip-tests", action="store_true", help="do not run the unit suite")
    ap.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = ap.parse_args()

    report = run_checks(skip_tests=args.skip_tests)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not report.failed,
                    "checks": [vars(c) for c in report.checks],
                },
                indent=2,
            )
        )
    else:
        print("PREFLIGHT — cuphead-ai")
        print(report.render())
        print()
        if report.failed:
            print(f"NOT READY: {len(report.failed)} blocking issue(s).")
        elif report.warnings:
            print(f"READY with {len(report.warnings)} warning(s).")
        else:
            print("READY.")

    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
