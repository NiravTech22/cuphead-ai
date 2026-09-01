# CUPHEAD AI — MASTER ENGINEERING INSTRUCTIONS

You are working on CUPHEAD-AI, a research-grade autonomous Cuphead speedrunning agent.

The system combines:

- visual perception
- latent state representation
- learned world modeling
- action-conditioned prediction
- model-predictive planning
- high-frequency reflex control
- LLM strategic reasoning
- episodic memory
- failure analysis
- iterative learning
- speedrun optimization

The full technical breakdown lives in `docs/SPEEDRUN_PLAN.md`. Read it before making
architectural decisions.

## CORE RULE

Do not treat this project as one monolithic application.

Maintain strict separation between:

1. perception — `src/cuphead/perception/`
2. state estimation — `src/cuphead/state/`
3. event detection — `src/cuphead/events/`
4. world model — `src/cuphead/world_model/`
5. planner — `src/cuphead/planner/`
6. low-level policy — `src/cuphead/policy/`
7. controller — `src/cuphead/control/`
8. LLM strategist — `src/cuphead/strategist/`
9. memory — `src/cuphead/memory/`
10. evaluation — `src/cuphead/evaluation/`
11. orchestration — `src/cuphead/orchestration/`

Layers communicate through the typed interfaces in each package's `__init__.py`. A change
that reaches across three layers is a design smell — raise it with the `architect` agent
before writing it.

## THE THREE TIMESCALES

Every control decision belongs to exactly one timescale. Violating this is the primary
failure mode of the project.

| Timescale | Owner | Rate | Never does |
|---|---|---|---|
| Reflex | `policy/` | 60 Hz | plan, call an LLM |
| Tactical | `planner/` + `world_model/` | 15 Hz | attempt 5-frame parry timing |
| Strategic | `strategist/` | offline | touch the control loop |

**The LLM is never in the 60 Hz loop.** No exceptions, no "just for debugging".

## ENGINEERING RULE

Never claim that something works merely because code exists.

Distinguish:

- **IMPLEMENTED** — code exists and runs.
- **VALIDATED** — objective tests or experiments provide evidence it behaves as specified.
- **SUCCESSFUL** — it satisfies its defined acceptance criteria under the statistical gate.

A component is VALIDATED only when objective tests or experiments provide evidence.
A component is SUCCESSFUL only when it satisfies its defined acceptance criteria.

Every task in `.ai/tasks.json` carries an `acceptance` field. A task cannot be closed
without evidence recorded against that field.

## THE STATISTICAL GATE

No performance change is accepted on the basis of a single run, a plausible argument, or
an LLM's assertion.

A change to the agent's behaviour ships only when:

- n ≥ 30 attempts per arm,
- median time-to-kill improves,
- a 10 000-sample bootstrap CI on the difference excludes zero,
- death rate does not regress beyond its configured bound.

The strategist proposes. The evaluation harness disposes.

## STATE

The persistent project state lives in:

    .ai/project_state.json

The task queue lives in:

    .ai/tasks.json

Agent status lives in:

    .ai/agent_status.json

Failure records live in:

    .ai/failures.json

Experiment records live in:

    experiments/

All four `.ai/` files are read and written through `src/cuphead/orchestration/state_store.py`.
Never hand-edit them while the babysitter loop is running.

## GIT

Make incremental commits. One task per commit where possible.

Never use destructive Git commands such as:

- `git reset --hard` on work that is not yours
- `git push --force` to a shared branch
- `git clean -fdx` without listing first
- history rewrites on a branch someone else has checked out

The babysitter loop refuses to start on a dirty working tree. That is deliberate.

## ORCHESTRATION

```
babysitter → selects task → delegates to worker agent → qa runs acceptance
           → gate → state update → git commit → next iteration
```

Agents live in `.claude/agents/`:

| Agent | Owns |
|---|---|
| `architect` | decomposition, interfaces, build order |
| `vision` | capture, HUD reading, encoder, event detection |
| `world-model` | latent dynamics, prediction heads, planner, cost function |
| `data` | replay logging, buffer curation, dataset integrity |
| `qa` | evaluation harness, statistical gates, regression suite |
| `babysitter` | the outer loop itself |

Entry points:

- `python3 scripts/preflight.py` — verify the foundation before running anything.
- `python3 scripts/babysitter_loop.py --once` — one supervised iteration.
- `python3 scripts/babysitter_loop.py --max-iterations N` — autonomous run.

## TESTING

    python3 -m unittest discover -s tests -v

The orchestration and pure-logic layers are stdlib-only and must stay that way, so the
test suite runs with no dependencies installed. ML modules (`world_model/`, `perception/`)
import numpy/torch lazily and are exercised by separate, optional suites.

## SCOPE

Single-player, offline, locally-run game on a legitimately owned copy. There is no online
component and no other player is affected by anything this system does.
