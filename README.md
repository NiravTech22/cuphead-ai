# cuphead-ai



An autonomous Cuphead speedrunning agent: a learned latent world model with
model-predictive control, a 60 Hz reflex layer, and an LLM strategist; built and
driven by a self-orchestrating multi-agent development loop.


<img width="1152" height="648" alt="cuphead1-1152x648" src="https://github.com/user-attachments/assets/b5e6a412-8c9d-46e0-8357-b4fc18344442" />





**Start here:** [`docs/SPEEDRUN_PLAN.md`](docs/SPEEDRUN_PLAN.md) — the full technical
breakdown. [`CLAUDE.md`](CLAUDE.md) — the engineering rules the agents work under.

---

## The idea

Cuphead is a boss-rush game made of deterministic pattern automata running at 60 Hz with
4–8 frame reaction windows. That structure decomposes across three timescales, and the
whole architecture follows from respecting it:

| Timescale | Component | Rate | Job |
|---|---|---|---|
| Reflex | distilled reactive policy | 60 Hz | parry windows, i-frame dashes |
| Tactical | world model + MPC planner | 15 Hz | positioning, DPS uptime, ~1 s lookahead |
| Strategic | LLM strategist | offline | loadout, route, phase priors, post-mortems |

**The LLM is never in the control loop.** The world model never tries to hit a 5-frame
parry. The reflex layer never plans.

The objective is not survival. Cuphead bosses change phase on HP thresholds rather than
timers, so out-damaging a phase skips its attack patterns entirely — time saved is
superlinear in damage dealt. The planner is a risk-constrained damage maximizer:

```
J = Σ γ^t [ dps_uptime(z_t, a_t) − λ(hp, phase) · P_hit(z_t, a_t) ]
```

## Layout

```
CLAUDE.md                  engineering rules the agents work under
docs/SPEEDRUN_PLAN.md      the full technical breakdown (§1–§12)
docs/DECISIONS.md          architecture decision record

.claude/agents/            six agent definitions
.ai/                       persistent project state, task queue, failures
experiments/               experiment records — a run with no record did not happen

scripts/preflight.py       verify the foundation before anything autonomous runs
scripts/babysitter_loop.py the outer orchestration loop

src/cuphead/
  perception/   capture + integrity checking (done), HUD templates, pink parry
                prior, latent encoder (not yet)
  state/        symbolic + latent fused agent state
  events/       the discrete event vocabulary
  world_model/  latent dynamics, reward/value/hit heads
  planner/      CEM/MPPI MPC and the risk-constrained cost function
  policy/       60 Hz reflex layer and override arbitration
  control/      action space, virtual-gamepad actuation, human-input recording
  strategist/   LLM: loadout, phase priors, post-mortems, routing
  memory/       replay format (done), episodic store and distilled lessons
  evaluation/   metrics, the statistical gate, round-trip latency measurement
  orchestration/ state store, task queue, the statistical gate

scripts/latency_canary.py   round-trip actuate->observe latency, gated
scripts/record_session.py   record one attempt: synced frames + human input

tests/                     stdlib unittest — no dependencies required
```

## Running it

```bash
python3 scripts/preflight.py                       # verify the foundation
python3 -m unittest discover -s tests -v           # 149 tests, stdlib only

python3 scripts/babysitter_loop.py --once --dry-run   # show the plan, invoke nothing
python3 scripts/babysitter_loop.py --once             # one supervised iteration
python3 scripts/babysitter_loop.py --max-iterations 20 --max-budget-usd 25
```

`preflight.py` checks the state documents, agent frontmatter, task graph, git baseline,
the `claude` CLI and its flags, and the test suite. The loop runs it first and refuses to
start on a broken foundation or a dirty working tree.

The orchestration, control, event, planner and evaluation layers are **stdlib-only**, so
everything above runs on a bare machine. `requirements.txt` covers the ML layers, which
import numpy/torch lazily.

## Recording your first session

The phase-0 harness — capture with duplicate/drop detection, virtual-gamepad actuation,
the frame-indexed replay format, and a human-demonstration recorder — is code-complete
and self-tested. It is not yet *validated*: that requires running it against the real
game, which this sandbox cannot do. On the machine actually running Cuphead:

```bash
# Sanity-check the pipeline with no game and no hardware:
python3 scripts/latency_canary.py --synthetic
python3 scripts/record_session.py --synthetic --frames 300

# The real thing — game window visible, virtual pad wired up as its active controller:
python3 scripts/latency_canary.py --real
python3 scripts/record_session.py --real --boss goopy_le_grande \
    --loadout weapon=peashooter,charm=smoke_bomb --max-seconds 120
```

`--real` needs `pip install -r requirements.txt` for `mss` (screen capture) and `evdev`
(controller input and the virtual pad); both import lazily, so nothing else in the repo
needs them installed. Record several dozen short attempts — including deaths — before
touching any training code; see `docs/SPEEDRUN_PLAN.md` §2.5 and §4.4 for why the first
data is human, not agent, and why a corrupted or misaligned recording is refused rather
than silently written.

## The rules that make it work

Three, and they are all in `CLAUDE.md`:

1. **IMPLEMENTED → VALIDATED → SUCCESSFUL.** Code existing is not evidence of anything.
   Only the `qa` agent moves a verdict, and only on measurements.
2. **The statistical gate.** n ≥ 30 per arm, median TTK improves, a 10 000-sample
   bootstrap CI on the difference excludes zero, death rate does not regress, the
   stalling detector does not fire, latency stays in budget.
3. **The strategist proposes; the evaluation harness disposes.** Every LLM suggestion is
   a typed hypothesis carrying a numeric prediction it gets scored on.

## Scope

Single-player, offline, locally-run game on a legitimately owned copy. No online
component; no other player is affected.
