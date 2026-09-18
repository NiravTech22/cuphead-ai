# cuphead-ai



An autonomous Cuphead speedrunning agent: a learned latent world model with
model-predictive control, a 60 Hz reflex layer, and an LLM strategist; built and
driven by a self-orchestrating multi-agent development loop.

The core insight i've identified from hours worth of gameplay: the overworld map navigation is largely deterministic: same paths, same NPC positions, same interactions. Streamers walking the map is essentially labeled data for "what's the optimal route." The boss fights are where the RNG lives mainly, and this is what the world model + MPC planner should handle online.


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

### Native Windows input and capture

Install `python -m pip install -r requirements.txt`. The Windows-only dependencies
include [vgamepad](https://github.com/yannbouteiller/vgamepad) (requires its
ViGEmBus driver) and [DXcam](https://github.com/ra1nty/DXcam).
Create the virtual controller before launching Cuphead so it can enumerate the pad:

```powershell
python scripts/launch_with_vgamepad.py -- "C:\GOG Games\Cuphead\Cuphead.exe"
```

`open_vgamepad_actuator()` selects vgamepad on Windows and evdev/uinput on Linux.
Both support combat actions, timed button states, menu input, and neutral release.
`open_screen_source()` selects DXcam on Windows and MSS elsewhere. Dependencies
remain lazily imported, so hardware-free tests do not need either native library.

The agent and session recorder use the primary monitor on Windows: run Cuphead
fullscreen there, and use `--controller gamepad` / `--input-source gamepad`.
Keyboard control/recording remains X11-only. Linux retains X11 window capture.
Windows background video recording uses a separate MSS source, avoiding DXcam's
shared camera instance for the agent's output.

For custom capture, import `open_screen_source` from `cuphead.perception`:

```python
source = open_screen_source(backend="dxcam", monitor=1, device_idx=0,
                            region={"left": 0, "top": 0, "width": 960, "height": 540})
try:
    frame = source.read()  # owned RGB bytes, dimensions, checksum, monotonic timestamp
finally:
    source.close()
```

DXcam monitor numbers start at 1 per GPU; crop coordinates are relative to that
output. `backend="mss"` explicitly selects the fallback (desktop coordinates).
DXcam waits for a fresh desktop frame and raises `TimeoutError` after two seconds
(configurable with `timeout=`), including on a completely static desktop. Frame
indices count delivered samples; they cannot establish whether the game dropped
presentations. Native throughput and game response still require a live benchmark.

### Existing workflows

An optional [frozen FLAN-T5 sequence model](docs/FROZEN_T5_SEQUENCING.md)
adds causal multi-step latent prediction with trainable adapters for offline experiments.

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

# Verify Wine sees the virtual Xbox pad: it must be created before Cuphead starts.
python3 scripts/latency_canary.py --real --launch wine /path/to/Cuphead.exe

# For a real game session, keep that virtual pad alive for the whole process.
python3 scripts/launch_with_vgamepad.py -- wine /path/to/Cuphead.exe

# Record keyboard play with default bindings; focus Cuphead to start, switch away to stop:
python3 scripts/record_session.py --real --input-source keyboard --boss forest_follies \
    --loadout weapon=peashooter,charm=smoke_bomb --max-seconds 120 --label-after
```

Real recording uses the existing X11/XWayland capture dependencies in
`requirements-memory.txt`. Keyboard capture needs no gamepad or `/dev/input` access.
`--input-source gamepad` selects the existing evdev reader instead. Both sources use
the same normalized frame/action schema and save raw input snapshots alongside it.
See [gameplay collection and its mandatory gate](docs/GAMEPLAY_DATASET.md) for
controls, segment labels, schema limitations, and the 100,000-frame minimum.
Training remains blocked until coverage and the measured encoder-diversity check pass.

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

## Frozen encoder and episodic consequence memory

The [frozen V-JEPA 2.1 memory agent](docs/FROZEN_LATENT_MEMORY.md) adds CPU INT8
pretrained encoding, 128×128 static-scene inputs, 384/768-dimensional latents,
action-specific kNN consequence prediction, local untried-action exploration,
and verified visual breadcrumb navigation. Run `python3 scripts/run_memory_agent.py --synthetic` for the dependency-free integration fixture or follow the linked
guide for pretrained benchmarks and real Wine control. First-level combat
completion is not yet validated.

An optional [ring-buffered CUDA engine](docs/CUDA_RECORDED_AGENT.md) adds per-slot
manual CUDA graphs, asynchronous pinned-memory transfers, GPU cosine kNN action
voting and a measured <33 ms deadline. It includes a frozen V-JEPA tensor adapter,
an active-loop benchmark and opt-in CUDA tests. Actual GPU latency and overlap
require validation on the target hardware.
