# Cuphead Speedrun via Learned World Model + LLM Strategist

**A full technical breakdown: what to build, in what order, and how to know it works.**

---

## 0. The thesis in one paragraph

Cuphead is not a game you beat with a large language model, and it is not a game you
beat with pixels-to-buttons reinforcement learning either. It is a **boss-rush game
made of deterministic pattern automata executed at 60 Hz with 4–8 frame reaction
windows.** That structure decomposes cleanly across three timescales, and the entire
architecture follows from respecting that decomposition:

| Timescale | Component | Rate | Job |
|---|---|---|---|
| **Reflex** | Distilled reactive policy | 60 Hz (16.7 ms) | Parry windows, i-frame dashes, contact avoidance |
| **Tactical** | Latent world model + MPC planner | 15 Hz (66 ms) | Positioning, DPS uptime, 1-second lookahead through a projectile wave |
| **Strategic** | LLM strategist | 0.02–1 Hz (offline) | Loadout, boss route, phase policy priors, failure post-mortems, next experiment |

The LLM never touches the control loop. The world model never tries to hit a 5-frame
parry. The reflex layer never plans. Every failure mode in this project traces back to
someone violating one of those three sentences.

---

## 1. Why this game, and what "winning" actually means

### 1.1 The speedrun objective is not survival

The naive framing — "learn to not die" — produces an agent that runs to the corner and
lives forever. The actual objective for **All Bosses / Regular** category is:

```
minimize  T_total = Σ_bosses  TTK(boss)          # time-to-kill, in-game timer
subject to  P(death) low enough that reset cost doesn't dominate
```

Because Cuphead bosses transition phases on **HP thresholds, not timers**, higher DPS
does not shave time linearly — it *skips attack patterns entirely*. Killing Goopy Le
Grande fast enough means you never see the tombstone phase. This is the single most
important fact in the whole project:

> **Damage output is superlinear in time saved. Survival is a constraint, not the goal.**

So the planner's cost function is a **risk-constrained damage maximizer**, not a
survival maximizer:

```
J = Σ_t γ^t [ dps_uptime(z_t, a_t)  −  λ(h_t, φ_t) · P_hit(z_t, a_t) ]
```

where `λ` is annealed by current HP `h_t` and phase `φ_t`. At 3 HP with a fresh boss,
λ is small — trade hits for aggression. At 1 HP in the final phase, λ spikes — a death
costs the entire segment.

### 1.2 Metrics that actually gate progress

Everything below is measured per boss, over n ≥ 30 attempts, before any change is
called an improvement:

| Metric | Why it matters |
|---|---|
| **Median TTK** | The headline number. Median, not mean — deaths are outliers. |
| **p10 TTK** | "PB potential." A run is only as good as its achievable tail. |
| **Death rate** | Constraint check. |
| **DPS uptime %** | Fraction of frames with the shoot input held *and* aimed at the hitbox. The single best leading indicator of TTK. |
| **Hits taken / attempt** | Perception + reflex quality proxy. |
| **Parry conversion** | Parries executed / parryable objects presented. Drives super-meter economy. |
| **Super efficiency** | Damage per card spent, and whether cards were dumped in the right phase. |
| **Reset rate** | Full-run health. |

**Statistical gate:** a change ships only if median TTK improves and a 10 000-sample
bootstrap CI on the difference excludes zero. This is the operational meaning of the
`IMPLEMENTED → VALIDATED → SUCCESSFUL` ladder in `CLAUDE.md`.

---

## 2. The harness comes first (and it is where projects die)

There is no Cuphead API, no gym environment, no save states. **Phase 0 is not glamorous
and it is non-negotiable.** If the harness is 30 ms of jitter, no amount of model
quality recovers it.

### 2.1 Latency budget

60 fps means 16.67 ms per frame. With action repeat 4 (decide at 15 Hz) the tactical
loop has 66 ms, but the *reflex* loop must close inside two frames:

```
capture       →  encode      →  reflex head  →  actuate
  ≤ 4 ms          ≤ 5 ms         ≤ 2 ms         ≤ 2 ms      = 13 ms  ✅ fits one frame
                                  ↓
                        (tactical, every 4th frame)
                     CEM 512×12 latent rollouts ≤ 10 ms
```

Measure each stage with a hardware-level round trip: emit a controller input, detect the
first frame in which the sprite moves, count frames. Do not trust wall-clock timers
across the capture boundary.

### 2.2 Capture

- Linux + Proton/Wine, or native Windows. Both are viable; Linux is easier to automate.
- **X11 SHM / PipeWire screencast** into a ring buffer, or an OBS virtual-camera tap.
  Avoid anything that round-trips through a compositor copy.
- Downscale on the GPU to 128×128 (or 160×120) before it ever reaches Python.
- Stamp every frame with a monotonic counter. **Detect duplicate frames** — a repeated
  frame silently corrupts world-model training data with a fake "nothing happened"
  transition.

### 2.3 Input

- Virtual **XInput** gamepad — Cuphead's controller path is lower-latency and less
  ambiguous than keyboard. `uinput`/`evdev` on Linux, `vgamepad`/ViGEm on Windows.
- Input is a *held state*, not an event. The action space is the set of buttons held
  during a frame window.
- Log every actuation with the frame index it was intended for. Training on
  misaligned `(s, a)` pairs is the most common silent killer of a world model.

### 2.4 Determinism and the replay log

You cannot save-state Cuphead, so you buy reproducibility a different way:

- Fixed difficulty, fixed loadout, fixed boss, fixed starting position.
- Every attempt writes a **replay record**: frames (compressed), actions, frame indices,
  detected events, HUD ground truth, outcome.
- Evaluation is always **distributional over N attempts**, never a single run.
- Keep a frozen `eval/` set of replays for offline regression testing of perception and
  world-model prediction, so most iterations never touch the game at all.

---

## 3. Perception: hybrid symbolic + latent

Pure end-to-end latents give the world model something to learn from but give the LLM
nothing to reason about, and give you nothing to debug. Pure symbolic detectors are
brittle. Use both, and let each supervise the other.

### 3.1 Free ground truth from the HUD

The Cuphead HUD is **deterministic pixels**. Template matching against fixed screen
regions gives you, for free, at 100% accuracy:

- Player HP (heart count)
- Super meter (card count, including partial fill)
- Current weapon
- The "KNOCKOUT" / "A KNOCKOUT!" end-card, and death frames

This is your label source. Use it three ways:
1. Direct state features (exact, no learning needed).
2. **Auxiliary prediction heads** on the encoder — predicting HP and cards from the
   latent grounds the representation without a single hand label.
3. Reward/event signal — HP decrement *is* the hit event.

### 3.2 The pink prior

Every parryable object in Cuphead is **pink**. That is a designed, game-wide invariant,
and it is the highest-leverage engineering shortcut available:

- HSV threshold + connected components → parryable candidate mask, in ~1 ms, at 60 Hz.
- Feeds the reflex parry head directly.
- Gives the world model an explicit "parry opportunity" channel rather than making it
  discover pinkness from scratch.

Do not skip this in favour of purity. It converts the hardest timing problem in the game
into a tractable one.

### 3.3 Learned encoder

- Small CNN (or tiny ViT) → 256-d latent, trained with:
  - auxiliary HUD heads (HP, cards),
  - a **contrastive/temporal objective** (nearby frames close, distant frames far),
  - and later, jointly with the world model.
- Frame stack of 4 at 60 Hz gives velocity information without recurrence.
- Boss HP is **not displayed** — infer a phase-progress proxy from observed phase
  transitions, and track elapsed damage from your own shot log.

### 3.4 Event detector — the symbolic layer

A discrete event stream is what makes the LLM useful and what makes failures readable:

```
PHASE_ENTER(boss, phase_id)   HIT_TAKEN(hp_before, hp_after)   PARRY(success|miss)
SUPER_FIRED(card_cost)        DEATH(phase_id, cause_hypothesis) KNOCKOUT(time)
PROJECTILE_WAVE(kind, lane)   DPS_WINDOW(open|close)
```

Events are extracted from HUD deltas + motion segmentation + a learned phase classifier.
Every replay reduces to an event trace of a few hundred tokens — which is exactly the
right size for an LLM to read a hundred of them.

---

## 4. World model: learn to predict getting hit, not to predict pixels

### 4.1 Architecture choice

Two credible families:

- **RSSM / Dreamer-style** — stochastic latent, decoder reconstruction, imagination
  training. Great for sample efficiency, but reconstruction spends capacity on
  background parallax you do not care about.
- **TD-MPC2-style** — deterministic latent dynamics with reward + value heads, *no
  decoder*, trained purely for planning utility.

**Recommendation: TD-MPC2-flavoured.** You want planning accuracy, not imagination
fidelity. Dropping the decoder frees capacity for the 6-pixel bullet that actually kills
you, and removes the classic failure where the model reconstructs a beautiful background
and ignores projectiles.

```
z_t     = E(o_{t-3..t})                    # encoder, 256-d
z_{t+1} = D(z_t, a_t)                      # latent dynamics (GRU or residual MLP)
r̂_t    = R(z_t, a_t)                      # dense reward: damage dealt − hit penalty
v̂_t    = V(z_t)                           # value bootstrap beyond horizon
ĥ_t    = H(z_t, a_t)                       # P(hit within k frames)  ← the key head
π̂      = P(z_t)                            # policy prior, amortizes CEM
```

### 4.2 Horizon and rate

Decision rate 15 Hz (action repeat 4), horizon 12–20 latent steps = **0.8–1.3 seconds**.
That is deliberately matched to the duration of a Cuphead projectile wave. Longer
horizons buy nothing and compound model error; shorter ones cannot see a wave coming.

### 4.3 The acceptance criterion that matters

Do **not** gate the world model on multi-step latent MSE. Gate it behaviourally:

> **Hit-prediction AUC:** given `(z_t, a_{t..t+k})`, does the model predict `HIT_TAKEN`
> within k steps? Target **AUC ≥ 0.85 at k = 8 (≈0.5 s)** on held-out replays, per boss.

A model with mediocre reconstruction and strong hit-prediction AUC will plan well. The
reverse is not true. Secondary gates: phase-transition prediction accuracy, and
open-loop action-conditioned divergence measured in *hit-probability* space rather than
latent L2.

### 4.4 Data flywheel

- Seed with 5–15 hours of human play (yours), covering deaths — *especially* deaths.
- Then agent play dominates: it is on-distribution for the policy you are improving,
  which is worth roughly 5× the same volume of human data.
- Maintain a **prioritized replay buffer** weighted toward frames near `HIT_TAKEN`,
  `PARRY`, and `PHASE_ENTER`. Uneventful mid-arena frames are cheap and uninformative.
- Rebalance per boss. A model trained 90% on one boss will confidently mispredict
  another and you will not notice until TTK regresses.

---

## 5. Planner: MPC in latent space

### 5.1 The loop

```
every 4 frames:
    z ← encode(recent frames)
    seed N=512 action sequences from π̂(·|z)      # policy prior, not uniform
    for k in 1..3:                                # CEM iterations
        roll out H=12 steps through D, score with J
        keep elite top 10%, refit sampling distribution
    a* ← first action of best sequence
    execute a* for 4 frames (reflex layer may override)
```

Budget: 512 × 12 latent steps in fp16 on a single modern GPU is ~5–8 ms. Keep the latent
small and the dynamics shallow; this is a hard real-time constraint, not a preference.

### 5.2 Action space

Factorized and pruned:

```
move   ∈ {none, L, R}                      (3)
vert   ∈ {none, jump, duck}                (3)
dash   ∈ {0, 1}                            (2)
shoot  ∈ {0, 1}                            (2)
lock   ∈ {0, 1}                            (2)   aim-lock
aim    ∈ {8 directions}  (only when lock)  (8)
```

Prune combinations that are illegal or never useful (dash + duck, aim without lock).
A pruned space of ~60 effective actions samples far better than a raw 576.

### 5.3 Amortization

Train the policy prior `π̂` by behaviour-cloning the planner's own chosen actions. Over
time CEM iterations can drop from 3 to 2 to 1 with no loss, buying latency headroom.
This is also the distillation path to the reflex layer.

### 5.4 Reward hacking to watch for

- **Stalling:** an agent that survives by never engaging. Fix: explicit per-frame time
  penalty and a DPS-uptime bonus in `J`.
- **Corner camping:** add a positional entropy term or per-phase preferred-lane priors.
- **Super hoarding:** cards expire with the run. Penalize unspent cards at knockout.

---

## 6. The reflex layer (why MPC alone is not enough)

Parry windows in Cuphead are roughly **5–6 frames (~90 ms)**. A 66 ms tactical loop plus
model error plus actuation latency does not reliably hit that. So:

- A dedicated **small convnet head at full 60 Hz** consuming the pink mask + player
  position + a short frame stack, outputting `{no-op, parry-jump}`.
- Trained by behaviour cloning on human parries plus hard-negative mining from missed
  ones — then improved by the same replay flywheel.
- Same treatment for **i-frame dash timing** against contact damage.
- **Override arbitration:** the reflex head can preempt the planner's action, but only
  within a whitelist of (parry, dash) and only when its confidence exceeds a per-boss
  threshold. Log every override with its outcome — override precision is itself a gated
  metric.

This split is what makes the whole thing feasible. The planner decides *where to be*; the
reflex decides *what to do in the next 90 ms*.

---

## 7. LLM strategist: five concrete jobs, none of them in the control loop

The LLM is the layer that makes this project different from a standard RL agent, but only
if its outputs are **typed, validated, and A/B tested**. Free-form advice is worthless
here; structured, falsifiable output is the whole value.

### 7.1 Loadout selection

Per boss, choose weapon + charm from the failure memory. Output is a schema, not prose:

```json
{
  "boss": "goopy_le_grande",
  "weapon": "lobber",
  "charm": "smoke_bomb",
  "rationale": "phase-2 tombstone requires ground-level arcing damage; smoke-bomb dash grants i-frames through the slam",
  "confidence": 0.7,
  "expected_ttk_delta_s": -3.5
}
```

Every such claim goes into the eval queue. `expected_ttk_delta_s` is a **prediction the
system scores the LLM on**, which over time produces a calibration record you can trust.

### 7.2 Phase policy priors

Structured shaping of the planner's cost, per boss phase:

```json
{
  "phase": "goopy_p2",
  "aggression": 0.35,
  "preferred_lane": "left",
  "lambda_risk": 2.4,
  "forbid": ["dash_under_boss"],
  "parry_priority": "high",
  "super_policy": "hold_for_p3"
}
```

Validated by a schema, clamped to safe ranges, then A/B'd against the current prior.
Never applied blind.

### 7.3 Failure post-mortem

The LLM reads the **event traces** (not the pixels) of the last N failed attempts plus
retrieved similar past failures, and produces hypotheses:

> "17 of 23 deaths occurred within 400 ms of `PHASE_ENTER(goopy_p3)`. The planner is
> mid-commitment to a dash when the arena geometry changes. Hypothesis: phase-transition
> frames are under-represented in the world model's training distribution."

Each hypothesis becomes a **task in `.ai/tasks.json`** with an acceptance criterion.
This is the loop that makes the system self-improving rather than merely automated.

### 7.4 Route and category strategy

Boss ordering, where to spend resets, which segments carry the most theoretical time
loss versus gold splits. This is genuine long-horizon reasoning over aggregated stats —
the thing LLMs are actually good at.

### 7.5 Experiment authoring

Writes eval configs, curriculum orders, and ablation specs. The QA agent runs them. The
LLM never reports its own results.

### 7.6 The hard rule

> **The strategist proposes. The evaluation harness disposes.**
> No LLM output is ever accepted as evidence of improvement. Every suggestion is a
> hypothesis with a numeric prediction, tested against the statistical gate in §1.2.

---

## 8. Memory

- **Episodic store:** every attempt as `(boss, phase, loadout, event_trace, outcome, ttk)`.
- **Structured index** for exact filters (boss, phase, weapon) + **embedding index** over
  event traces for "have we seen this failure before?"
- **Distilled lessons:** periodically compact clusters of similar failures into a short
  written lesson with an evidence count. These, not raw traces, are what get injected
  into the strategist's context — a hundred raw traces will not fit; ten distilled
  lessons with counts will.

---

## 9. Build order (the actual roadmap)

Each phase has a hard exit criterion. Do not start phase N+1 before phase N is
**VALIDATED**.

| Phase | Build | Exit criterion |
|---|---|---|
| **0. Harness** | Capture, virtual gamepad, frame sync, replay logger | Round-trip latency measured < 30 ms, p99 < 50 ms; zero duplicate/dropped frames over a 10-min capture |
| **1. Perception** | HUD templates, pink mask, encoder, event detector | HP/card read 100% on 5 k held-out frames; event detector F1 ≥ 0.95 vs. hand-labelled traces |
| **2. Baseline + metrics** | Scripted/reflex-only agent, full eval harness | Reproducible TTK distribution for one boss over 30 attempts; CI width < 15% of median |
| **3. Reflex** | Parry head, dash head, override arbitration | Parry conversion ≥ human median; override precision ≥ 0.9 |
| **4. World model** | Encoder+dynamics+heads, prioritized buffer, training loop | **Hit-prediction AUC ≥ 0.85 at k=8** on held-out, per target boss |
| **5. Planner** | CEM/MPPI MPC, policy prior, cost function | Median TTK beats phase-2 baseline, bootstrap CI excludes 0, n ≥ 30 |
| **6. Strategist** | Loadout, phase priors, post-mortems, task generation | ≥ 3 LLM-proposed changes validated by the gate; calibration record started |
| **7. Route** | Multi-boss generalization, segment routing, full-run assembly | Sub-X full-run time; per-segment golds tracked |

**First boss target: The Root Pack or Goopy Le Grande.** Ground-based, single-lane,
readable phases, generous parry objects. Do not start on Grim Matchstick or Dr. Kahl's
Robot — you will be debugging the harness while also debugging a nightmare.

---

## 10. Risk register

| Risk | Signature | Mitigation |
|---|---|---|
| Input/capture latency drift | TTK regresses with no code change | Latency canary in every eval run; abort if p99 exceeds budget |
| Frame/action misalignment | World model plateaus at chance-level hit AUC | Frame-indexed actuation log; assert alignment in the data loader |
| Single-boss overfit | Great on boss A, worse than baseline on B | Per-boss eval matrix in every gate; rebalanced buffer |
| Reward hacking | Death rate falls, TTK rises | Time penalty + DPS-uptime term; stalling detector in eval |
| LLM overconfidence | Proposals stop validating | Calibration record; require numeric prediction on every proposal |
| Non-stationarity from game updates | Everything degrades at once | Pin the game build; version replays against it |
| Compute | Training queues behind evaluation | Offline-first: most iterations run on frozen replays, not the live game |

**Compute reality check:** a single 24 GB consumer GPU is sufficient. The encoder is
small, the dynamics model is small, and the expensive part — CEM rollouts — is a batched
matmul. This is a latency-bound project, not a FLOPs-bound one.

**Scope note:** single-player, offline, locally-run game on your own copy. No online
component, no other players affected.

---

## 11. How the agent layer maps onto this

The six agents in `.claude/agents/` are not decoration; each owns a slice of §1–§9:

```
architect     → §2, §9      system decomposition, interfaces, build order
vision        → §3          capture, HUD, pink prior, encoder, event detector
world-model   → §4, §5      dynamics, heads, planner, cost function
data          → §2.4, §4.4  replay logging, buffer curation, dataset integrity
qa            → §1.2, §9    eval harness, statistical gates, regression suite
babysitter    → §11         orchestration: select → delegate → review → gate → commit
```

The babysitter runs the outer loop:

```
read .ai/project_state.json + tasks.json
        ↓
select highest-priority unblocked task
        ↓
delegate to the owning worker agent  (claude -p --agent <name>)
        ↓
worker implements, marks IMPLEMENTED
        ↓
qa agent runs the acceptance criterion
        ↓
gate: VALIDATED / SUCCESSFUL / FAILED → .ai/failures.json
        ↓
state update + git commit
        ↓
next iteration
```

`scripts/babysitter_loop.py` implements exactly this, with per-iteration budget caps,
a clean-tree precondition, and a hard stop on repeated failure of the same task — so it
can be left running for hours without producing a pile of unreviewable commits.

Run `python3 scripts/preflight.py` before the first iteration. It checks the state files,
the agent frontmatter, the git baseline, and the CLI, and refuses to let the loop start
on a broken foundation.

---

## 12. What to do on day one

1. `python3 scripts/preflight.py` — confirm the foundation is sound.
2. Commit the baseline (the loop refuses to run on a dirty tree with no history).
3. Run **one** supervised iteration: `python3 scripts/babysitter_loop.py --once --dry-run`,
   read the plan it produces, then drop `--dry-run`.
4. The first real task in the queue is `harness-capture-latency` — because §2 says so,
   and because every other number in this document is meaningless until it passes.
