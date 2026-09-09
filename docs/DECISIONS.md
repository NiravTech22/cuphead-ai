# Architecture Decision Record

Every non-obvious decision, with the observation that would reverse it. The last column
matters most: a decision with no falsifier is a preference, not a decision.

---

## ADR-001 — Three timescales, strictly separated

**Context.** Parry windows are 5–6 frames (~90 ms). A tactical loop at 15 Hz plus model
error plus actuation latency cannot reliably hit that. Meanwhile an LLM cannot run at
60 Hz at all.

**Decision.** Reflex (60 Hz, reactive), tactical (15 Hz, MPC), strategic (offline, LLM).
Every control decision belongs to exactly one. The LLM is never in the control loop.

**Reverses if.** Reflex override precision stays below 0.9 after the flywheel matures,
suggesting the split is in the wrong place rather than the head being undertrained.

---

## ADR-002 — TD-MPC2-style latent dynamics, no decoder

**Context.** Dreamer-style RSSMs reconstruct observations. Cuphead backgrounds are busy
and parallaxed; the thing that kills you is a six-pixel projectile. Reconstruction loss
spends capacity in exactly the wrong place.

**Decision.** Deterministic latent dynamics with reward, value, hit-probability and
policy-prior heads. No decoder.

**Reverses if.** Sample efficiency proves unworkable without reconstruction — i.e. hit
AUC plateaus below 0.85 with the full data flywheel running, and an RSSM ablation on the
same data clears it.

---

## ADR-003 — Gate the world model on hit-prediction AUC, not prediction error

**Context.** Latent MSE and reconstruction quality both correlate poorly with planning
quality. What the planner actually consumes is `P(hit)`.

**Decision.** Acceptance is AUC ≥ 0.85 at k = 8 steps on held-out replays, per boss.

**Reverses if.** A model clears the AUC bar per boss and still fails to improve median
TTK through the planner, which would mean the bottleneck moved elsewhere (reward head,
horizon, or action space).

---

## ADR-004 — The pink prior is engineered, not learned

**Context.** Every parryable object in Cuphead is pink. That is a designed, game-wide
invariant.

**Decision.** HSV threshold plus connected components, ~1 ms, feeding the reflex head and
the world model as an explicit channel. Not discovered by a network.

**Reverses if.** A boss presents parryable objects outside the threshold, or non-parryable
pink objects at a rate that pushes precision below 0.95. Both are measurable on the
fixture set, per boss.

---

## ADR-005 — The HUD is read by template matching, and is the label source

**Context.** The HUD is deterministic pixels at fixed positions. A network reading it
would be strictly worse and would need labels.

**Decision.** Template matching, exact, ~1 ms. HP and card reads then serve as auxiliary
heads that ground the encoder with zero hand labelling.

**Reverses if.** A resolution or UI-scale change breaks position assumptions — in which
case the templates are re-derived, not replaced by a model.

---

## ADR-006 — Deaths are excluded from the TTK sample, not imputed

**Context.** A death has no time-to-kill. Imputing one (as infinity, or as a timeout)
corrupts the median the entire statistical gate rests on.

**Decision.** `evaluation.metrics.aggregate` counts deaths in `deaths`/`attempts` and
excludes them from `ttk_s`. The gate checks median TTK *and* death rate, so the
information is not lost — it is just not smuggled into the wrong statistic.

**Reverses if.** Death rates rise high enough that the surviving TTK sample is a biased
subset (survivorship). At that point the primary metric becomes expected time including
reset cost.

---

## ADR-007 — Action space pruned by game facts, not by heuristics

**Context.** The raw product space is 576 combinations. CEM sampling that space wastes
most of its budget on states the game cannot express.

**Decision.** Six pruning rules, each a fact about Cuphead: dash drops duck; dash cancels
the shot; aim-lock roots the player; ducking overrides the aim stance; idle aim-lock does
nothing; aim is meaningless without lock. Result: **56 legal actions**.

**Reverses if.** A charm or weapon changes one of those facts (a dash that preserves the
shot, say), which would make the corresponding rule loadout-conditional rather than
global.

---

## ADR-008 — λ spans ~20× between full health and empty

**Context.** Decisions happen at 15 Hz. A per-step hit probability of 0.15 means being hit
inside a second. At full health that is a fair price for a second of damage; at one heart
it ends the attempt.

**Decision.** `LAMBDA_HEALTHY = 0.6`, `LAMBDA_CRITICAL = 12.0`, interpolated by health and
scaled by phase progress.

**Reverses if.** The agent becomes visibly over-cautious at one heart — surviving phases
it could have out-damaged. The stalling detector plus per-phase TTK breakdown will show
this before a human notices it.

---

## ADR-009 — The orchestration layer is stdlib-only

**Context.** The layer whose job is to tell you whether the project is safe to start
cannot itself depend on the project being set up.

**Decision.** `orchestration/`, `control/`, `events/`, `planner/objective.py` and
`evaluation/metrics.py` import nothing outside the standard library. The full test suite
runs on a bare machine.

**Reverses if.** Never, realistically. If a gate genuinely needs scipy, it computes the
statistic in the ML layer and hands the *number* to the gate.

---

## ADR-010 — The implementing agent never grades its own work

**Context.** An agent that both implements and validates will report success. This is not
a moral failing; it is what the objective asks for.

**Decision.** Implementation and acceptance are two separate `claude` invocations with
two different agents. Only `qa` moves a verdict, and `.ai/failures.json` records every
disagreement between a worker's claim and QA's measurement.

**Reverses if.** Nothing. This one is structural.

---

## ADR-011 — Imitation learning bootstraps every boss, before RL or MPC touches the live game

**Context.** A policy that starts from random exploration against the real game spends
its first hours failing to survive long enough to see anything worth learning from, and
every one of those attempts costs real, un-parallelizable wall-clock time against a game
that cannot be save-stated or sped up.

**Decision.** The first several dozen attempts on any new boss are human demonstrations
(`scripts/record_session.py --real`, §2.5), and the first policy trained from them is
behaviour cloning — imitate the recorded `state → action` mapping directly — before any
RL fine-tuning or MPC planning runs against the live game. RL/MPC then refines a policy
that is already in the right neighborhood, instead of finding that neighborhood by random
exploration.

**Reverses if.** A boss turns out to have attack patterns simple enough that a
from-scratch policy converges in fewer live attempts than collecting and cloning a human
dataset would cost — plausible for an early, low-complexity boss, worth checking with a
timed ablation rather than assumed.

---

## ADR-012 — Duplicate-frame detection is content-based, not index-based

**Context.** A capture backend can advance its own frame counter correctly while still
handing back a stale buffer underneath — the index says the pipeline is healthy while the
content says the world stood still for a frame. Checking only the index would miss
exactly the failure mode most dangerous to a world model: a fake "nothing happened"
transition that trains it to (mis)predict a frozen world.

**Decision.** `perception.capture.IntegrityTracker` runs two independent checks: an index
gap (a drop) and a content-checksum repeat (a duplicate), and treats a duplicate as the
more dangerous of the two because it is silent where a drop is loud.

**Reverses if.** A real capture backend proves the checksum check produces false
positives at a rate that matters (e.g. a genuinely static boss-intro frame held for
several real frames) -- at that point the check needs a per-context exemption, not
removal.

---

## ADR-013 — Human demonstrations are normalized into the pruned action space at record time

**Context.** The planner only ever searches the 56 legal actions from ADR-007. A
demonstration recorded as free-form raw controller state (which can express illegal
combinations no plan ever produces, like holding aim-lock while airborne with the stick
centered) is not directly usable as an imitation-learning target for a policy that must
itself only ever output legal actions.

**Decision.** `control.human_input.Action.from_raw` applies the exact same precedence
rules `Action.is_legal` enforces (dash cancels the shot, aim-lock roots the player, an
idle lock is dropped, …) to raw device state before it is ever logged, so every recorded
action is one the planner itself could have chosen. `tests/test_human_input.py` checks
this holds over every combination of raw input flags, not just the common ones.

**Reverses if.** The pruning rules in ADR-007 change (a charm alters what's physically
possible) -- `from_raw` and `is_legal` must change together, or a demonstration recorded
under the old rules silently stops being a legal imitation target under the new ones.

---

## ADR-014 — The virtual pad impersonates an Xbox 360 wired controller, exactly

**Context.** `latency_canary.py --real` failed with "no visible response within 600
frames of actuation" against a uinput device that was created successfully and visible in
`/dev/input/`. Cuphead under Wine does not read evdev; `winebus.sys` enumerates Linux
input devices through udev/SDL, classifies them, and only devices that classify as an
Xbox-compatible gamepad are exposed through `xinput1_3`/`xinput1_4`. The previous device
(4 buttons, `ABS_X`/`ABS_Y`, name `cuphead-ai-virtual-pad`, evdev's default
vendor/product/version of 1) classified as a generic joystick and was never routed to the
game. Nothing in the failure distinguishes this from a wrong capture region or a dead
actuator, which is what made it expensive to diagnose.

**Options considered.** (a) Fall back to keyboard input -- rejected: §2.3 chose the
controller path for latency and unambiguous held state, and keyboard reintroduces the
event-vs-state ambiguity the action space is built on. (b) Depend on an external
emulation layer (`xboxdrv`, ViGEm-alike, an SDL virtual joystick) -- rejected: another
runtime dependency and another process in the input path, for a signature we can declare
ourselves in ~40 lines. (c) Declare the full Xbox 360 signature on our own uinput device.

**Decision.** (c). `control.actuator.XBOX360_SIGNATURE` declares name, USB identity
(`045e:028e`, `BUS_USB`, version `0x0110` -- the tuple SDL's built-in mapping database
keys on) and the complete `xpad` capability set: both thumbsticks, both analog triggers
as unsigned bytes, the D-pad hat, and the whole gamepad button block, including buttons
the planner never presses. The signature is stdlib data lowered to evdev types only at
open time, so `tests/test_actuator.py` can assert it and the write path with a fake
`ecodes` and a fake device -- no `/dev/uinput`, no root, no evdev. Two behaviours ride
along: a settle delay after device creation (winebus rescans on hotplug), and a neutral
state published at open and at close so the game never sees an undefined axis or a stuck
hold. The `Action`-based `Actuator` interface is unchanged; no caller moves.

**Reverses if.** A future Wine/SDL release classifies gamepads from the HID descriptor
alone and stops keying on the VID/PID -- at which point impersonating Microsoft's IDs
buys nothing and an honest name/ID is preferable. Also reverses in the narrow sense if
`--real` shows RB bound to something in the running Cuphead config: aim lock is currently
held on both `BTN_TR` and `ABS_RZ` (the default Xbox binding), on the assumption that
neither surface is bound to anything else; observing an unintended action on lock means
dropping the `BTN_TR` write and keeping the trigger.

**Status.** IMPLEMENTED, not VALIDATED. The signature tests are evidence that the
device *declaration* is complete; only `scripts/latency_canary.py --real` on the machine
running Cuphead is evidence that Wine actually routes it.
