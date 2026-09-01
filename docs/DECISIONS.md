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
