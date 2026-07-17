# Scene Sense — Decision Records

Dated, ADR-style. One short entry per non-obvious choice.

---

## ADR-008 — 2026-07-15 — Timing badge is conditional and client-measured

**Context.** A local-7B pipeline is impressive when cached (~15–25s) but a
cold run can take minutes; a timer that mostly reports slow numbers is
anti-marketing.

**Decision.** Wall time is measured in the browser (performance.now around
the SSE stream — the number the user actually experienced). Under
`TIMING_BADGE_MAX_S = 8` seconds the result shows "Scene found in X.Xs";
"⚡ cached" shows whenever the backend reports `cache_hit` (download served
from cache); at/over threshold no time is shown, ever. One constant governs
the threshold.

**Consequences.** Fast runs self-advertise, slow runs stay quiet, and the ⚡
gives the demo a cache story even when the LLM keeps totals over 8s. The
badge can't drift from perceived time since it *is* perceived time.

## ADR-007 — 2026-07-15 — Recent Quests re-filtered server-side at render, fail-closed

**Context.** Recent quests live in the browser (localStorage) but the
blocklist lives server-side and can change after a quest was stored. The
filter must own every rendered surface.

**Decision.** The frontend never renders stored quest text without first
POSTing it to `/api/filter/check` (batch, blocked-flags only; categories are
logged server-side, never the text). If the check fails (server down,
network error), the row renders nothing — fail-closed beats showing
unchecked text. Blocked entries are simply skipped, not deleted, so a later
blocklist relaxation restores them.

**Consequences.** One tiny POST per hero render. No blocklist duplication in
JS; localStorage stays the single local store (privacy note in README).

## ADR-006 — 2026-07-15 — Modes are prompt/render variations, injected post-locate

**Context.** Analyze/Metadata modes must not create new subsystems or
perturb retrieval, and default Find behavior must stay byte-identical.

**Decision.** `mode` is per-query metadata threaded through
`/api/chat?mode=` → `assistant.run(mode=)`. The identify→fetch→locate→clip
loop is untouched; the single mode instruction is appended as a user message
only after `result.clip` exists, so no mode text is in context during
retrieval (isolation). Metadata mode is mostly a frontend render decision
(suppress prose, expand the strip); its extra payload fields
(`kb_confidence`, `match_score`, `mode`) are attached only when mode ≠ find.

**Consequences.** Modes cost zero extra tool calls; a `mode=find` request is
bit-for-bit the pre-Phase-3 request. The analyze answer quality rides the
same evidence the loop already gathered.

## ADR-004 — 2026-07-15 — Featured scenes are curated cache, not trending

**Context.** A "featured" rail could be driven by popularity APIs or scraped
imagery; the brief forbids fake data and web-scraped assets, and demo
reliability matters more than freshness.

**Decision.** Featured entries are a hand-curated constant over clips the
pipeline has ALREADY cut locally; assets (thumb + 3s loop) are generated
offline from those clips by `generate_featured_assets.py`. Nothing is
downloaded at render time; a curated clip missing from cache is skipped.
Each card carries the canonical query that reproduces the scene through the
normal loop, so the rail is also an honest demo of retrieval.

**Consequences.** The rail only shows scenes that will definitely work
(cache-hit fast) in a demo. Refreshing it is a manual, deliberate act.

## ADR-005 — 2026-07-15 — Featured filtering is server-side, in manifest()

**Context.** Hooks/titles/queries are content that must pass the filter; the
filter lives in Python and reloads `blocked_terms.txt` on change.

**Decision.** `/api/featured` never serves an entry that fails the current
filter — checked per request in `manifest()`, not at generation time — so a
blocklist update retroactively hides an already-generated scene without
regenerating assets.

**Consequences.** A tiny per-request cost (regex over ~6 strings) buys
always-current filtering; the frontend needs no filter logic.

## ADR-002 — 2026-07-15 — Placeholder examples asserted at backend startup

**Context.** The cycling placeholder strings live in frontend JS, but the
content filter is Python. "All strings pass the content filter at build
time" needs a single source of truth without duplicating the blocklist in JS.

**Decision.** The backend, at startup, regex-extracts the
`PLACEHOLDER_QUERIES` constant from the static harness and runs each string
through `filter.check()`; a blocked string raises and fails startup. The
strings stay in one editable JS constant; the guard lives where the filter
lives.

**Consequences.** Editing a placeholder to something blocked is caught the
next time the server starts, not in production render. If the constant is
renamed, the guard logs a warning instead of silently passing.

## ADR-003 — 2026-07-15 — Hero input by re-parenting the existing dock

**Context.** Phase 1 wants the input in the focal zone under the headline,
but the input, its listeners, and the submit flow already exist as the fixed
bottom dock — duplicating the form risks divergent behavior.

**Decision.** One form: JS moves the existing form element into the hero on
load (`.in-hero` restyles it in place) and moves it back to the body/fixed
dock when the first query dismisses the hero. Listeners survive re-parenting;
submit logic is untouched.

**Consequences.** Zero duplicated logic; the enlarged hero styling is pure
CSS scoped to `.dock.in-hero`.

## ADR-001 — 2026-07-15 — Docs are in-repo, dated, phase-gated

**Context.** The engagement layer lands in five phases; decisions made mid-
build evaporate unless written down at the moment they're made.

**Decision.** `docs/CHANGELOG.md` (reverse-chronological, one dated entry
per phase) and this file (one ADR per non-obvious choice). A phase is not
"done" until both entries exist, dated with the actual implementation date.
`docs/DEMO_SCRIPT.md` is deliberately written last (Phase 5) so it describes
the system as it actually ends up, not as it was planned.

**Consequences.** Slight ceremony per phase; in exchange the demo script and
future contributors get an accurate history.
