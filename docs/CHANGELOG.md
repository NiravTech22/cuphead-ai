# Scene Sense — Changelog

Reverse-chronological. Every engagement-layer phase gets a dated entry on
completion: what was built, files created/modified, key decisions, and how it
was verified.

## [2026-07-16] Featured rail → auto-scrolling carousel, scrollbars removed

**What was built.** The Featured Scenes rail now auto-scrolls continuously
(~30px/s via requestAnimationFrame): the card set is rendered twice
(duplicates aria-hidden, untabbable) so the wrap at the halfway point is
seamless. Scrolling pauses on hover/touch/focus — hover previews play on a
still rail — and resumes on leave; manual scrolling still works. Under
`prefers-reduced-motion` the rail is static (single card set, no
auto-scroll). Scrollbars are hidden in all engines (`scrollbar-width: none`
+ `::-webkit-scrollbar { display: none }`).

**Files modified.** `backend/app/static/index.html`.

**How it was verified.** Headless Chrome: 12 cards rendered (6 + 6 clones);
scrollLeft advanced 17→77 over 2s; froze during hover and resumed after;
jumped to just before the halfway point and confirmed the wrap
(scrollLeft 35 < half 1208, no snap-back visible); computed
`scrollbar-width: none`; reduced-motion context rendered 6 static cards.

## [2026-07-15] Phase 5 — Cinematic feedback loop + demo script

**What was built.** The loading state is now a film-reel SVG (accent token,
1.6s spin, hidden on settle; static under reduced-motion via the existing
global rule) leading a stage stepper that renders the pipeline's real status
labels (knowledge base → fetch → locate → clip) — passed stages dim, the
current one pulses. Total query wall-time is instrumented client-side:
results show "Scene found in X.Xs" only when total < 8s
(`TIMING_BADGE_MAX_S`, one constant), "⚡ cached" whenever the source video
came from the download cache (new additive `cache_hit` payload field), and
nothing time-related on slow runs. Result cards enter with a single
fade + 8px rise at 250ms (tightened from 450ms). `docs/DEMO_SCRIPT.md`
written: pre-demo checklist (cache warm-up, asset generation, Ollama
keep-alive, localStorage seeding) plus a click-by-click flow with expected
timings, including a deliberate cached query and an Analyze-mode query.

**Files created/modified.** `backend/app/assistant.py` (cache_hit),
`backend/app/static/index.html`, `docs/DEMO_SCRIPT.md`.

**Key decisions.** ADR-008 (timing badge is conditional and client-measured;
slow runs show ⚡ at most).

**How it was verified.** renderMeta unit-tested in-browser against five
synthetic payloads (fast+cached, fast, slow+cached, slow, no-clip) — all
badge outputs correct. Live cached Dark Knight run: reel spinning
(computed animation) during the run, hidden on settle, "⚡ cached" shown and
no time advertised (run was ~20s). Regression sweep in one session: scene A
(Dark Knight) → unrelated scene B (Interstellar) both resolved to the
correct sources with clips; suggestion chips still rendered (3); a
blocklisted query got the polite one-line refusal with no clip and was not
recorded into Recent Quests.

## [2026-07-15] Phase 4 — Returning-user greeting + Recent Quests

**What was built.** `ss_visits` / `ss_recent` in localStorage (respecting
the existing "remember me" toggle via the `persist` helper). Return visits
crossfade the headline (≤280ms; instant under reduced-motion) to "Welcome
back. Where should we look today?" in the same serif style. Successful clip
deliveries record {query, title, timestamp} (last 5, deduped); up to 3
render as a "Recent" chip row above the featured rail, each with a ✕ and a
row-level "clear". Chips resubmit their stored query as a normal fresh
query. Before rendering, stored text is re-checked against the CURRENT
blocklist via a new `POST /api/filter/check` (covers a filter list updated
after a quest was stored); if the check can't run, nothing renders. New
`POST /api/filter/check` logs categories only, never the text. The
suggestion-chip system is untouched — Recent is a separate additive row.
README already carries the privacy note (all personalization is local).

**Files created/modified.** `backend/app/main.py` (filter/check),
`backend/app/static/index.html`.

**Key decisions.** ADR-007 (recent quests re-filtered server-side at render
time, fail-closed).

**How it was verified.** Headless Chrome, fresh profile: visit 1 showed the
untouched headline and no Recent row; a Breaking Bad clip query succeeded;
visit 2 showed the greeting and one "Breaking Bad (2008) ✕" chip; clicking
it reran the full loop and returned Source "Breaking Bad (2008)" (no state
leakage — server log shows the auto-fetched video cleared between queries);
✕ removed the chip, "clear" emptied the row, and the cleared state survived
reload. `POST /api/filter/check` returned `[false, true]` for a benign
string vs one containing a blocklisted term. A load-order bug (the block
referenced `persist` before its `const` initialization, killing the script)
was caught by the browser run and fixed by relocating the block.

## [2026-07-15] Phase 3 — "+ Modes" pill (Find / Analyze / Extract Metadata)

**What was built.** A "+ Modes" pill inside the input bar opening a compact
radio-style popover (existing tokens; tab/enter/esc accessible). Selecting a
mode shows a dismissible label chip ("Analyze ✕") in the bar. `mode` rides
the chat request as a query param (`find` sends nothing). Backend:
`assistant.run(..., mode=)` appends one mode instruction to the
orchestration prompt only AFTER the clip exists — Analyze asks for lead-in +
emotional read + why-it-matters (5–8 sentences); Metadata asks for a
one-liner and the UI suppresses prose, rendering an expanded metadata strip
(source, line, moment, clip window, duration, KB confidence, match score,
video). Mode-only payload fields are added only when mode ≠ find, keeping
the default path byte-identical.

**Files created/modified.** `backend/app/assistant.py`,
`backend/app/main.py`, `backend/app/static/index.html`.

**Key decisions.** ADR-006 (modes are prompt/render variations over the
existing loop; instruction injected post-locate so retrieval is identical).

**How it was verified.** Headless Chrome, cached Dark Knight query in all
three modes in one session: Find sent no `mode` param and returned clip +
normal answer; Analyze returned the clip plus a 736-char contextual
explanation (vs 363 for Find); Metadata suppressed prose and rendered the
expanded strip (KB confidence 97%, match 53%, window 93.3–99.3s) with the
clip. Popover opened/closed via keyboard (Tab/Enter/Esc); chip dismissed
back to "+ Modes". Server logs confirmed per-query isolation still clears
the auto-fetched video between mode queries. Two UI bugs found and fixed
during verification: the fixed dock's `pointer-events: none` blocked popover
clicks, and the send-button styling (`.bar button`) squashed the pill to a
46px circle — both scoped correctly now.

## [2026-07-15] Phase 2 — Featured Scenes rail (real cache data)

**What was built.** `app/featured.py`: a curated constant mapping six
existing cached clips (Dark Knight, Breaking Bad, Interstellar, The Matrix
×2, Blade Runner 2049) to title/hook/canonical-query, plus `manifest()`
which drops entries with missing assets or filter hits.
`app/generate_featured_assets.py` (run: `python -m
app.generate_featured_assets`) uses ffmpeg to cut, per scene, a 480px
thumbnail (frame at 40%, ~20% desaturated via `eq=saturation=0.8`) and a
3-second muted ~480p faststart loop from the clip midpoint — cache-only,
nothing downloaded. New `GET /api/featured` + `/featured` static mount.
UI: a horizontal rail under the hero input at 0.75 opacity (full on
hover/focus/in-view), card hover → scale(1.05) + inline muted loop
autoplay; reduced-motion shows a play glyph and never autoplays. Click →
preview overlay (panel/card tokens) with the loop, hook line, and a single
"Get this scene" button that closes the overlay and submits the canonical
query as a normal fresh query. Esc/backdrop close. Missing/empty manifest →
rail doesn't render.

**Files created/modified.** `backend/app/featured.py`,
`backend/app/generate_featured_assets.py`, `backend/app/main.py`,
`backend/app/static/index.html`.

**Key decisions.** ADR-004 (featured = curated cache, not trending),
ADR-005 (filtering happens server-side in `manifest()`).

**How it was verified.** Generator produced 6 thumb+loop pairs (5–120KB);
`/api/featured` returned 6 items. Headless Chrome: 6 cards rendered with
real thumbnails; hover flipped the loop from paused→playing; overlay
opened, Esc closed it; "Get this scene" on The Dark Knight ran the full
standard pipeline and rendered a clip with source chip "The Dark Knight
(2008)". Empty state: with featured.json moved aside the API returned
`{"items":[]}` and the page rendered with no rail; manifest restored after.

## [2026-07-15] Phase 1 — Hero input field

**What was built.** The central input now renders inside the hero, directly
under the headline (the form re-parents into `#hero` on load and returns to
the fixed bottom dock on the first ask). Scaled ~1.2× (18px text, 46px send
button) with a layered soft shadow. Placeholder examples cycle every 4s with
a 360ms crossfade from a single editable `PLACEHOLDER_QUERIES` constant;
cycling stops permanently on focus/typing; `prefers-reduced-motion` gets the
static first string. Focus transitions the bar border to the existing accent
(`--link`) plus one derived glow (`color-mix` of the accent at 24%) over
250ms. Example chips gained a hover lift (translateY(-1px) + panel-token
background). Backend startup asserts every placeholder string passes the
content filter.

**Files created/modified.** `backend/app/static/index.html`,
`backend/app/main.py` (startup assertion).

**Key decisions.** ADR-002 (placeholders asserted server-side at startup),
ADR-003 (dock re-parenting instead of a second input).

**How it was verified.** Headless Chrome (system, via Playwright): input
sits under the headline at 1366px and 390px; placeholder text changed across
a 4.4s wait and stopped after focus; `.bar` bounding box identical across a
cycle (no layout shift); focus-glow screenshots in light and dark; dock
`position: fixed` again after submitting; reduced-motion context kept the
first string static. Server log: `placeholder examples pass content filter
(4 checked)`.

## [2026-07-15] Phase 0 — Documentation scaffolding

**What was built.** `docs/` directory with this changelog and
`DECISIONS.md` (ADR-style records). `docs/DEMO_SCRIPT.md` is written last,
in Phase 5. Created the project `README.md` (the repo had none) with a dated
Engagement Layer section linking to these docs.

**Files created/modified.** `docs/CHANGELOG.md`, `docs/DECISIONS.md`,
`README.md` (new).

**Key decisions.** See ADR-001 (docs live in-repo, dated with implementation
date, one entry per phase — no phase is done until its entries exist).

**How it was verified.** Files render as valid Markdown; README links
resolve to the docs files.
