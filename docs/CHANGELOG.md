# Scene Sense — Changelog

## [2026-07-17] To-the-Stars Phase 5 — Scene DNA card + demo script v2

**What was built.** A "⬡ Scene DNA" button on every clip result generates a
1200×630 PNG share card, rendered SERVER-SIDE with Pillow (ADR-013):
desaturated key frame from the clip midpoint under an espresso gradient,
the quote in serif italic (adaptive size, ≤3 lines, content-filtered before
drawing — as is the title), source + timestamp + emotion in tracked caps,
the Phase-0 emotion waveform as the "DNA" strip (±10s context, autoscaled
so uniform-intensity clips still read as a shape), and the wordmark with a
restrained line-dot ornament. Dark-token palette only. `GET
/api/dna/{clip_id}` serves it as an attachment — download-only, no hosting,
no share links. `docs/DEMO_SCRIPT.md` rewritten as the 90-second five-beat
sequence (speak → waveform → vibe → glass box → DNA card) with the full
pre-demo checklist.

**Files created/modified.** `backend/app/dna.py` (new),
`backend/app/main.py`, `backend/app/static/index.html`,
`docs/DEMO_SCRIPT.md`.

**Key decisions.** ADR-013 (server-side Pillow, download-only).

**How it was verified.** Direct render inspected at full size (two
iterations: the first strip was a flat slab on a uniformly intense clip;
autoscale + context fixed it). In-app: clicking ⬡ Scene DNA downloaded
`scene-dna-<id>.png` (260KB) via the browser download event. Final
regression sweep in one session: carousel (12 cards) + header icon buttons
+ mic present; scene A (Dark Knight, ⚡ cached, scrubber, 3 suggestion
chips) → unrelated scene B (Interstellar, clip) with clean isolation;
blocked query → polite refusal, no clip; Recent chips (2) on return visit.
**Demo-ready: yes.**

## [2026-07-17] To-the-Stars Phase 4 — Glass-box agent panel

**What was built.** The pipeline now emits `trace` SSE events at every REAL
step — one terse ≤80-char line per tool result plus one per LLM turn
(`kb.search "…" → hit/miss (conf)`, `fetch (cache hit): … · N segs`,
`locate: 3.8s–6.9s (score 0.59)`, `ffmpeg: cut 6.0s clip`, `vibe: below
threshold — honest miss`), each stamped with seconds since query start.
Nothing is fabricated or timer-driven; lines are derived from actual tool
outputs in the same generator, riding the existing SSE stream
(fire-and-forget, zero added latency). Frontend: a slim fixed panel
(bottom-right; full-width above the dock on mobile) in monospace metadata
styling — collapsed by default to a one-line ticker of the current step,
expandable, open/closed remembered in localStorage, with per-query
separators and scrollback across the session.

**Files modified.** `backend/app/assistant.py`,
`backend/app/static/index.html`.

**Key decisions.** Trace lines are constructed at the yield site from real
tool outputs — the panel cannot show a step that didn't run.

**How it was verified.** Matrix query with the panel open: 10 lines
streamed; three spot-checked 1:1 against server logs for the same request —
"fetch (cache hit)" ↔ `download cache hit for j_urZ5KDPec`, "locate:
3.8s–6.9s" ↔ `chose [3.84-6.86]`, "ffmpeg: cut 6.0s clip" ↔ `clip -> …
(6.0s)`. All lines ≤80 chars; ticker showed the live step and settled to
"agent idle"; open state persisted; a second (vibe) query appended under a
new separator with history intact. Panel hidden entirely until first use.
**Demo-ready: yes.**

## [2026-07-17] To-the-Stars Phase 3 — Vibe search

**What was built.** A `vibe_search` tool the orchestration LLM calls when a
query describes a feeling rather than an identifiable scene (prompt case C;
no keyword regex — the model decides). `app/vibe.py` builds a CPU-only
index over PROCESSED videos: ~22s transcript scene-windows, each with a
MiniLM embedding + mean Phase-0 emotion vector, cached to
`data/vibe_index.json`. Score = 0.6·text-cosine + 0.4·emotion-cosine
against targets from editable `backend/feelings.json` (feeling-word →
emotion-vector; text-only when no feeling word matches). Threshold 0.42,
one scene per video for variety, results content-filtered. Top-3 render as
compact theme cards (on-demand desaturated thumbs served at `/vibe/`) with
a one-line why ("high sad, feeling-matched dialogue — 47% vibe match");
clicking a card submits a normal fresh clip query. Below threshold → an
honest empty note, never a fake match. The clip-nudge logic is bypassed
when vibe_search ran.

**Files created/modified.** `backend/app/vibe.py` (new),
`backend/feelings.json` (new), `backend/app/assistant.py`,
`backend/app/main.py`, `backend/app/static/index.html`.

**Key decisions.** ADR-012 (blend weights + threshold; honest empty).

**How it was verified.** Four distinct queries: "quiet heartbreak" →
Blade Runner "you look lonely" / Dark Knight sad window / No Strings
Attached; "menacing threat" → Breaking Bad "one who knocks" on top; "pure
joy and celebration" → honest empty state (the cached library genuinely
has no joyful scene above threshold); "existential dread" → highest-fear
scene. Warm latency 0.01–0.23s (<2s target). Full UI e2e: LLM invoked the
tool, 3 cards rendered, clicking the top card delivered the Blade Runner
2049 (2017) clip via the standard loop. **Demo-ready: yes.**

## [2026-07-17] To-the-Stars Phase 2 — Emotion scrubber

**What was built.** Every delivered clip now renders a 64px interactive
emotional waveform under the player, built from the Phase-0 timeline for
the clip's window (clip `source_path` → videoId → `/api/emotions`). Custom
SVG, no chart look: Catmull-Rom-smoothed 1.5px accent line over
per-dominant-emotion area slices, each a vertical gradient of a
color-mix-derived token tint fading to transparent (no new raw colors).
Hover: hairline cursor + lowercase caption ("angry · 0:51") in the
metadata typography; click seeks the video (clip-relative); a playhead
hairline tracks playback; touch = tap to seek, hold ≥350ms for the
caption. Entrance draws the line left-to-right (600ms stroke-dashoffset);
reduced-motion renders instantly. Fewer than 8 timeline points → no
scrubber (honest absence, e.g. dialogueless sources).

**Files modified.** `backend/app/static/index.html`.

**Key decisions.** Consumes ADR-010's contract; tints derived exclusively
from existing tokens via color-mix.

**How it was verified.** Live Breaking Bad query: scrubber rendered; click
at 50% seeked to 3.23s of a 6.01s clip (±0.5s target met); playhead
advanced past midpoint during playback; hover caption read "angry · 0:51"
— emotionally correct for the "I am the danger" beat. Screenshot confirms
it reads as native linework, with ⚡ badge and suggestion chips intact
alongside. **Demo-ready: yes** (voice-in + waveform-out both live).

## [2026-07-17] To-the-Stars Phase 1 — Voice search

**What was built.** A thin line-art mic button inside the input bar with
idle / recording (accent pulse ring, 1.2s period, live 0:SS elapsed
readout) / transcribing states. Capture via MediaRecorder (webm/opus),
click-to-stop, 15s hard stop. New `POST /api/voice` transcribes with the
EXISTING faster-whisper install (no second instance; a new module-level
lock in transcriber.py serializes the single model between voice and the
pipeline). The transcript passes the content filter BEFORE returning — a
blocked transcript never reaches the client (polite refusal, input
cleared). On success the text lands visibly in the input for a 1.2s
editable beat (typing cancels the auto-submit), then submits through the
normal clean-context loop. Mic-permission denial disables the button with
"Mic unavailable"; the app is otherwise unaffected. Transcription time and
device are logged per call.

**Files created/modified.** `backend/app/main.py`,
`backend/app/transcriber.py`, `backend/app/static/index.html`.

**Key decisions.** ADR-011 (voice reuses the one Whisper via a lock;
filter-before-return).

**How it was verified.** Direct endpoint: 8s of real movie audio
transcribed in **0.87s** on the warm model (target ≤3s), device=cuda, no
new model loaded. Full e2e with Chrome's fake-mic fed a synthesized spoken
query: recording state + timer showed, transcript "Show me the scene in
the dark night where the Joker says why so serious." landed in the input,
auto-submitted after the beat, and the correct Dark Knight clip was
delivered (KB absorbed the "dark night" homophone). Permission-denial
path: button disabled, title "Mic unavailable", rest of the app usable.
**Demo-ready: yes.**

## [2026-07-17] To-the-Stars Phase 0 — Shared emotion-data contract

**What was built.** One per-video contract file
`data/emotions/<videoId>.emotions.json`: {videoId, source, fps_sampled,
duration, points:[{t, emotions:{7 classes}, dominant, intensity}]}, where
**intensity = 1 − P(neutral)** (documented choice, ADR-010). Generator
"transcript-affect-v1": each transcript segment is embedded with the
existing CPU MiniLM and softmaxed against per-emotion anchor sentences —
zero new models, zero VRAM (the facial `fer` detector was never installed;
when it is, it can write the same contract). `app/emotion_data.py` exposes
`build_for_video`, `classify`, `get_timeline(videoId, start, end)` (linear
resampling, ≥40 and ≤200 plot-ready points), a `--backfill` CLI, and
`GET /api/emotions/{video_id}?start&end`. `video_id` added to the session
video summary so the frontend can address timelines.

**Files created/modified.** `backend/app/emotion_data.py` (new),
`backend/app/main.py`, `backend/app/session.py`.

**Key decisions.** ADR-010 (single contract; dialogue-affect generator).

**How it was verified.** Backfill produced emotions.json for 14/17 cached
videos — the 3 without are dialogueless (UFC montage, Sardaukar chant, Cars
engine scene): no data is fabricated for them and the accessor/endpoint
return an empty-points contract cleanly. Spot-check: the Dark Knight scar
monologue window (93–99s) resolves to angry/disgust at 0.77–0.86 intensity.
Endpoint returned 40 resampled points for that window; full video capped
at 200.

Reverse-chronological. Every engagement-layer phase gets a dated entry on
completion: what was built, files created/modified, key decisions, and how it
was verified.

## [2026-07-17] Header controls replaced: morphing menu icon + celestial theme button

**What was built.** Both header pill switches are gone. Menu (top-left): an
icon-only button with two staggered strokes (18/12px, 1.5px weight) that
MORPH into a ✕ over 250ms (rotate + translate, no fade); hover nudges the
lines 2px and lifts opacity; `aria-expanded`/`aria-controls` + live
aria-label. Theme (top-right): an icon-only button drawing a thin line-art
sun whose rays retract while a masked "bite" circle slides in (CSS cx/cy
geometry transitions on SVG mask + arc, ~350ms) to carve a line-art
crescent — a real morph, not a crossfade — with a soft 1.05 scale pulse on
click; aria-label/title always state the action ("Switch to light mode").
Both controls: 40×40 hit areas with negative margins so header metrics are
unchanged, accent focus-visible rings, reduced-motion instant swaps (global
rule). Theme logic untouched — the button only flips `data-theme` and
persists as before. The pill `.switch` component survives solely for the
Settings mini toggles.

**Files modified.** `backend/app/static/index.html`.

**Key decisions.** ADR-009 (header controls differentiated by function).

**How it was verified.** Headless Chrome: zero `.switch` elements in the
header; menu open set aria-expanded=true with a rotated stroke (computed
transform) and closed back on Esc; mid-transition the bite circle's
computed `cx` read 15.97px between its 19px and 14.8px endpoints — proof of
interpolation, not a swap; rays' opacity hit 0 in dark; theme toggled
dark→light→dark with localStorage persistence and aria-labels updating;
Enter on each focused control operated it (menu open/close, theme toggle);
3× zoom screenshots of light (sun), dark (crescent), and menu-open (✕)
states; Settings mini-switches confirmed intact.

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
