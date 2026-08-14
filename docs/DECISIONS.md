# Verbatim — Decision Records

Dated, ADR-style. One short entry per non-obvious choice.

---

## ADR-016 — 2026-08-14 — Pivot to quote verification; leave retrieval pipeline untouched

**Context.** Scene Sense's differentiated engineering — provenance tags, a
three-part confidence ledger, a verifier with a hard zero-fabrication rule —
was built for a domain (movie/TV scene-finding) where a wrong-but-confident
answer costs nothing. That machinery is the actual product for a domain
where it does: verifying whether a public figure really said a claimed
statement.

**Decision.** Retarget the app (renamed Verbatim) at public figures'
on-the-record statements, and change ONLY what the domain requires: system
prompts and tool descriptions (`app/assistant.py`), the knowledge base
content and its ingestion path (`app/knowledge/`), and UI copy
(`app/static/index.html`). The retrieval pipeline itself — identify source →
fetch/transcribe → locate moment → cut clip → verify quote — is
content-agnostic by construction and needed no changes; this is treated as
confirmation the original architecture was sound, not luck.

**Consequences.** The Featured-statements rail lost its curated examples
(they pointed at cached movie clips) and starts empty until re-populated
with real statement clips. The movie knowledge base (TMDB-backed) was
replaced with a small, manually curated, high-confidence seed set — quantity
was traded for verifiable accuracy, since the app's own ethos is zero
fabrication and a KB entry is itself a claim.

## ADR-015 — 2026-07-17 — BTW scope enforced in code, not prompt

**Context.** The aside channel must never trigger heavy work on a GPU
already running the main pipeline — and prompts alone don't enforce
anything against a 7B model's whims.

**Decision.** `/api/btw` is a separate route whose implementation simply
has no path to fetch/whisper/ffmpeg — it calls `llm.chat` (no tools,
num_predict=300) and the CPU KB, nothing else. The model can only *ask*
for the pipeline by returning needs_pipeline, which the UI converts into a
user-approved queued MAIN query through the normal isolated flow. Aside
context is capped at the job registry's one-liner (query + resolved title)
plus the client's last 4 exchanges; both isolation directions hold because
the aside is a stateless separate request that never writes into main
retrieval messages.

**Consequences.** A misbehaving model can at worst answer wrongly, never
spin up video work. Factual-vs-action classification needed explicit
prompt examples (the verify loop caught over-triggering).

## ADR-014 — 2026-07-17 — Cancellation boundary and gate ownership

**Context.** "Stop" must free CPU/GPU/disk, but a single in-flight Whisper
forward pass or Ollama chunk cannot be interrupted mid-call, downloads are
in-process yt-dlp (no child to kill), and Starlette never finalizes a sync
SSE generator once the client disconnects.

**Decision.** Cancellation is cooperative at dense checkpoints — before
each LLM turn and tool stage, per whisper segment, per yt-dlp progress
callback — so worst-case latency is one segment/chunk; subprocesses
(ffmpeg) are registered and SIGTERM/SIGKILLed immediately regardless. The
Ollama adapter streams internally so cancel can close the connection and
end generation. Partial artifacts are registered as glob patterns and
deleted by the reaper; creators unregister on success so caches only ever
hold whole files. The single-pipeline gate is owned by the Job, released
exactly once via job lifecycle — including from cancel(), because
generator finalization is not guaranteed; overlap after a cancel is
bounded to one checkpoint interval.

**Consequences.** ffmpeg dies in ~0.2s, downloads abort within one
progress tick, LLM within one chunk; a cancelled job can linger at most a
few seconds of Whisper before its checkpoint fires, which the 4GB GPU
tolerates. Tab-close relies on a sendBeacon cancel rather than server-side
disconnect detection.

## ADR-013 — 2026-07-17 — Scene DNA renders server-side with Pillow

**Context.** The share card needs the source video's key frame, real fonts,
the emotion timeline, and the content filter — all of which live server-side.
A client canvas would re-download frame data, fight font loading, and
duplicate filter logic in JS.

**Decision.** One `GET /api/dna/{clip_id}` composes the PNG with Pillow
(already installed): ffmpeg key frame, DejaVu Serif italic for the quote
(closest system face to the site's serif voice — Fraunces isn't installed
locally and bundling a webfont for one image wasn't worth it), tracked-caps
metadata, autoscaled waveform strip, dark-token palette. Served as a
Content-Disposition attachment: download-only, no hosting, no og-tags.

**Consequences.** ~1s per card, zero client complexity, filter runs where
it lives. The serif is an approximation of Fraunces; dropping a Fraunces
TTF next to dna.py and pointing SERIF/SERIF_IT at it upgrades fidelity with
no code change.

## ADR-012 — 2026-07-17 — Vibe scoring: 0.6 text + 0.4 emotion, threshold 0.42

**Context.** "Feels like quiet heartbreak" has two signals: what the
dialogue *means* (semantic) and how the scene *feels* (Phase-0 emotion
profile). Either alone misfires — pure text matches words like "heart",
pure emotion can't tell two sad scenes apart.

**Decision.** Blended cosine score: `0.6·cos(query_emb, scene_text_emb) +
0.4·cos(feeling_target, scene_emotion)`, where feeling targets come from an
editable `feelings.json` (word → emotion vector); with no feeling word
matched, scoring is text-only. Floor 0.42 (tuned on the current library so
real matches clear it and off-vibe scenes don't); below it the tool returns
an empty result and the assistant says so — a fake match would poison trust
in every other answer. One scene per video in the top-3 for variety.

**Consequences.** CPU-only, ~0.2s warm. The joy query on today's library
returns the empty state — correct, if unflattering; the fix is processing
more videos, which the empty-state copy says.

## ADR-011 — 2026-07-17 — Voice reuses the single Whisper, filter-before-return

**Context.** The 4GB GPU already hosts Whisper transiently plus the LLM; a
second ASR instance is not an option, and voice transcripts are a new
inbound surface the content filter must own.

**Decision.** `/api/voice` calls the same `transcribe()` as the pipeline; a
module-level lock added in transcriber.py serializes all model access (the
lock lives where the model lives, so both callers get it for free). The
filter runs server-side on the transcript BEFORE the response — a blocked
utterance returns only the standard refusal string, never the text. The
client shows the transcript for a 1.2s editable beat before auto-submitting
so the speech→text moment is visible and correctable.

**Consequences.** Zero VRAM delta; a voice request during an active
pipeline transcription waits briefly instead of racing the model. Blocked
speech is never rendered, logged only by category.

## ADR-010 — 2026-07-17 — One emotion contract; dialogue-affect generator

**Context.** Three features (scrubber, vibe search, DNA card) consume
emotion timelines. The facial detector (`fer`) was never installed — it
drags in TensorFlow, and the 4GB GPU is already committed to Whisper +
Ollama. New GPU-resident models are banned.

**Decision.** A single, generator-agnostic contract per video
(`<videoId>.emotions.json`, 7-class distribution + dominant + intensity per
point; intensity = 1 − P(neutral), i.e. "how charged is this moment").
Today's generator scores emotion from the *dialogue*: transcript segments
embedded on the existing CPU MiniLM, softmax over cosine similarity to a
small set of per-emotion anchor sentences (T=0.06). Videos without usable
dialogue get no file — consumers must handle absence honestly rather than
render fabricated feeling.

**Consequences.** Zero VRAM, ~ms per segment, works for every transcribed
video retroactively. The signal reflects what is *said* rather than shown —
good for dialogue-driven scenes, blind to silent visuals; a future facial
generator can write the same file and consumers won't change.

## ADR-009 — 2026-07-17 — Header controls differentiated by function

**Context.** The menu opener and the theme toggle were both iOS pill
switches — a navigation control and a state control wearing the same
clothes, and two identical pills bracketing the wordmark competed with it.

**Decision.** Morphing icon for menu (staggered strokes → ✕, 250ms
transform morph), celestial morph for theme (line-art sun → crescent via
CSS geometry transitions on an SVG mask, 350ms + 1.05 pulse). Both are
quiet icon-only buttons in existing tokens with ≥40px hit areas and
negative margins preserving header metrics; reduced-motion collapses both
morphs to instant swaps via the existing global rule. The pill switch
survives only where it genuinely represents a boolean setting (Settings
mini toggles).

**Consequences.** Control shape now communicates control kind; the header
reads as punctuation around the wordmark. The crescent relies on CSS
cx/cy geometry-property animation (supported in all evergreen browsers);
if a legacy browser ignores it, the states still render correctly —
only the tween is lost.

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
