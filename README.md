# Verbatim

A conversational, provenance-grounded quote-verification tool for public
figures' on-the-record statements. Ask "did they really say that?" about a
speech, interview, or press conference — Verbatim identifies the source
(local RAG knowledge base first, web fallback), fetches and transcribes the
footage, pinpoints the moment down to the spoken line, cuts a playable clip,
and explains the statement with cited sources. It refuses to confirm a quote
it can't ground in a transcript or citation — an honest "couldn't verify"
beats a confident wrong answer.

> Formerly Scene Sense (a movie-scene finder). Pivoted 2026-08-14 to target
> journalists, fact-checkers, and researchers who need to verify a claimed
> public statement rather than relive a favorite movie line — same pipeline,
> different domain. See [docs/CHANGELOG.md](docs/CHANGELOG.md) for the
> rename.

**Scope.** Public figures' public, on-the-record statements only — speeches,
interviews, press conferences, floor debates. Not for private individuals,
not for surveillance, not for content someone hasn't already put on the
record. Runs entirely locally: FastAPI backend, local LLM via Ollama
(Anthropic/OpenAI switchable), faster-whisper on GPU, sentence-transformers
KB on CPU.

## Run

```bash
cd backend
.venv/bin/python -m app.main        # http://127.0.0.1:8000
```

Configuration via `backend/.env` (see `.env.example`). `LLM_PROVIDER=ollama`
is the default; the KB, preference engine, and content filter
(`blocked_terms.txt`) are always active.

## Knowledge base

```bash
# ~30 well-documented public statements, no key needed:
.venv/bin/python -m app.knowledge.seed

# your own structured events (JSON) — see the module docstring for the shape:
.venv/bin/python -m app.knowledge.ingest_events /path/to/events.json

# your own transcripts (.srt of speeches/interviews/press conferences):
.venv/bin/python -m app.knowledge.ingest_subtitles /path/to/srts
```

## Engagement Layer (added 2026-07-15)

A demo-ready layer over the existing pipeline — all additive, no retrieval
or theme changes:

- **Hero input** — enlarged focal input under the headline with cycling
  example placeholders and an accent-derived focus glow.
- **Featured Statements** — a rail of real, cache-derived statement previews
  (`app/featured.py` + `python -m app.generate_featured_assets`); hover
  plays a 3-second muted loop; click opens a preview overlay whose "Get
  this clip" button submits the canonical query through the normal loop.
  Starts empty after the pivot — populate it by asking a few real
  verification questions, then adding the resulting clip_ids to
  `app/featured.py`'s `CURATED` list.
- **Modes** — Find Statement / Analyze Context / Extract Metadata: prompt and
  render variations over the unchanged retrieval loop.
- **Returning-user greeting + Recent Quests** — localStorage-only
  personalization; recent successful queries resubmit as fresh queries.
- **Cinematic feedback** — film-reel loading stepper, conditional
  "Clip found in X.Xs" / "⚡ cached" badge (never shown on slow runs).

Documentation: [docs/CHANGELOG.md](docs/CHANGELOG.md) ·
[docs/DECISIONS.md](docs/DECISIONS.md) ·
[docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)

**Privacy.** All personalization is local: visit count, recent quests, and
profile live in the browser's localStorage; taste profiles live in
`backend/data/preferences.json`. Nothing is sent to any server beyond your
own backend.
