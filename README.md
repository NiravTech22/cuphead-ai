# Scene Sense

A conversational, provenance-grounded clip finder and scene analyst. Describe
a moment from a movie, show, or video — Scene Sense identifies the source
(local RAG knowledge base first, web fallback), fetches and transcribes the
footage, pinpoints the moment down to the spoken line, cuts a playable clip,
and explains the scene with cited sources. Runs locally: FastAPI backend,
local LLM via Ollama (Anthropic/OpenAI switchable), faster-whisper on GPU,
sentence-transformers KB on CPU.

## Run

```bash
cd backend
.venv/bin/python -m app.main        # http://127.0.0.1:8000
```

Configuration via `backend/.env` (see `.env.example`). `LLM_PROVIDER=ollama`
is the default; the KB, preference engine, and content filter
(`blocked_terms.txt`) are always active.

## Engagement Layer (added 2026-07-15)

A demo-ready layer over the existing pipeline — all additive, no retrieval
or theme changes:

- **Hero input** — enlarged focal input under the headline with cycling
  example placeholders and an accent-derived focus glow.
- **Featured Scenes** — a rail of real, cache-derived scene previews
  (`app/featured.py` + `python -m app.generate_featured_assets`); hover
  plays a 3-second muted loop; click opens a preview overlay whose "Get
  this scene" button submits the canonical query through the normal loop.
- **Modes** — Find Scene / Analyze Context / Extract Metadata: prompt and
  render variations over the unchanged retrieval loop.
- **Returning-user greeting + Recent Quests** — localStorage-only
  personalization; recent successful queries resubmit as fresh queries.
- **Cinematic feedback** — film-reel loading stepper, conditional
  "Scene found in X.Xs" / "⚡ cached" badge (never shown on slow runs).

Documentation: [docs/CHANGELOG.md](docs/CHANGELOG.md) ·
[docs/DECISIONS.md](docs/DECISIONS.md) ·
[docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md)

**Privacy.** All personalization is local: visit count, recent quests, and
profile live in the browser's localStorage; taste profiles live in
`backend/data/preferences.json`. Nothing is sent to any server beyond your
own backend.
