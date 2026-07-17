# Scene Sense — Demo Script v2 (2026-07-17)

A 90-second, five-beat sequence. Each beat lands one idea; don't narrate the
plumbing — the app shows it.

---

## Pre-demo checklist (~10 min, day of)

1. **Ollama up + warm** (`ollama list` shows `qwen2.5:7b`, then one throwaway
   generate with `keep_alive=30m`):
   ```bash
   curl -s http://localhost:11434/api/generate \
     -d '{"model":"qwen2.5:7b","prompt":"ok","stream":false,"keep_alive":"30m"}' >/dev/null
   ```
2. **Pre-process list** — these must be in `data/downloads/` with transcripts
   (run each query once the night before if the cache is cold): Dark Knight
   "why so serious", Breaking Bad "one who knocks", Interstellar messages,
   Matrix "guns lots of guns", Blade Runner 2049 "you look lonely".
3. **Regenerate derived data**:
   ```bash
   cd backend
   .venv/bin/python -m app.emotion_data --backfill      # emotions.json per video
   .venv/bin/python -m app.generate_featured_assets     # carousel thumbs + loops
   rm -f data/vibe_index.json                           # rebuilt on first vibe query
   ```
4. **Start** `.venv/bin/python -m app.main`; check `/api/health` (cuda +
   ollama) and the startup log line `placeholder examples pass content filter`.
5. **Seed localStorage**: open the app, run one successful query (any example
   chip), close the tab → returning-user greeting + a Recent chip for beat 1.
   Leave the glass-box panel COLLAPSED (default) so beat 4 has a reveal.
6. **Mic check**: browser has mic permission for localhost; speak once to
   confirm levels. Quiet room or lean close.
7. Run one vibe query ("feels like quiet heartbreak") to build the vibe index
   off-stage (~6s first time), so beat 3 is instant.

## The five beats (~90s)

**Beat 1 — Speak it (0:00–0:25).**
Click the mic. Say, clearly: *"Show me the scene in The Dark Knight where the
Joker says why so serious."* The words appear in the input — pause a breath so
the room sees speech become text — then it submits itself. The stepper reels
through kb → fetch → locate → clip; the scene lands with "⚡ cached".
> Say: "Nothing typed. Local speech-to-text, straight into the agent."

**Beat 2 — The scene has a pulse (0:25–0:45).**
Point at the waveform drawing in under the player. Hover it slowly — "angry ·
0:51" follows the cursor. Click the tallest peak: the video jumps there.
> Say: "That's the scene's emotional intensity — computed locally. Click a
> peak, land on the beat."

**Beat 3 — Search by feeling (0:45–1:05).**
Type (or speak): *"find me a scene that feels like quiet heartbreak."* Three
scene cards return — Blade Runner's "you look lonely" on top, with the why:
"high sad, feeling-matched dialogue — 47% vibe match." Click it; the clip
delivers through the normal pipeline.
> Say: "No title, no quote — just a mood, matched against what it's already
> watched. And if nothing truly matches, it says so instead of guessing."

**Beat 4 — Glass box (1:05–1:20).**
Before the clip of beat 3 finishes, expand the ticker at bottom-right. Scroll
the log: `kb.search → miss`, `fetch (cache hit)`, `locate: 3.8s–6.9s (score
0.59)`, `ffmpeg: cut 6.0s clip` — timestamped, real.
> Say: "Every line is the actual pipeline, not an animation. This is how it
> found the scene."

**Beat 5 — Take it with you (1:20–1:30).**
On the Dark Knight result, click **⬡ Scene DNA**. A 1200×630 card downloads:
key frame, the quote in serif, the emotional waveform as the DNA strip.
Open it full-screen. Hold.
> Say: "Every scene, with its fingerprint. Generated locally, yours to keep."

## Expected timings & recovery

- Cached queries: ~15–25s wall (LLM-dominated); the reel + stepper carry it.
- Vibe query (warm index): <1s to cards.
- Voice: stop-to-text ≤1s warm; first call after idle ~+3s (model load).
- If the mic misfires, the transcript is editable — fix a word and hit enter;
  that's a feature, show it.
- If a beat stalls, the featured carousel is the safety net: hover a card,
  click "Get this scene" — everything in it is cache-backed.
