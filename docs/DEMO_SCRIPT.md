# Scene Sense — Demo Script (written 2026-07-15, Phase 5)

A click-by-click walkthrough for demoing the engagement layer to clients.
Executable by someone who didn't build the app. Timings measured on the
reference machine (RTX 3050 4GB, Ollama `qwen2.5:7b`, warm caches).

---

## Pre-demo checklist (~10 minutes, day of demo)

1. **Ollama is up with the model pulled**
   ```bash
   ollama list            # must show qwen2.5:7b
   ```
2. **Warm the model** (first LLM call after idle costs ~10s extra; the app
   sets `keep_alive=30m`, so one throwaway request keeps it resident):
   ```bash
   curl -s http://localhost:11434/api/generate \
     -d '{"model":"qwen2.5:7b","prompt":"ok","stream":false,"keep_alive":"30m"}' >/dev/null
   ```
3. **Start the backend**
   ```bash
   cd backend && .venv/bin/python -m app.main
   ```
   `curl http://127.0.0.1:8000/api/health` → expect `"device": "cuda"`,
   `"llm_provider": "ollama"`. The startup log must show
   `placeholder examples pass content filter`.
4. **Verify the video cache** — the demo leans on these already-processed
   sources (in `backend/data/downloads/`): The Dark Knight ("why so
   serious", `PoyejjJGajk`), Breaking Bad ("one who knocks"), Interstellar
   (23 years of messages), The Matrix ("guns, lots of guns" + Trinity),
   Blade Runner 2049 ("you look lonely"). On a fresh machine, run each
   featured card's query once the night before to fill the download +
   transcript caches.
5. **Generate featured assets** (idempotent, cache-only, ~5s):
   ```bash
   .venv/bin/python -m app.generate_featured_assets
   ```
   `curl http://127.0.0.1:8000/api/featured` → 6 items.
6. **Seed the returning-user moment**: open http://127.0.0.1:8000 in the
   demo browser, run one successful query (click the "why so serious"
   example chip), wait for the clip, close the tab. This stores the visit
   count and one Recent chip. (To demo the *first-visit* state instead:
   Settings → "Clear everything".)
7. Pick light or dark theme in the demo browser (both are polished);
   100% zoom.

## The demo flow

| # | Action | What to say | Expect |
|---|--------|-------------|--------|
| 1 | Open http://127.0.0.1:8000 | "It remembers you — locally. Nothing leaves the machine." | "Welcome back. Where should we look today?" headline; placeholder examples cycling in the input; Recent chip; Featured rail. Instant. |
| 2 | Hover 2–3 featured cards slowly | "These previews are cut from footage the pipeline actually processed — no stock, no scraping." | Muted 3s loops play instantly on hover; card lifts. |
| 3 | Click the **Interstellar** card → "Get this scene" | "One click re-runs the full find→fetch→locate→clip pipeline, fresh." | Overlay → stepper (knowledge base → fetching → locating → cutting) with the spinning reel → playable clip, Source chip "Interstellar", "⚡ cached" badge, 2–3 suggestion chips. **~15–25s.** |
| 4 | Click the **Recent** chip (the seeded Dark Knight quest) | "Your history is a launcher, not a filter bubble — it re-asks from scratch." | Correct Dark Knight clip again, ⚡ cached. **~15–25s.** |
| 5 | **+ Modes → Analyze Context**, then ask: `show me the 'I am the one who knocks' scene from Breaking Bad` | "Same retrieval, richer read-out: context, emotion, why it matters." | Clip + a 5–8 sentence explanation. **~25–40s** (longer generation). |
| 6 | Dismiss the mode chip (✕). **+ Modes → Extract Metadata**, ask: `show me the scene in The Dark Knight where the Joker says 'why so serious'` | "And for tooling/integration people: the structured layer underneath." | No prose — expanded strip: Source, Line, At, Window, Duration, KB confidence (~97%), Match. **~15–25s.** |
| 7 | *(Optional)* type a blocked request | "Guardrails are on every surface — inputs, suggestions, even old history." | One-line polite refusal, no pipeline run, UI keeps flowing. |
| 8 | *(Optional, honest cold run)* ask for an uncached scene, e.g. `show me the scene in Cars where Lightning McQueen gets a makeover` | "Cold path: it searches, downloads, transcribes on-GPU, then cuts." | Stepper keeps progress visible. **~60–120s** — only do this if the room wants to see the real pipeline. |

## Timing notes

- The "Scene found in X.Xs" badge only appears when a query completes in
  **under 8s** (`TIMING_BADGE_MAX_S` in `app/static/index.html`); with a
  local 7B model most warm runs land ~15–25s, so expect the **⚡ cached**
  badge alone — that is by design (never advertise slow runs).
- If Ollama went idle >30 min, the first query eats a ~10s model reload;
  re-run step 2 of the checklist.
- If YouTube search hiccups on the optional cold run, fall back to a
  featured card — everything else in the demo is cache-backed and offline.
