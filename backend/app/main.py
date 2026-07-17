"""main.py — FastAPI app: streaming chat (SSE), /clips static mount, video load.

Endpoints:
  GET  /api/health                      -> status + device profile + api-key presence
  GET  /api/chat?question=&session_id=  -> Server-Sent Events stream of the answer
  POST /api/load_url                     -> {url, session_id}: fetch+transcribe a video
  POST /api/upload                       -> multipart file: upload+transcribe a video
  GET  /api/session/{session_id}         -> loaded-video summary + emotion timeline
  POST /api/emotion/{session_id}         -> build the emotion timeline for the loaded video
  /clips/*                               -> static clip files (inline <video> playback)
  /                                      -> minimal built-in chat harness (or the React build)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import assistant, config, featured
from . import filter as content_filter
from .device import detect
from .llm import local_model, provider
from .downloader import download
from .frame_emotion import build_timeline
from .logging_setup import get_logger
from .session import STORE, LoadedVideo
from .transcriber import transcribe

log = get_logger(__name__)

app = FastAPI(title="Scene Sense")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve exported clips so the frontend can embed them inline.
app.mount("/clips", StaticFiles(directory=str(config.CLIPS_DIR)), name="clips")

# Featured-scene assets (thumbnails + preview loops), generated from cache by
# `python -m app.generate_featured_assets`.
featured.FEATURED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/featured", StaticFiles(directory=str(featured.FEATURED_DIR)), name="featured")

FRONTEND_DIST = config.BACKEND_DIR.parent / "frontend" / "dist"
STATIC_HARNESS = Path(__file__).resolve().parent / "static"


@app.get("/api/health")
def health() -> dict:
    prov = provider()
    return {
        "status": "ok",
        "device": detect(),
        "llm_provider": prov,
        "model": local_model() if prov == "ollama" else config.ASSISTANT_MODEL,
    }


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/api/chat")
def chat(question: str, session_id: str = "default", mode: str = "find") -> StreamingResponse:
    """Stream the assistant's answer as Server-Sent Events. Runs in a threadpool.

    `mode` is per-query render/prompt metadata (find|analyze|metadata) — it never
    enters the retrieval context (see assistant.run)."""

    def gen():
        try:
            for event in assistant.run(question, session_id, mode=mode):
                yield _sse(event)
        except Exception as exc:  # never crash the stream
            log.exception("chat stream error")
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/load_url")
def load_url(url: str = Form(...), session_id: str = Form("default")) -> dict:
    """Fetch a video by URL, transcribe it, and load it into the session (Mode A)."""
    info = download(url)
    if not info.local_path:
        return {"ok": False, "error": "download failed"}
    transcript = transcribe(info.local_path)
    session = STORE.get(session_id)
    session.video = LoadedVideo(
        path=info.local_path,
        title=info.title,
        channel=info.channel,
        webpage_url=info.webpage_url,
        duration=info.duration,
        transcript=transcript,
        video_id=info.id,
        pinned=True,  # user-loaded: survives across questions
    )
    return {"ok": True, "video": session.video.summary()}


@app.post("/api/upload")
def upload(file: UploadFile = File(...), session_id: str = Form("default")) -> dict:
    """Upload a local video file, transcribe it, and load it into the session."""
    dest = config.UPLOADS_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    transcript = transcribe(str(dest))
    session = STORE.get(session_id)
    session.video = LoadedVideo(
        path=str(dest),
        title=file.filename,
        duration=transcript.duration,
        transcript=transcript,
        pinned=True,  # user-loaded: survives across questions
    )
    return {"ok": True, "video": session.video.summary()}


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict:
    session = STORE.get(session_id)
    if not session.video:
        return {"video": None, "emotion_timeline": []}
    return {
        "video": session.video.summary(),
        "emotion_timeline": session.video.emotion_timeline,
    }


@app.post("/api/emotion/{session_id}")
def build_emotion(session_id: str, fps: float = config.EMOTION_FPS) -> dict:
    """Build the facial-EXPRESSION timeline for the loaded video (step 8)."""
    session = STORE.get(session_id)
    if not session.video:
        return {"ok": False, "error": "no video loaded"}
    tl = build_timeline(session.video.path, fps=fps)
    session.video.emotion_timeline = tl["timeline"]
    return {"ok": True, **tl}


@app.get("/api/featured")
def get_featured() -> dict:
    """Curated cache-derived scenes for the hero rail (already filter-checked)."""
    return {"items": featured.manifest()}


@app.post("/api/filter/check")
def filter_check(payload: dict = Body(...)) -> dict:
    """Batch content-filter check for client-side surfaces (e.g. Recent Quests
    stored in localStorage: the blocklist may have changed since they were
    saved). Returns blocked flags only — categories are logged server-side."""
    texts = payload.get("texts") or []
    blocked = []
    for t in texts[:50]:
        cat = content_filter.check(t if isinstance(t, str) else "")
        if cat:
            log.info("filter/check blocked a client string (category=%s)", cat)
        blocked.append(bool(cat))
    return {"blocked": blocked}


@app.get("/api/settings")
def get_settings() -> dict:
    return {"filter_strict": content_filter.is_strict()}


@app.post("/api/settings/filter")
def set_filter(strict: bool = Form(...)) -> dict:
    return {"filter_strict": content_filter.set_strict(strict)}


@app.get("/")
def index() -> FileResponse:
    dist_index = FRONTEND_DIST / "index.html"
    if dist_index.exists():
        return FileResponse(str(dist_index))
    return FileResponse(str(STATIC_HARNESS / "index.html"))


# Serve the built React app if present.
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


@app.on_event("startup")
def _assert_placeholders_clean() -> None:
    """Build-time guard: the frontend's cycling placeholder examples (the
    PLACEHOLDER_QUERIES constant in static/index.html) must pass the content
    filter. Fails startup loudly rather than shipping a blocked example."""
    import re

    try:
        html = (STATIC_HARNESS / "index.html").read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    m = re.search(r"PLACEHOLDER_QUERIES\s*=\s*\[(.*?)\];", html, re.S)
    if not m:
        log.warning("placeholder assertion: PLACEHOLDER_QUERIES not found in static harness")
        return
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    for s in strings:
        cat = content_filter.check(s)
        assert not cat, f"placeholder example fails content filter [{cat}]: {s!r}"
    log.info("placeholder examples pass content filter (%d checked)", len(strings))


@app.on_event("startup")
def _warm_kb() -> None:
    """Preload the CPU KB embedder + index off the request path."""
    import threading

    def warm():
        try:
            from .knowledge import search_knowledge

            search_knowledge("warmup", k=1)
        except Exception as exc:
            log.warning("KB warmup failed: %s", exc)

    threading.Thread(target=warm, daemon=True).start()


def main() -> None:
    import uvicorn

    log.info("starting Scene Sense on %s:%s", config.HOST, config.PORT)
    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
