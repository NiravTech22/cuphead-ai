"""Runtime configuration & paths. Loads .env if present."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

# --- Paths -----------------------------------------------------------------
DATA_DIR = Path(os.getenv("SCENE_SENSE_DATA_DIR", BACKEND_DIR / "data"))
CLIPS_DIR = Path(os.getenv("SCENE_SENSE_CLIPS_DIR", BACKEND_DIR / "clips"))
UPLOADS_DIR = Path(os.getenv("SCENE_SENSE_UPLOADS_DIR", BACKEND_DIR / "uploads"))
DOWNLOADS_DIR = DATA_DIR / "downloads"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
for _p in (DATA_DIR, CLIPS_DIR, UPLOADS_DIR, DOWNLOADS_DIR, TRANSCRIPTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --- Anthropic -------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Opus 4.8 is the most capable current model; good for the orchestration spine.
ASSISTANT_MODEL = os.getenv("SCENE_SENSE_MODEL", "claude-opus-4-8")
# A cheaper model for the moment-ranker fallback.
RANKER_MODEL = os.getenv("SCENE_SENSE_RANKER_MODEL", "claude-haiku-4-5-20251001")

# --- Speed / models ----------------------------------------------------------
# Whisper model size (tiny|base|small|...). Empty = auto-pick per device profile.
WHISPER_MODEL = os.getenv("SCENE_SENSE_WHISPER_MODEL", "")
WHISPER_BEAM = int(os.getenv("SCENE_SENSE_WHISPER_BEAM", "1"))
# Emotion sampling: one frame every 1/fps seconds (0.4 = every 2.5s) at capped width.
EMOTION_FPS = float(os.getenv("SCENE_SENSE_EMOTION_FPS", "0.4"))
EMOTION_MAX_WIDTH = int(os.getenv("SCENE_SENSE_EMOTION_MAX_WIDTH", "480"))
# Keep the local LLM resident in (V)RAM between calls.
OLLAMA_KEEP_ALIVE = os.getenv("SCENE_SENSE_OLLAMA_KEEP_ALIVE", "30m")
EMOTIONS_DIR = DATA_DIR / "emotions"
EMOTIONS_DIR.mkdir(parents=True, exist_ok=True)

# --- Clipping --------------------------------------------------------------
CLIP_PAD_SECONDS = float(os.getenv("SCENE_SENSE_CLIP_PAD", "1.5"))
CLIP_MAX_SECONDS = float(os.getenv("SCENE_SENSE_CLIP_MAX", "45"))
# A pinned one-liner can be ~2s; expand symmetrically so clips don't feel abrupt.
CLIP_MIN_SECONDS = float(os.getenv("SCENE_SENSE_CLIP_MIN", "6"))

# --- Server ----------------------------------------------------------------
HOST = os.getenv("SCENE_SENSE_HOST", "127.0.0.1")
PORT = int(os.getenv("SCENE_SENSE_PORT", "8000"))


def has_api_key() -> bool:
    return bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.strip())
