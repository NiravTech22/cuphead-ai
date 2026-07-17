"""session.py — in-memory session state (no DB; personal-use tool).

A session holds the currently loaded/fetched video: its file path, metadata,
transcript, and (optionally) emotion timeline. The assistant reads/writes this
as it identifies sources, fetches videos, and locates moments.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from .transcriber import Transcript


@dataclass
class LoadedVideo:
    path: str
    title: str = ""
    channel: str = ""
    webpage_url: str = ""
    duration: float = 0.0
    transcript: Optional[Transcript] = None
    emotion_timeline: list[dict] = field(default_factory=list)
    video_id: str = ""
    # pinned=True only for videos the user loaded explicitly (load_url/upload).
    # Auto-fetched videos are cleared at the start of each new question so a
    # previous query's source/transcript can never leak into the next one.
    pinned: bool = False

    def summary(self) -> dict:
        return {
            "title": self.title,
            "video_id": self.video_id,
            "channel": self.channel,
            "duration": self.duration,
            "webpage_url": self.webpage_url,
            "has_transcript": self.transcript is not None,
            "segments": len(self.transcript.segments) if self.transcript else 0,
            "has_emotion": bool(self.emotion_timeline),
        }


@dataclass
class Session:
    session_id: str
    video: Optional[LoadedVideo] = None
    # delivered scenes this session: {title, genres, tone, year, ts} — the
    # "session vibe" layer of the preference engine
    history: list[dict] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id=session_id)
            return self._sessions[session_id]


STORE = SessionStore()
