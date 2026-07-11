"""transcriber.py — faster-whisper with word-level timestamps.

Produces a transcript: a list of segments, each with start/end/text and the
underlying words (start/end/word). Caches to data/transcripts/<key>.json so we
never re-transcribe the same file. Standalone-runnable:

    python -m app.transcriber /path/to/video.mp4
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .device import detect
from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    source_path: str
    language: str
    duration: float
    segments: list[Segment]

    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [asdict(w) for w in s.words],
                }
                for s in self.segments
            ],
        }

    @staticmethod
    def from_dict(d: dict) -> "Transcript":
        segs = [
            Segment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                words=[Word(**w) for w in s.get("words", [])],
            )
            for s in d["segments"]
        ]
        return Transcript(
            source_path=d.get("source_path", ""),
            language=d.get("language", ""),
            duration=d.get("duration", 0.0),
            segments=segs,
        )


@functools.lru_cache(maxsize=1)
def _get_model():
    from faster_whisper import WhisperModel

    prof = detect()
    size = prof["whisper_model"]
    device = prof["device"]
    compute_type = prof["compute_type"] if device == "cuda" else "int8"
    log.info("loading faster-whisper '%s' on %s (%s)", size, device, compute_type)
    return WhisperModel(size, device=device, compute_type=compute_type)


def release_model() -> None:
    """Drop the Whisper model so its VRAM is free for the local LLM."""
    if _get_model.cache_info().currsize == 0:
        return
    _get_model.cache_clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    log.info("released whisper model from memory")


def _cache_key(path: str) -> str:
    p = Path(path)
    stat = p.stat()
    raw = f"{p.resolve()}::{stat.st_size}::{int(stat.st_mtime)}::{detect()['whisper_model']}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _cache_path(path: str) -> Path:
    return config.TRANSCRIPTS_DIR / f"{_cache_key(path)}.json"


def transcribe(path: str, use_cache: bool = True) -> Transcript:
    cache = _cache_path(path)
    if use_cache and cache.exists():
        log.info("transcript cache hit -> %s", cache.name)
        return Transcript.from_dict(json.loads(cache.read_text()))

    model = _get_model()
    log.info("transcribing %s ...", path)
    seg_iter, info = model.transcribe(
        path,
        word_timestamps=True,
        vad_filter=True,
        beam_size=config.WHISPER_BEAM,
    )
    segments: list[Segment] = []
    for s in seg_iter:
        words = [Word(w.start, w.end, w.word) for w in (s.words or [])]
        segments.append(Segment(start=s.start, end=s.end, text=s.text, words=words))
    transcript = Transcript(
        source_path=str(Path(path).resolve()),
        language=info.language,
        duration=float(info.duration or (segments[-1].end if segments else 0.0)),
        segments=segments,
    )
    cache.write_text(json.dumps(transcript.to_dict(), indent=2))
    log.info(
        "transcribed %d segments, lang=%s, %.1fs -> cached %s",
        len(segments), transcript.language, transcript.duration, cache.name,
    )
    return transcript


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video/audio file")
    ap.add_argument("path")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    t = transcribe(args.path, use_cache=not args.no_cache)
    print(f"language={t.language} duration={t.duration:.1f}s segments={len(t.segments)}")
    for s in t.segments[:12]:
        print(f"  [{s.start:7.2f}-{s.end:7.2f}]  {s.text.strip()}")


if __name__ == "__main__":
    _cli()
