"""frame_emotion.py — facial EXPRESSION over time (emotion, never identity).

Samples frames from a video (via ffmpeg, ~1 fps) and runs a facial-expression
detector to build an emotion-over-time timeline: a list of
{t, emotion, confidence}. This is an EMOTION signal only — it never matches a
face to a named person.

The detector (fer / deepface) pulls in heavy TensorFlow deps, so it is imported
lazily and this module degrades gracefully (returns an empty timeline + a note)
when those libraries aren't installed. That keeps the step-6 backend gate free of
the heavy install; enable it for step 8.

    python -m app.frame_emotion /path/video.mp4 --fps 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from . import config
from .device import detect
from .logging_setup import get_logger

log = get_logger(__name__)

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def _detector_available() -> bool:
    try:
        import fer  # noqa: F401

        return True
    except Exception:
        return False


def _sample_frames(video_path: str, fps: float, out_dir: Path) -> list[tuple[float, Path]]:
    """Extract frames at `fps` frames/sec; return [(timestamp, path), ...]."""
    pattern = str(out_dir / "frame_%05d.jpg")
    # Cap width before inference — expression detection doesn't need more.
    width = config.EMOTION_MAX_WIDTH
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps},scale='min({width},iw)':-2", "-q:v", "3", pattern,
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        log.warning("ffmpeg frame sampling failed: %s", cp.stderr[-400:])
        return []
    frames = sorted(out_dir.glob("frame_*.jpg"))
    step = 1.0 / fps
    return [(i * step, p) for i, p in enumerate(frames)]


def _cache_path(video_path: str, fps: float) -> Path:
    p = Path(video_path)
    stat = p.stat()
    raw = f"{p.resolve()}::{stat.st_size}::{int(stat.st_mtime)}::fps={fps}"
    return config.EMOTIONS_DIR / f"{hashlib.sha1(raw.encode()).hexdigest()[:16]}.json"


def build_timeline(video_path: str, fps: float = config.EMOTION_FPS,
                   use_cache: bool = True) -> dict:
    """Return {device, available, note, timeline: [{t, emotion, confidence}]}."""
    prof = detect()
    cache = _cache_path(video_path, fps)
    if use_cache and cache.exists():
        log.info("emotion timeline cache hit -> %s", cache.name)
        return json.loads(cache.read_text())
    if not _detector_available():
        log.info("fer/deepface not installed; emotion timeline disabled")
        return {
            "device": prof["device"],
            "available": False,
            "note": "expression detector (fer) not installed; run `pip install fer`.",
            "timeline": [],
        }

    from fer import FER  # heavy import, done lazily

    detector = FER(mtcnn=False)
    timeline: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        frames = _sample_frames(video_path, fps, Path(tmp))
        log.info("running expression detection on %d frames (%s)", len(frames), prof["device"])
        import cv2  # provided by opencv (fer dependency)

        for t, path in frames:
            img = cv2.imread(str(path))
            if img is None:
                continue
            faces = detector.detect_emotions(img)
            if not faces:
                continue
            # take the most confident face's dominant emotion
            emo_scores = faces[0]["emotions"]
            emotion = max(emo_scores, key=emo_scores.get)
            timeline.append(
                {"t": round(t, 2), "emotion": emotion, "confidence": round(float(emo_scores[emotion]), 3)}
            )
    log.info("emotion timeline: %d points", len(timeline))
    out = {"device": prof["device"], "available": True, "note": "", "timeline": timeline}
    cache.write_text(json.dumps(out))
    return out


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Build an emotion-over-time timeline")
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=config.EMOTION_FPS)
    args = ap.parse_args()
    print(json.dumps(build_timeline(args.video, args.fps), indent=2))


if __name__ == "__main__":
    _cli()
