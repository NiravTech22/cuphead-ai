"""clipper.py — cut [start-pad, end+pad] into a standalone playable mp4.

Stream-copy when possible (fast, lossless); re-encode only when a stream-copy
clip won't start on a keyframe (which makes players show a black/frozen lead-in).
Writes a sidecar .json next to each clip. Output lands in CLIPS_DIR, served
statically by the API so the frontend can embed it inline.

    python -m app.clipper /path/video.mp4 12.0 20.0 --quote "why so serious"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from . import config
from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Clip:
    clip_id: str
    filename: str
    url: str            # served path, e.g. /clips/<file>.mp4
    start: float
    end: float
    duration: float
    source_path: str
    quote: str = ""
    dominant_emotion: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    log.debug("ffmpeg: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True)


def _has_stream(path: str, kind: str) -> bool:
    """kind = 'v' (video) or 'a' (audio)."""
    cp = _run(
        ["ffprobe", "-v", "error", "-select_streams", kind,
         "-show_entries", "stream=index", "-of", "csv=p=0", path]
    )
    return bool(cp.stdout.strip())


def make_clip(
    source_path: str,
    start: float,
    end: float,
    quote: str = "",
    dominant_emotion: Optional[str] = None,
    pad: Optional[float] = None,
    reencode: Optional[bool] = None,
) -> Clip:
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"source not found: {source_path}")

    pad = config.CLIP_PAD_SECONDS if pad is None else pad
    start = max(0.0, start - pad)
    end = end + pad
    # too short feels abrupt: grow symmetrically to the minimum
    if end - start < config.CLIP_MIN_SECONDS:
        grow = (config.CLIP_MIN_SECONDS - (end - start)) / 2
        start = max(0.0, start - grow)
        end = end + grow
    # enforce a sane max length
    if end - start > config.CLIP_MAX_SECONDS:
        end = start + config.CLIP_MAX_SECONDS
    duration = max(0.5, end - start)

    clip_id = uuid.uuid4().hex[:12]
    filename = f"{clip_id}.mp4"
    out = config.CLIPS_DIR / filename

    # Default: re-encode. Stream-copy is faster but seeks to the nearest keyframe,
    # which for arbitrary in/out points often produces a frozen/black lead-in.
    # For short scene clips, a clean re-encode is worth the few seconds.
    do_reencode = True if reencode is None else reencode
    has_audio = _has_stream(source_path, "a")

    if do_reencode:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", source_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
        cmd.append(str(out))
    else:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", source_path,
            "-t", f"{duration:.3f}", "-c", "copy",
            "-movflags", "+faststart", str(out),
        ]

    cp = _run(cmd)
    if cp.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        # Fall back to a re-encode if stream-copy failed.
        if not do_reencode:
            log.warning("stream-copy failed, retrying with re-encode")
            return make_clip(source_path, start + pad, end - pad, quote,
                             dominant_emotion, pad=pad, reencode=True)
        raise RuntimeError(f"ffmpeg failed:\n{cp.stderr[-800:]}")

    clip = Clip(
        clip_id=clip_id,
        filename=filename,
        url=f"/clips/{filename}",
        start=round(start, 3),
        end=round(end, 3),
        duration=round(duration, 3),
        source_path=str(src.resolve()),
        quote=quote,
        dominant_emotion=dominant_emotion,
    )
    (config.CLIPS_DIR / f"{clip_id}.json").write_text(json.dumps(clip.to_dict(), indent=2))
    log.info("clip -> %s (%.1fs)", clip.url, duration)
    return clip


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Cut a clip from a video")
    ap.add_argument("source")
    ap.add_argument("start", type=float)
    ap.add_argument("end", type=float)
    ap.add_argument("--quote", default="")
    ap.add_argument("--copy", action="store_true", help="stream-copy instead of re-encode")
    args = ap.parse_args()
    clip = make_clip(args.source, args.start, args.end, quote=args.quote,
                     reencode=not args.copy)
    print(json.dumps(clip.to_dict(), indent=2))


if __name__ == "__main__":
    _cli()
