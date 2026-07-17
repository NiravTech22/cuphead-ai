"""generate_featured_assets.py — materialize the Featured Scenes rail assets.

    python -m app.generate_featured_assets

Cache-only, run manually before a demo. For every entry in featured.CURATED
whose clip exists in CLIPS_DIR it produces, via the same ffmpeg the clipper
uses:

  <clip_id>_thumb.jpg — one representative frame (40% in), 480px wide,
                        desaturated ~20% to sit with the palette
  <clip_id>_loop.mp4  — 3 seconds from the clip's midpoint, muted, ~480p,
                        faststart, for the inline hover preview

then writes data/featured/featured.json. It never downloads or scrapes
anything: a curated clip that is not in the local cache is skipped with a
warning.
"""
from __future__ import annotations

import json
import subprocess
import sys

from . import config
from .featured import CURATED, FEATURED_DIR, MANIFEST_PATH
from .logging_setup import get_logger

log = get_logger(__name__)

LOOP_SECONDS = 3.0
THUMB_WIDTH = 480
DESATURATION = 0.8  # eq=saturation: 1.0 = untouched, 0.8 = ~20% desaturated


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def _probe_duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True).stdout.strip()
    return float(out or 0)


def generate() -> list[dict]:
    FEATURED_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for cur in CURATED:
        clip = config.CLIPS_DIR / f"{cur['clip_id']}.mp4"
        if not clip.is_file():
            log.warning("skip %s: clip not in cache (%s)", cur["clip_id"], clip)
            continue
        duration = _probe_duration(clip)
        thumb = f"{cur['clip_id']}_thumb.jpg"
        loop = f"{cur['clip_id']}_loop.mp4"
        _run(["ffmpeg", "-y", "-ss", f"{duration * 0.4:.2f}", "-i", str(clip),
              "-frames:v", "1",
              "-vf", f"scale={THUMB_WIDTH}:-2,eq=saturation={DESATURATION}",
              "-q:v", "4", str(FEATURED_DIR / thumb)])
        loop_start = max(0.0, (duration - LOOP_SECONDS) / 2)
        _run(["ffmpeg", "-y", "-ss", f"{loop_start:.2f}", "-i", str(clip),
              "-t", f"{LOOP_SECONDS}", "-an",
              "-vf", f"scale=-2:480,eq=saturation={DESATURATION}",
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
              "-movflags", "+faststart", str(FEATURED_DIR / loop)])
        entries.append({**cur, "thumb": thumb, "loop": loop, "duration": round(duration, 2)})
        log.info("featured asset ready: %s (%s)", cur["title"], cur["clip_id"])
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    log.info("featured.json written: %d entr%s", len(entries),
             "y" if len(entries) == 1 else "ies")
    return entries


if __name__ == "__main__":
    n = len(generate())
    sys.exit(0 if n else 1)
