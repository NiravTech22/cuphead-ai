"""downloader.py — yt-dlp wrapper.

Search YouTube (`ytsearch:`), download a chosen video, and capture metadata
(title, channel, description, duration). Standalone-runnable:

    python -m app.downloader "ytsearch1:big buck bunny" --download
    python -m app.downloader "https://youtu.be/..." --download
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from yt_dlp import YoutubeDL

from . import config
from .logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class VideoInfo:
    id: str
    title: str
    channel: str
    description: str
    duration: float          # seconds
    webpage_url: str
    thumbnail: str = ""
    local_path: Optional[str] = None  # set once downloaded

    def short(self) -> dict:
        d = asdict(self)
        d["description"] = (self.description or "")[:500]
        return d


def _base_opts(quiet: bool = True) -> dict:
    return {
        "quiet": quiet,
        "no_warnings": quiet,
        "noplaylist": True,
        "extract_flat": False,
        # Cap resolution — 720p is plenty for clip playback and much smaller.
        # Fall through progressively so a missing exact combo never hard-fails.
        "format": "best[height<=720][ext=mp4]/best[height<=720]/"
                  "bestvideo[height<=720]+bestaudio/18/best",
        "merge_output_format": "mp4",
        "ignoreerrors": False,
        "outtmpl": str(config.DOWNLOADS_DIR / "%(id)s.%(ext)s"),
        "restrictfilenames": True,
    }


def _to_info(entry: dict) -> VideoInfo:
    vid = entry.get("id", "") or ""
    # In flat-search results webpage_url is often absent; fall back to `url` or the
    # canonical watch URL built from the id so download() always has a target.
    url = (
        entry.get("webpage_url")
        or entry.get("original_url")
        or entry.get("url")
        or (f"https://www.youtube.com/watch?v={vid}" if vid else "")
    )
    return VideoInfo(
        id=vid,
        title=entry.get("title", "") or "",
        channel=entry.get("channel") or entry.get("uploader") or "",
        description=entry.get("description") or "",
        duration=float(entry.get("duration") or 0.0),
        webpage_url=url,
        thumbnail=entry.get("thumbnail") or "",
    )


def search(query: str, limit: int = 5) -> list[VideoInfo]:
    """Search YouTube. `query` may be a plain string or a full `ytsearchN:` term."""
    if not query.startswith("ytsearch") and not query.startswith("http"):
        query = f"ytsearch{limit}:{query}"
    log.info("searching: %s", query)
    opts = _base_opts()
    opts["extract_flat"] = "in_playlist"
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(query, download=False)
    entries = info.get("entries", [info]) if isinstance(info, dict) else []
    results = [_to_info(e) for e in entries if e]
    log.info("found %d results", len(results))
    return results


def get_metadata(url: str) -> VideoInfo:
    """Fetch full metadata (incl. description) for a single URL without downloading."""
    log.info("metadata: %s", url)
    with YoutubeDL(_base_opts()) as ydl:
        entry = ydl.extract_info(url, download=False)
    return _to_info(entry)


_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def cached_download(video_id: str) -> Optional[VideoInfo]:
    """Return VideoInfo for an already-downloaded video (file + sidecar), else None."""
    if not video_id:
        return None
    for f in config.DOWNLOADS_DIR.glob(f"{video_id}.*"):
        if f.suffix == ".json" or not f.is_file():
            continue
        sidecar = f.with_suffix(".info.json")
        if not sidecar.exists():
            continue
        try:
            d = json.loads(sidecar.read_text())
        except Exception:
            continue
        info = VideoInfo(
            id=d.get("id", video_id), title=d.get("title", ""),
            channel=d.get("channel", ""), description=d.get("description", ""),
            duration=float(d.get("duration") or 0.0),
            webpage_url=d.get("webpage_url", ""), thumbnail=d.get("thumbnail", ""),
            local_path=str(f),
        )
        return info
    return None


def download(url: str) -> VideoInfo:
    """Download a single video to DOWNLOADS_DIR and return VideoInfo with local_path."""
    m = _YT_ID_RE.search(url)
    if m:
        hit = cached_download(m.group(1))
        if hit:
            log.info("download cache hit: %s -> %s", url, hit.local_path)
            return hit
    log.info("downloading: %s", url)
    with YoutubeDL(_base_opts(quiet=True)) as ydl:
        entry = ydl.extract_info(url, download=True)
        # requested_downloads gives the true merged output path
        path = None
        if entry.get("requested_downloads"):
            path = entry["requested_downloads"][0].get("filepath")
        if not path:
            path = ydl.prepare_filename(entry)
            # merge may have changed ext to mp4
            if not Path(path).exists():
                path = str(Path(path).with_suffix(".mp4"))
    info = _to_info(entry)
    info.local_path = str(path) if path and Path(path).exists() else None
    log.info("downloaded -> %s", info.local_path)
    _write_sidecar(info)
    return info


def _write_sidecar(info: VideoInfo) -> None:
    if not info.local_path:
        return
    sidecar = Path(info.local_path).with_suffix(".info.json")
    sidecar.write_text(json.dumps(asdict(info), indent=2))


def _cli() -> None:
    ap = argparse.ArgumentParser(description="yt-dlp search/download")
    ap.add_argument("query", help="ytsearchN:term, plain text, or a URL")
    ap.add_argument("--download", action="store_true", help="download top result")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    if args.query.startswith("http"):
        results = [get_metadata(args.query)]
    else:
        results = search(args.query, args.limit)

    for i, r in enumerate(results):
        print(f"[{i}] {r.title}  |  {r.channel}  |  {r.duration:.0f}s")
        print(f"     {r.webpage_url}")

    if args.download and results:
        info = download(results[0].webpage_url)
        print(json.dumps(info.short(), indent=2))


if __name__ == "__main__":
    _cli()
