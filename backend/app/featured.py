"""featured.py — the "Featured Scenes" rail: curated, cache-only, real data.

Curation is the CURATED constant below: each entry points at an existing clip
in CLIPS_DIR by clip_id (clips the pipeline already cut — nothing is ever
downloaded or scraped for this). `python -m app.generate_featured_assets`
materializes a thumbnail + 3s muted preview loop per entry into
data/featured/ and writes featured.json.

`manifest()` is what the API serves: it reads featured.json, drops entries
whose assets are missing, and drops anything the content filter blocks
(title, hook, or query). Missing/empty manifest -> [] -> the rail simply
doesn't render.
"""
from __future__ import annotations

import json

from . import config
from . import filter as content_filter
from .logging_setup import get_logger

log = get_logger(__name__)

FEATURED_DIR = config.DATA_DIR / "featured"
MANIFEST_PATH = FEATURED_DIR / "featured.json"

# One editable constant: clip_id -> presentation. clip_id must exist in
# CLIPS_DIR (see clips/<id>.json sidecars for what's available).
CURATED: list[dict] = [
    {
        "clip_id": "462209e8fae8",
        "title": "The Dark Knight",
        "source": "Why so serious? | The Dark Knight",
        "hook": "The story behind the scars.",
        "query": "show me the scene in The Dark Knight where the Joker says 'why so serious'",
    },
    {
        "clip_id": "3353d976f7d3",
        "title": "Breaking Bad",
        "source": "I Am the One Who Knocks (S4E6)",
        "hook": "Walt stops apologizing.",
        "query": "show me the 'I am the one who knocks' scene from Breaking Bad",
    },
    {
        "clip_id": "30c4b8e0b6c2",
        "title": "Interstellar",
        "source": "Cooper watches 23 years of messages",
        "hook": "Twenty-three years arrive at once.",
        "query": "show me the scene in Interstellar where Cooper watches 23 years of messages",
    },
    {
        "clip_id": "18821fba7034",
        "title": "The Matrix",
        "source": 'Neo: "guns, lots of guns"',
        "hook": "The armory arrives on rails.",
        "query": "show me the scene in The Matrix where Neo says 'guns, lots of guns'",
    },
    {
        "clip_id": "cc729dd85bc3",
        "title": "Blade Runner 2049",
        "source": "You look lonely",
        "hook": "A hologram sees right through K.",
        "query": "show me the scene in Blade Runner 2049 where she says 'you look lonely'",
    },
    {
        "clip_id": "4278687a6ae7",
        "title": "The Matrix",
        "source": "Trinity escapes the Agents",
        "hook": "Your men are already dead.",
        "query": "show me the scene in The Matrix where Trinity says 'your men are already dead'",
    },
]


def manifest() -> list[dict]:
    """Entries that are safe to render: assets on disk + content filter pass."""
    try:
        entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    out = []
    for e in entries:
        thumb = FEATURED_DIR / e.get("thumb", "")
        loop = FEATURED_DIR / e.get("loop", "")
        if not (thumb.is_file() and loop.is_file()):
            log.warning("featured: assets missing for %s — skipped", e.get("clip_id"))
            continue
        cat = (content_filter.check(e.get("title")) or content_filter.check(e.get("hook"))
               or content_filter.check(e.get("query")))
        if cat:
            log.info("featured: dropped %s (filter category=%s)", e.get("clip_id"), cat)
            continue
        out.append({
            "clip_id": e["clip_id"], "title": e["title"], "source": e.get("source", ""),
            "hook": e["hook"], "query": e["query"],
            "thumb_url": f"/featured/{e['thumb']}", "loop_url": f"/featured/{e['loop']}",
        })
    return out
