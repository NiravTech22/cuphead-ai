"""featured.py — the "Featured Statements" rail: curated, cache-only, real data.

Curation is the CURATED constant below: each entry points at an existing clip
in CLIPS_DIR by clip_id (clips the pipeline already cut — nothing is ever
downloaded or scraped for this). `python -m app.generate_featured_assets`
materializes a thumbnail + 3s muted preview loop per entry into
data/featured/ and writes featured.json.

`manifest()` is what the API serves: it reads featured.json, drops entries
whose assets are missing, and drops anything the content filter blocks
(title, hook, or query). Missing/empty manifest -> [] -> the rail simply
doesn't render.

CURATED starts empty after the pivot to public statements — the old entries
pointed at cached movie clips (Dark Knight, Matrix, ...) that don't belong in
this domain. Ask a few real verification questions (see README), then add
entries here pointing at the resulting clip_ids and re-run
`python -m app.generate_featured_assets`.
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
# CLIPS_DIR (see clips/<id>.json sidecars for what's available). Empty until
# you've run some real queries and cached real clips — see module docstring.
CURATED: list[dict] = []


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
