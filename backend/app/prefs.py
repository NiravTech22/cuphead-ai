"""prefs.py — preference engine: session vibe, persistent profile, next-clip prediction.

Three layers:
  A. Session vibe — per-session list of delivered scenes (genres/tones/eras),
     summarized on demand. Works from the first request; no history needed.
  B. Persistent profile — data/preferences.json accumulates delivery events
     across ALL sessions with exponential recency decay (half-life ~7 days),
     so last week's taste outweighs last month's.
  C. Prediction — KB nearest-neighbors to the just-delivered scene, boosted by
     the profile + session vibe, turned into 2-3 tappable suggestions.
     Text-only: nothing is downloaded or pre-processed until a chip is tapped.
"""
from __future__ import annotations

import json
import threading
import time
from collections import Counter
from typing import Optional

from . import config
from .knowledge import search_knowledge, title_info
from .logging_setup import get_logger
from .session import Session

log = get_logger(__name__)

PREFS_PATH = config.DATA_DIR / "preferences.json"
HALF_LIFE_DAYS = 7.0
MAX_EVENTS = 500
_lock = threading.Lock()


def _load() -> dict:
    try:
        return json.loads(PREFS_PATH.read_text())
    except Exception:
        return {"events": []}


def _decade(year: Optional[int]) -> Optional[str]:
    return f"{(year // 10) * 10}s" if year else None


def _genre_list(genres: str) -> list[str]:
    return [g.strip() for g in (genres or "").split(",") if g.strip()]


# ── A + B: record a delivered clip ───────────────────────────────────────────

def record_delivery(session: Session, *, title: str, quote: str = "",
                    tone: Optional[str] = None, year: Optional[int] = None,
                    genres: str = "") -> dict:
    """Track a delivered scene in the session vibe + the persistent profile."""
    kb_row = title_info(title)
    if kb_row:
        year = year or kb_row.get("year")
        genres = genres or kb_row.get("genres", "")
        title = kb_row.get("title") or title
    event = {
        "t": time.time(), "title": title, "quote": quote[:120],
        "genres": _genre_list(genres), "tone": tone, "decade": _decade(year),
    }
    session.history.append(event)
    with _lock:
        data = _load()
        data["events"] = (data.get("events", []) + [event])[-MAX_EVENTS:]
        PREFS_PATH.write_text(json.dumps(data, indent=1))
    log.info("prefs: recorded %r (genres=%s tone=%s %s); session vibe=%s",
             title, event["genres"], tone, event["decade"], session_vibe(session))
    return event


def _weighted_counts(events: list[dict], now: Optional[float] = None) -> dict[str, Counter]:
    now = now or time.time()
    genres: Counter = Counter()
    tones: Counter = Counter()
    decades: Counter = Counter()
    titles: Counter = Counter()
    for e in events:
        age_days = max(0.0, (now - e.get("t", now)) / 86400)
        w = 0.5 ** (age_days / HALF_LIFE_DAYS)  # recency decay
        for g in e.get("genres", []):
            genres[g] += w
        if e.get("tone"):
            tones[e["tone"]] += w
        if e.get("decade"):
            decades[e["decade"]] += w
        if e.get("title"):
            titles[e["title"]] += w
    return {"genres": genres, "tones": tones, "decades": decades, "titles": titles}


def get_user_profile() -> dict:
    """Top genres/tones/eras across all sessions, recency-weighted."""
    data = _load()
    c = _weighted_counts(data.get("events", []))
    return {
        "top_genres": [g for g, _ in c["genres"].most_common(5)],
        "top_tones": [t for t, _ in c["tones"].most_common(3)],
        "top_eras": [d for d, _ in c["decades"].most_common(3)],
        "top_titles": [t for t, _ in c["titles"].most_common(5)],
        "events": len(data.get("events", [])),
    }


def session_vibe(session: Session) -> dict:
    """Dominant genres/tones/era of THIS session's requests."""
    c = _weighted_counts(session.history)
    return {
        "genres": [g for g, _ in c["genres"].most_common(3)],
        "tones": [t for t, _ in c["tones"].most_common(2)],
        "era": next(iter([d for d, _ in c["decades"].most_common(1)]), None),
        "scenes": len(session.history),
    }


# ── C: prediction ─────────────────────────────────────────────────────────────

def _cached_titles() -> list[str]:
    """Lowercased titles of videos whose downloads are already on disk."""
    out = []
    for sc in config.DOWNLOADS_DIR.glob("*.info.json"):
        try:
            out.append(json.loads(sc.read_text()).get("title", "").lower())
        except Exception:
            continue
    return out


_TITLE_STOP = {"the", "a", "an", "of", "and", "part"}


def _is_cached(title: str, cached: list[str]) -> bool:
    words = [w for w in title.lower().split() if len(w) > 2 and w not in _TITLE_STOP]
    return bool(words) and any(all(w in c for w in words) for c in cached)


def suggest_next(session: Session, *, title: str, quote: str = "",
                 tone: Optional[str] = None, k: int = 3) -> list[dict]:
    """KB nearest-neighbors to the delivered scene, boosted by taste. Text-only."""
    profile = get_user_profile()
    vibe = session_vibe(session)
    kb_row = title_info(title)
    genres = _genre_list(kb_row.get("genres", "")) if kb_row else []

    # neighbor search seeded by the scene's own text + its vibe words
    seed = " ".join(filter(None, [quote, title, ", ".join(genres), tone or ""]))
    cands = search_knowledge(seed, k=12)

    delivered_norm = title.lower()
    scored: list[tuple[float, dict]] = []
    seen_titles: set[str] = set()
    for c in cands:
        cnorm = (c["title"] or "").lower()
        if not cnorm or cnorm in delivered_norm or delivered_norm in cnorm:
            continue  # never suggest the scene we just delivered
        if cnorm in seen_titles:
            continue
        seen_titles.add(cnorm)
        score = c["confidence"]
        cgenres = _genre_list(c.get("genres", ""))
        score += 0.10 * len(set(cgenres) & set(profile["top_genres"]))   # long-term taste
        score += 0.08 * len(set(cgenres) & set(vibe["genres"]))          # session vibe
        if profile["top_eras"] and c.get("year") and _decade(c["year"]) in profile["top_eras"]:
            score += 0.04
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])

    cached = _cached_titles()
    out = []
    for score, c in scored[:k]:
        if c.get("matched_quote"):
            q = c["matched_quote"].strip().rstrip(".")
            query = f"show me the scene in {c['title']} where they say '{q[:70]}'"
            label = f"“{q[:44]}” — {c['title']}"
        else:
            query = f"show me an iconic scene from {c['title']} ({c['year']})"
            label = f"Something from {c['title']} ({c['year']})"
        out.append({
            "label": label, "query": query, "title": c["title"], "year": c["year"],
            "cached": _is_cached(c["title"], cached), "score": round(score, 3),
        })
    log.info("prefs: %d suggestions after %r (vibe=%s)", len(out), title, vibe["genres"])
    return out
