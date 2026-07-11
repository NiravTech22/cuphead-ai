"""websearch.py — local web search (DuckDuckGo, no API key).

Replaces the Anthropic server-side web_search tool now that the default LLM is
local. Returns lightweight results the model can cite: {title, url, snippet}.
Degrades gracefully (empty list + note) if the search backend is unreachable.
"""
from __future__ import annotations

from .logging_setup import get_logger

log = get_logger(__name__)


def search(query: str, max_results: int = 5) -> list[dict]:
    """Return [{title, url, snippet}] for a query, or [] on failure."""
    try:
        from ddgs import DDGS
    except Exception:  # package name fallback
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except Exception as exc:
            log.warning("no search backend available (%s)", exc)
            return []

    out: list[dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                out.append(
                    {
                        "title": r.get("title", "") or "",
                        "url": r.get("href") or r.get("url") or "",
                        "snippet": (r.get("body") or r.get("snippet") or "")[:400],
                    }
                )
    except Exception as exc:
        log.warning("web search failed for %r: %s", query, exc)
        return []
    log.info("web search %r -> %d results", query[:60], len(out))
    return out


def format_results(results: list[dict]) -> str:
    """Render results as a compact numbered list for an LLM prompt."""
    if not results:
        return "(no web results)"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n    {r['url']}\n    {r['snippet']}")
    return "\n".join(lines)
