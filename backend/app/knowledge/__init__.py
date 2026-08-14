"""knowledge — local public-statements knowledge base (RAG retrieval layer, CPU-only).

Public surface:
    search_knowledge(query, k=5)  -> ranked candidates (quotes above vibes)
    title_info(title)             -> KB row for a title (genres/year/type)
    stats()                       -> {titles, quotes, embeddings}
"""
from .db import stats
from .search import CONFIDENT, search_knowledge, title_info

__all__ = ["search_knowledge", "title_info", "stats", "CONFIDENT"]
