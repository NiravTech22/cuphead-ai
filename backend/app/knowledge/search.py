"""knowledge/search.py — semantic + keyword search over the movie KB.

`search_knowledge(query, k)` merges three signals:
  1. exact/near-exact quote matches (quoted phrase found in a quote line) — top rank
  2. FTS5/BM25 keyword hits on quote lines and titles
  3. cosine similarity of the query embedding vs quote/overview embeddings (vibes)

CPU-only, in-memory vector index (a few MB), target <300ms per lookup.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..logging_setup import get_logger
from . import db
from .embedder import embed, from_blob

log = get_logger(__name__)

_QUOTED_RE = re.compile(r"[\'\"‘“]([^\'\"’”]{3,})[\'\"’”]")
_WORD_RE = re.compile(r"[a-z0-9']+")

# candidates at or above this are a "KB hit" the assistant can trust
CONFIDENT = 0.6

_STOP = set(
    "a an the and or but of in on at to for with from by as is was are were be been "
    "being am do does did done have has had i you he she it we they me him her them "
    "my your his its our their this that these those there here what which who whom "
    "when why how not no so if then than too very just about into over under again "
    "say says said scene part moment clip show me find play where guy girl man woman "
    "someone somebody something like get gets got go goes went".split())


def _norm(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _content(text: str) -> set[str]:
    """Stopword-stripped, lightly stemmed content tokens."""
    return {w.rstrip("s") or w for w in _WORD_RE.findall(text.lower()) if w not in _STOP}


@dataclass
class _Index:
    mtime: float = 0.0
    title_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    title_vecs: np.ndarray = field(default_factory=lambda: np.empty((0, 384), dtype=np.float32))
    quote_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    quote_vecs: np.ndarray = field(default_factory=lambda: np.empty((0, 384), dtype=np.float32))
    titles: dict = field(default_factory=dict)   # id -> row dict
    quotes: dict = field(default_factory=dict)   # id -> row dict


_index = _Index()
_lock = threading.Lock()


def _load_index() -> _Index:
    """(Re)load vectors + rows into memory when the DB file changes."""
    global _index
    try:
        mtime = db.DB_PATH.stat().st_mtime
    except FileNotFoundError:
        return _index
    if _index.mtime == mtime:
        return _index
    with _lock:
        if _index.mtime == mtime:
            return _index
        t0 = time.perf_counter()
        conn = db.connect()
        try:
            idx = _Index(mtime=mtime)
            idx.titles = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM titles")}
            idx.quotes = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM quotes")}
            for table, ids_attr, vecs_attr in (("titles", "title_ids", "title_vecs"),
                                               ("quotes", "quote_ids", "quote_vecs")):
                rows = conn.execute(
                    "SELECT row_id, vector FROM embeddings WHERE source_table=?", (table,)
                ).fetchall()
                if rows:
                    setattr(idx, ids_attr, np.array([r["row_id"] for r in rows], dtype=np.int64))
                    setattr(idx, vecs_attr, np.vstack([from_blob(r["vector"]) for r in rows]))
            _index = idx
            log.info("KB index loaded: %d titles, %d quotes (%.0fms)",
                     len(idx.titles), len(idx.quotes), 1000 * (time.perf_counter() - t0))
        finally:
            conn.close()
    return _index


def _fts(conn, table: str, query: str, limit: int = 30) -> set[int]:
    """FTS5 candidate rowids for the query's content tokens (phrase, then OR)."""
    all_tokens = _WORD_RE.findall(query.lower())
    content = [t for t in all_tokens if t not in _STOP]
    out: set[int] = set()
    for match in ([f'"{" ".join(all_tokens)}"'] if len(all_tokens) > 1 else []) + \
                 ([" OR ".join(content)] if content else []):
        try:
            rows = conn.execute(
                f"SELECT rowid FROM {table} WHERE {table} MATCH ? "
                f"ORDER BY bm25({table}) LIMIT ?", (match, limit)).fetchall()
        except Exception:
            continue
        out.update(r["rowid"] for r in rows)
        if rows and match.startswith('"'):
            break  # full-phrase hit is enough
    return out


def _kw_score(query_tokens: set[str], text: str, symmetric: bool) -> float:
    """Keyword-overlap score in [0,1]. Symmetric (quotes) also rewards the text
    side being covered; asymmetric (long overviews) uses query coverage only."""
    ttoks = _content(text)
    inter = query_tokens & ttoks
    if not inter or not query_tokens:
        return 0.0
    cov_q = len(inter) / len(query_tokens)
    if not symmetric:
        return cov_q
    cov_t = len(inter) / max(1, len(ttoks))
    return (cov_q * cov_t) ** 0.5


def search_knowledge(query: str, k: int = 5) -> list[dict]:
    """Return top-k candidates: {kind, title, year, type, genres, confidence,
    matched_quote, character, timestamp, title_id}."""
    t0 = time.perf_counter()
    idx = _load_index()
    if not idx.titles:
        log.info("KB miss: knowledge base is empty")
        return []

    qv = embed([query])[0]
    quoted = _QUOTED_RE.search(query)
    phrase = _norm(quoted.group(1)) if quoted else None
    q_tokens = _content(quoted.group(1) if quoted else query)

    conn = db.connect()
    try:
        fts_quotes = _fts(conn, "quotes_fts", quoted.group(1) if quoted else query)
        fts_titles = _fts(conn, "titles_fts", query, limit=15)
    finally:
        conn.close()

    # ── score quotes ──────────────────────────────────────────────
    best_by_title: dict[int, dict] = {}

    def consider(title_id: int, conf: float, matched_quote: Optional[str] = None,
                 character: str = "", timestamp: Optional[float] = None) -> None:
        cur = best_by_title.get(title_id)
        if cur is None or conf > cur["confidence"]:
            t = idx.titles.get(title_id)
            if not t:
                return
            best_by_title[title_id] = {
                "kind": "quote" if matched_quote else "title",
                "title": t["title"], "year": t["year"], "type": t["type"],
                "genres": t["genres"], "confidence": round(float(conf), 3),
                "matched_quote": matched_quote, "character": character,
                "timestamp": timestamp, "title_id": title_id,
            }

    if len(idx.quote_ids):
        sims = idx.quote_vecs @ qv
        order = set(int(i) for i in np.argsort(-sims)[: max(20, k * 4)])
        pos = {int(qid): i for i, qid in enumerate(idx.quote_ids)}
        cand_ids = {int(idx.quote_ids[i]) for i in order} | set(fts_quotes)
        for qid in cand_ids:
            q = idx.quotes.get(qid)
            if not q:
                continue
            conf = float(sims[pos[qid]]) if qid in pos else 0.0
            qnorm = _norm(q["quote_text"])
            if phrase and (phrase in qnorm or qnorm in phrase):
                conf = max(conf, 0.97)          # exact quote beats everything
            elif qid in fts_quotes:
                kw = _kw_score(q_tokens, q["quote_text"], symmetric=True)
                conf = max(conf, min(0.9, 0.45 + 0.5 * kw))
            if conf < 0.3:
                continue
            consider(q["title_id"], conf, q["quote_text"], q["character"] or "",
                     q["approx_timestamp"])
        # exact-phrase quotes that both rankings missed
        if phrase:
            for qid, q in idx.quotes.items():
                if phrase in _norm(q["quote_text"]):
                    consider(q["title_id"], 0.97, q["quote_text"], q["character"] or "",
                             q["approx_timestamp"])

    # ── score titles (vibe / description matches) ─────────────────
    if len(idx.title_ids):
        sims = idx.title_vecs @ qv
        pos_t = {int(tid): i for i, tid in enumerate(idx.title_ids)}
        cand_t = {int(idx.title_ids[i]) for i in np.argsort(-sims)[: max(10, k * 2)]} \
            | set(fts_titles)
        vibe_tokens = _content(query)
        for tid in cand_t:
            t = idx.titles.get(tid)
            if not t:
                continue
            conf = (float(sims[pos_t[tid]]) if tid in pos_t else 0.0) * 0.9
            if tid in fts_titles:               # keyword coverage of the description
                kw = _kw_score(vibe_tokens, f'{t["title"]} {t["overview"]}', symmetric=False)
                conf = max(conf, min(0.85, 0.35 + 0.5 * kw))
            if conf >= 0.3:
                consider(tid, conf)

    out = sorted(best_by_title.values(), key=lambda c: -c["confidence"])[:k]
    ms = 1000 * (time.perf_counter() - t0)
    if out and out[0]["confidence"] >= CONFIDENT:
        log.info("KB hit (%.0fms): %r -> %s (%s) conf=%.2f quote=%r", ms, query[:60],
                 out[0]["title"], out[0]["year"], out[0]["confidence"],
                 (out[0]["matched_quote"] or "")[:50])
    else:
        log.info("KB miss (%.0fms): %r best=%s conf=%.2f", ms, query[:60],
                 out[0]["title"] if out else None, out[0]["confidence"] if out else 0.0)
    return out


def title_info(title: str) -> Optional[dict]:
    """Exact-ish lookup of a title row (for preference tracking)."""
    idx = _load_index()
    tnorm = _norm(title)
    best = None
    for t in idx.titles.values():
        cand = _norm(t["title"])
        if cand and (cand == tnorm or cand in tnorm or tnorm in cand):
            if best is None or len(cand) > len(_norm(best["title"])):
                best = t
    return best
