"""knowledge/ingest_events.py — ingest a LOCAL JSON file of public-statement events into the KB.

Point it at structured data you've assembled yourself (from official government
transcript archives, presidential libraries, C-SPAN, the Congressional Record,
or your own research) — the same "bring your own data" contract as
ingest_subtitles.py, just for event/quote metadata instead of raw .srt files.

This tool does NOT fetch anything from the web — bring your own JSON.

Expected shape (a list of events):

  [
    {
      "title": "Obama Election Night Victory Speech",
      "year": 2008,
      "type": "speech",
      "genres": "Politics",
      "overview": "Barack Obama addresses the nation after winning the 2008 election.",
      "popularity": 80,
      "quotes": [
        {"quote": "Change has come to America.", "speaker": "Barack Obama", "timestamp": null}
      ]
    },
    ...
  ]

Only "title" and "quotes[].quote" are required; everything else defaults sanely.
Re-running upserts by (title, year), so it's safe to run repeatedly as you grow
the file.

  python -m app.knowledge.ingest_events /path/to/events.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..logging_setup import get_logger
from . import db
from .embedder import embed, to_blob

log = get_logger(__name__)


def ingest(path: Path) -> dict:
    events = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise SystemExit(f"{path}: expected a JSON list of events")

    conn = db.connect()
    title_batch: list[tuple[int, str]] = []
    quote_batch: list[tuple[int, str]] = []
    n_events = n_quotes = 0
    try:
        for ev in events:
            title = (ev.get("title") or "").strip()
            if not title:
                log.warning("skipping event with no title: %r", ev)
                continue
            year = ev.get("year")
            type_ = ev.get("type") or "speech"
            genres = ev.get("genres") or ""
            overview = ev.get("overview") or ""
            popularity = float(ev.get("popularity") or 0)
            tid = db.upsert_title(conn, title=title, year=year, type_=type_,
                                  genres=genres, overview=overview, popularity=popularity)
            n_events += 1
            title_batch.append((tid, f"{title} ({year}). {genres}. {overview}"))

            conn.execute("DELETE FROM quotes WHERE title_id=?", (tid,))
            for q in ev.get("quotes") or []:
                quote_text = (q.get("quote") or "").strip()
                if not quote_text:
                    continue
                qid = db.add_quote(conn, tid, quote_text, q.get("speaker") or "",
                                   q.get("timestamp"))
                quote_batch.append((qid, quote_text))
                n_quotes += 1

        if title_batch:
            vecs = embed([t for _, t in title_batch])
            db.put_embeddings(conn, "titles", [(tid, to_blob(v)) for (tid, _), v in zip(title_batch, vecs)])
        if quote_batch:
            vecs = embed([q for _, q in quote_batch])
            db.put_embeddings(conn, "quotes", [(qid, to_blob(v)) for (qid, _), v in zip(quote_batch, vecs)])
        conn.commit()
        stats = db.stats(conn)
        log.info("event ingest done: %d event(s), %d quote(s) from %s; KB now %s",
                 n_events, n_quotes, path.name, stats)
        return stats
    finally:
        conn.close()


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Ingest a local JSON file of events into the statements KB")
    ap.add_argument("path", help="path to a JSON file of events (see module docstring for shape)")
    args = ap.parse_args()
    ingest(Path(args.path))


if __name__ == "__main__":
    _cli()
