"""knowledge/ingest_tmdb.py — pull popular/top-rated movie + TV metadata from TMDB.

Getting a key (free):
  1. Create an account at https://www.themoviedb.org/signup
  2. Settings -> API -> request a Developer key (instant, free tier)
  3. Put it in backend/.env as:  TMDB_API_KEY=<your v3 api key>

Usage (one-time, refresh whenever):
  python -m app.knowledge.ingest_tmdb                # top ~5,000 titles
  python -m app.knowledge.ingest_tmdb --depth 2000   # fewer

Stores title/year/genres/overview/popularity and embeds overviews on CPU.
Re-running refreshes existing rows (upsert by tmdb_id).
"""
from __future__ import annotations

import argparse
import time

import httpx

from .. import config
from ..logging_setup import get_logger
from . import db
from .embedder import embed, to_blob

log = get_logger(__name__)

BASE = "https://api.themoviedb.org/3"
LISTS = [("movie", "popular"), ("movie", "top_rated"), ("tv", "popular"), ("tv", "top_rated")]
PAGE_SIZE = 20


def _genres(client: httpx.Client, kind: str, key: str) -> dict[int, str]:
    r = client.get(f"{BASE}/genre/{kind}/list", params={"api_key": key})
    r.raise_for_status()
    return {g["id"]: g["name"] for g in r.json().get("genres", [])}


def ingest(depth: int = 5000) -> None:
    key = config.TMDB_API_KEY
    if not key:
        raise SystemExit(
            "TMDB_API_KEY is not set. Get a free key at https://www.themoviedb.org "
            "(Settings -> API) and add TMDB_API_KEY=... to backend/.env")

    per_list = max(1, depth // len(LISTS))
    pages = (per_list + PAGE_SIZE - 1) // PAGE_SIZE
    conn = db.connect()
    seen: set[int] = set()
    batch: list[tuple[int, str]] = []  # (title_id, text to embed)

    with httpx.Client(timeout=20) as client:
        genre_names = {kind: _genres(client, kind, key) for kind in ("movie", "tv")}
        for kind, list_name in LISTS:
            for page in range(1, pages + 1):
                r = client.get(f"{BASE}/{kind}/{list_name}",
                               params={"api_key": key, "page": page})
                if r.status_code == 429:  # rate limited: back off once
                    time.sleep(2)
                    r = client.get(f"{BASE}/{kind}/{list_name}",
                                   params={"api_key": key, "page": page})
                r.raise_for_status()
                for item in r.json().get("results", []):
                    tmdb_id = item["id"]
                    if tmdb_id in seen:
                        continue
                    seen.add(tmdb_id)
                    title = item.get("title") or item.get("name") or ""
                    date = item.get("release_date") or item.get("first_air_date") or ""
                    year = int(date[:4]) if date[:4].isdigit() else None
                    genres = ", ".join(
                        genre_names[kind].get(g, "") for g in item.get("genre_ids", []))
                    overview = item.get("overview") or ""
                    tid = db.upsert_title(
                        conn, title=title, year=year,
                        type_="show" if kind == "tv" else "movie",
                        genres=genres, overview=overview,
                        popularity=float(item.get("popularity") or 0), tmdb_id=tmdb_id)
                    batch.append((tid, f"{title} ({year}). {genres}. {overview}"))
                log.info("%s/%s page %d/%d (%d titles)", kind, list_name, page, pages, len(seen))
                time.sleep(0.05)

    # embed in chunks (CPU)
    for i in range(0, len(batch), 256):
        chunk = batch[i : i + 256]
        vecs = embed([t for _, t in chunk])
        db.put_embeddings(conn, "titles", [(tid, to_blob(v)) for (tid, _), v in zip(chunk, vecs)])
        log.info("embedded %d/%d overviews", min(i + 256, len(batch)), len(batch))
    conn.commit()
    log.info("TMDB ingest done: %s", db.stats(conn))
    conn.close()


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Ingest TMDB metadata into the movie KB")
    ap.add_argument("--depth", type=int, default=int(config.KB_TMDB_DEPTH),
                    help="approx number of titles to pull (default %(default)s)")
    args = ap.parse_args()
    ingest(args.depth)


if __name__ == "__main__":
    _cli()
