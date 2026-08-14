"""knowledge/db.py — local SQLite public-statements knowledge base (RAG store).

Schema:
  titles      (id, tmdb_id, title, year, type[speech|interview|press|debate], genres, overview, popularity)
              -- "title" is the event/speech title, "genres" doubles as topic tags,
              -- "tmdb_id" is a legacy column name now used as a generic external-id dedup key
  quotes      (id, title_id, quote_text, character, approx_timestamp)
              -- "character" holds the SPEAKER's name
  embeddings  (row_id, source_table, vector BLOB)   -- float32, normalized
  titles_fts / quotes_fts                            -- FTS5 keyword indexes

This is a retrieval layer, not training: nothing here touches the GPU.
"""
from __future__ import annotations

import sqlite3

from .. import config
from ..logging_setup import get_logger

log = get_logger(__name__)

DB_PATH = config.DATA_DIR / "statements.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS titles (
    id          INTEGER PRIMARY KEY,
    tmdb_id     INTEGER UNIQUE,
    title       TEXT NOT NULL,
    year        INTEGER,
    type        TEXT NOT NULL DEFAULT 'speech',
    genres      TEXT NOT NULL DEFAULT '',
    overview    TEXT NOT NULL DEFAULT '',
    popularity  REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS quotes (
    id               INTEGER PRIMARY KEY,
    title_id         INTEGER NOT NULL REFERENCES titles(id),
    quote_text       TEXT NOT NULL,
    character        TEXT DEFAULT '',
    approx_timestamp REAL
);
CREATE TABLE IF NOT EXISTS embeddings (
    row_id       INTEGER NOT NULL,
    source_table TEXT NOT NULL,
    vector       BLOB NOT NULL,
    PRIMARY KEY (row_id, source_table)
);
CREATE VIRTUAL TABLE IF NOT EXISTS titles_fts USING fts5(
    title, overview, content='titles', content_rowid='id', tokenize='porter unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS quotes_fts USING fts5(
    quote_text, content='quotes', content_rowid='id', tokenize='porter unicode61'
);
CREATE INDEX IF NOT EXISTS idx_quotes_title ON quotes(title_id);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def upsert_title(conn: sqlite3.Connection, *, title: str, year: int | None,
                 type_: str, genres: str, overview: str,
                 popularity: float = 0.0, tmdb_id: int | None = None) -> int:
    """Insert or update a title; returns its row id."""
    row = None
    if tmdb_id is not None:
        row = conn.execute("SELECT id FROM titles WHERE tmdb_id=?", (tmdb_id,)).fetchone()
    if row is None:
        row = conn.execute("SELECT id FROM titles WHERE title=? AND IFNULL(year,0)=IFNULL(?,0)",
                           (title, year)).fetchone()
    if row:
        tid = row["id"]
        conn.execute(
            "UPDATE titles SET title=?, year=?, type=?, genres=?, overview=?, popularity=?, "
            "tmdb_id=COALESCE(?, tmdb_id) WHERE id=?",
            (title, year, type_, genres, overview, popularity, tmdb_id, tid))
        conn.execute("INSERT INTO titles_fts(titles_fts, rowid, title, overview) "
                     "VALUES('delete', ?, '', '')", (tid,))
    else:
        cur = conn.execute(
            "INSERT INTO titles (tmdb_id, title, year, type, genres, overview, popularity) "
            "VALUES (?,?,?,?,?,?,?)",
            (tmdb_id, title, year, type_, genres, overview, popularity))
        tid = cur.lastrowid
    conn.execute("INSERT INTO titles_fts(rowid, title, overview) VALUES (?,?,?)",
                 (tid, title, overview))
    return tid


def add_quote(conn: sqlite3.Connection, title_id: int, quote_text: str,
              character: str = "", approx_timestamp: float | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO quotes (title_id, quote_text, character, approx_timestamp) "
        "VALUES (?,?,?,?)", (title_id, quote_text, character, approx_timestamp))
    qid = cur.lastrowid
    conn.execute("INSERT INTO quotes_fts(rowid, quote_text) VALUES (?,?)", (qid, quote_text))
    return qid


def put_embeddings(conn: sqlite3.Connection, source_table: str,
                   rows: list[tuple[int, bytes]]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (row_id, source_table, vector) VALUES (?,?,?)",
        [(rid, source_table, vec) for rid, vec in rows])


def stats(conn: sqlite3.Connection | None = None) -> dict:
    own = conn is None
    conn = conn or connect()
    try:
        return {
            "titles": conn.execute("SELECT COUNT(*) FROM titles").fetchone()[0],
            "quotes": conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0],
            "embeddings": conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0],
        }
    finally:
        if own:
            conn.close()
