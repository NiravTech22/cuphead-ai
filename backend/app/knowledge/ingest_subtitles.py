"""knowledge/ingest_subtitles.py — index a LOCAL folder of .srt transcripts into the KB.

Point it at transcripts you own (speeches, interviews, press conferences); each
file's lines are stored as quotes with timestamps and embedded for semantic
search. The event title is guessed from the filename
("Obama.Victory.Speech.2008.srt" -> "Obama Victory Speech", 2008) and matched
to an existing KB title when possible.

This tool does NOT fetch transcripts from anywhere — bring your own files.

  python -m app.knowledge.ingest_subtitles /path/to/srt/folder
  python -m app.knowledge.ingest_subtitles file.srt --title "Obama Victory Speech" --year 2008
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..logging_setup import get_logger
from . import db
from .embedder import embed, to_blob

log = get_logger(__name__)

_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}")
_TAG_RE = re.compile(r"<[^>]+>|\{[^}]+\}")
_CUE_RE = re.compile(r"^\s*[\[(♪#].*[\])♪#]\s*$")  # [door slams], (music), ♪ lyrics ♪
_YEAR_RE = re.compile(r"(19|20)\d{2}")

MIN_LINE_CHARS = 8


def _guess_title(path: Path) -> tuple[str, int | None]:
    stem = path.stem
    year = None
    m = _YEAR_RE.search(stem)
    if m:
        year = int(m.group(0))
        stem = stem[: m.start()]
    stem = re.sub(r"[._-]+", " ", stem)
    stem = re.sub(r"\b(1080p|720p|2160p|bluray|brrip|webrip|web|x264|x265|hdtv|dvdrip)\b.*",
                  "", stem, flags=re.I)
    return stem.strip().title(), year


def parse_srt(path: Path) -> list[tuple[float, str]]:
    """Return [(start_seconds, dialogue_line), ...] with tags/cues stripped."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out: list[tuple[float, str]] = []
    t = None
    buf: list[str] = []

    def flush():
        nonlocal buf, t
        if t is not None and buf:
            line = " ".join(buf).strip()
            if len(line) >= MIN_LINE_CHARS and not _CUE_RE.match(line):
                out.append((t, line))
        buf = []

    for raw in text.splitlines():
        line = _TAG_RE.sub("", raw).strip()
        m = _TIME_RE.match(line)
        if m:
            flush()
            h, mi, s, ms = (int(x) for x in m.groups())
            t = h * 3600 + mi * 60 + s + ms / 1000
        elif not line:
            flush()
            t = None
        elif line.isdigit() and t is None:
            continue  # cue number
        elif t is not None:
            buf.append(line.lstrip("- "))
    flush()
    return out


def ingest_file(conn, path: Path, title: str | None = None, year: int | None = None) -> int:
    g_title, g_year = _guess_title(path)
    title = title or g_title
    year = year if year is not None else g_year
    if not title:
        log.warning("skipping %s: could not determine a title", path.name)
        return 0

    # reuse an existing title row when we have one; else create a bare one
    row = conn.execute(
        "SELECT id FROM titles WHERE LOWER(title)=LOWER(?) "
        "AND (? IS NULL OR year IS NULL OR year=?)", (title, year, year)).fetchone()
    tid = row["id"] if row else db.upsert_title(
        conn, title=title, year=year, type_="speech", genres="", overview="")

    conn.execute("DELETE FROM quotes WHERE title_id=? AND approx_timestamp IS NOT NULL", (tid,))
    conn.execute("INSERT INTO quotes_fts(quotes_fts) VALUES('rebuild')")

    lines = parse_srt(path)
    rows: list[tuple[int, str]] = []
    for ts, line in lines:
        qid = db.add_quote(conn, tid, line, "", ts)
        rows.append((qid, line))

    for i in range(0, len(rows), 512):
        chunk = rows[i : i + 512]
        vecs = embed([q for _, q in chunk])
        db.put_embeddings(conn, "quotes", [(qid, to_blob(v)) for (qid, _), v in zip(chunk, vecs)])
    log.info("indexed %s -> %r (%s): %d lines", path.name, title, year, len(rows))
    return len(rows)


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Index local .srt transcripts into the statements KB")
    ap.add_argument("path", help=".srt file or a folder of .srt files")
    ap.add_argument("--title", help="override the title guessed from the filename")
    ap.add_argument("--year", type=int)
    args = ap.parse_args()

    p = Path(args.path)
    files = [p] if p.is_file() else sorted(p.glob("**/*.srt"))
    if not files:
        raise SystemExit(f"no .srt files found under {p}")
    conn = db.connect()
    total = 0
    for f in files:
        total += ingest_file(conn, f, args.title if p.is_file() else None,
                             args.year if p.is_file() else None)
    conn.commit()
    conn.close()
    log.info("subtitle ingest done: %d lines from %d file(s); KB now %s",
             total, len(files), db.stats())


if __name__ == "__main__":
    _cli()
