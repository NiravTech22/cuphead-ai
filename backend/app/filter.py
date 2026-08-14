"""filter.py — configurable content filter (blocklist-based).

Terms live in blocked_terms.txt (editable, no code changes needed; reloaded on
mtime change). Matching is case-insensitive with word boundaries, so innocent
substrings never false-positive ("assistant" does not trip on "ass").

Categories [sexual] [slurs] [violence] always apply; [strict] applies only when
FILTER_STRICT is on (default true — also blocks R-rated-adjacent asks).

Applied at three points (see assistant.py):
  inbound user query -> polite refusal, no tool loop
  predicted suggestions -> dropped before rendering
  outbound located statement quote -> "statement isn't available" instead of the clip

Logging: category only, never the user's full text.
"""
from __future__ import annotations

import re
import threading
from typing import Optional

from . import config
from .logging_setup import get_logger

log = get_logger(__name__)

REFUSAL = "I can't help with that request — try asking about a different statement."
UNAVAILABLE = "This statement isn't available."

_lock = threading.Lock()
_mtime: float = -1.0
_patterns: dict[str, re.Pattern] = {}
_strict: bool = config.FILTER_STRICT


def _compile(terms: list[str]) -> Optional[re.Pattern]:
    if not terms:
        return None
    parts = [re.escape(t).replace(r"\ ", r"\s+") for t in terms]
    return re.compile(r"(?<![\w])(?:" + "|".join(parts) + r")(?![\w])", re.IGNORECASE)


def _load() -> dict[str, re.Pattern]:
    global _mtime, _patterns
    path = config.BLOCKED_TERMS_PATH
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return {}
    if mtime == _mtime:
        return _patterns
    with _lock:
        if mtime == _mtime:
            return _patterns
        cats: dict[str, list[str]] = {}
        current = "misc"
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip().lower()
                continue
            cats.setdefault(current, []).append(line.lower())
        _patterns = {c: p for c, terms in cats.items() if (p := _compile(terms))}
        _mtime = mtime
        log.info("content filter loaded: %s",
                 {c: len(p.pattern.split('|')) for c, p in _patterns.items()})
    return _patterns


def check(text: Optional[str], strict: Optional[bool] = None) -> Optional[str]:
    """Return the matched category name if `text` is blocked, else None."""
    if not text:
        return None
    strict = _strict if strict is None else strict
    for cat, pattern in _load().items():
        if cat == "strict" and not strict:
            continue
        if pattern.search(text):
            return cat
    return None


def is_strict() -> bool:
    return _strict


def set_strict(value: bool) -> bool:
    global _strict
    _strict = bool(value)
    log.info("content filter strict mode -> %s", _strict)
    return _strict
