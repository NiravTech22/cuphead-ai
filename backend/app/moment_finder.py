"""moment_finder.py — locate the best moment window for a query.

Given a natural-language query + a Transcript (+ optional emotion timeline),
find the window that best matches. Strategy:

  1. Build overlapping windows of consecutive segments.
  2. Semantic match: sentence-transformers cosine over window text vs. query.
  3. If the top scores are close / weak, ask an LLM ranker to pick among the
     top-K candidates (fallback only — cheap model, optional).
  4. Optionally nudge toward windows whose emotion matches emotion words in the
     query ("celebrates" -> happy, "furious" -> angry, ...).

Returns a Moment: {start, end, quote, dominant_emotion, score, reasoning}.
Standalone-runnable against a saved transcript JSON:

    python -m app.moment_finder transcript.json "the part where he says the truth"
"""
from __future__ import annotations

import argparse
import difflib
import functools
import json
import re
from dataclasses import dataclass
from typing import Optional

from . import config
from .logging_setup import get_logger
from .transcriber import Transcript

log = get_logger(__name__)


@dataclass
class Moment:
    start: float
    end: float
    quote: str
    score: float
    dominant_emotion: Optional[str] = None
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "quote": self.quote,
            "score": round(self.score, 4),
            "dominant_emotion": self.dominant_emotion,
            "reasoning": self.reasoning,
        }


# emotion words in a query -> FER label, used as a soft prior only.
_EMOTION_HINTS = {
    "happy": ["happy", "celebrat", "joy", "smil", "laugh", "excit", "win", "scores"],
    "angry": ["angry", "furious", "rage", "yell", "mad", "shout"],
    "sad": ["sad", "cry", "grief", "mourn", "tear", "heartbreak"],
    "surprise": ["surprise", "shock", "gasp", "stunn", "reveal", "twist"],
    "fear": ["fear", "scared", "afraid", "terrif", "panic"],
    "disgust": ["disgust", "gross", "repuls"],
}


def _emotion_prior(query: str) -> Optional[str]:
    q = query.lower()
    for emo, keys in _EMOTION_HINTS.items():
        if any(k in q for k in keys):
            return emo
    return None


@functools.lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    from .device import torch_device

    dev = torch_device()
    log.info("loading sentence-transformer all-MiniLM-L6-v2 on %s", dev)
    return SentenceTransformer("all-MiniLM-L6-v2", device=dev)


@dataclass
class _Window:
    start: float
    end: float
    text: str
    seg_idx: tuple[int, int]
    emotion: Optional[str] = None


def _build_windows(t: Transcript, target_secs: float = 12.0) -> list[_Window]:
    """Sliding windows of consecutive segments, each ~target_secs long, 50% overlap."""
    segs = t.segments
    windows: list[_Window] = []
    n = len(segs)
    if n == 0:
        return windows
    i = 0
    while i < n:
        j = i
        while j < n and (segs[j].end - segs[i].start) < target_secs:
            j += 1
        j = min(max(j, i), n - 1)
        text = " ".join(s.text.strip() for s in segs[i : j + 1]).strip()
        if text:
            windows.append(
                _Window(
                    start=segs[i].start,
                    end=segs[j].end,
                    text=text,
                    seg_idx=(i, j),
                )
            )
        # advance by roughly half the window
        step = max(1, (j - i + 1) // 2)
        i += step
    return windows


def _attach_emotions(windows: list[_Window], emotion_timeline: Optional[list[dict]]) -> None:
    """Set each window's dominant emotion from the timeline (list of {t, emotion})."""
    if not emotion_timeline:
        return
    for w in windows:
        counts: dict[str, int] = {}
        for e in emotion_timeline:
            if w.start <= e.get("t", -1) <= w.end:
                lab = e.get("emotion")
                if lab:
                    counts[lab] = counts.get(lab, 0) + 1
        if counts:
            w.emotion = max(counts, key=counts.get)


# ── span refinement ──────────────────────────────────────────────────────────
# The matched window is 12-18s of pooled transcript; the actual line may sit
# anywhere inside it, which used to yield clips that were mostly lead-up.
# Refine to the exact moment using word-level timestamps.

_QUOTE_RE = re.compile(r"[\'\"‘“]([^\'\"’”]{3,})[\'\"’”]")


def _norm(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _find_phrase_span(phrase: str, transcript: Transcript) -> Optional[tuple[float, float, float]]:
    """Fuzzy-match a quoted phrase against the word stream; return (start, end, score)."""
    words = [
        (w.start, w.end, n)
        for seg in transcript.segments
        for w in seg.words
        for n in [" ".join(_norm(w.word))]
        if n
    ]
    target = " ".join(_norm(phrase))
    m = len(_norm(phrase))
    if not words or not m:
        return None
    best_score, best_span = 0.0, None
    for i in range(len(words) - m + 1):
        cand = " ".join(w[2] for w in words[i : i + m])
        score = difflib.SequenceMatcher(None, cand, target).ratio()
        if score > best_score:
            best_score, best_span = score, (words[i][0], words[i + m - 1][1])
    if best_span and best_score >= 0.72:
        return best_span[0], best_span[1], best_score
    return None


def _refine_window(query: str, transcript: Transcript, w: "_Window",
                   q_emb) -> tuple[float, float, str]:
    """Tighten a matched window to its best segment(s) instead of the whole window."""
    segs = transcript.segments[w.seg_idx[0] : w.seg_idx[1] + 1]
    if len(segs) <= 1:
        return w.start, w.end, w.text

    from sentence_transformers import util

    model = _embedder()
    embs = model.encode([s.text for s in segs], convert_to_tensor=True,
                        normalize_embeddings=True)
    sims = util.cos_sim(q_emb, embs)[0].cpu().tolist()
    best = max(range(len(segs)), key=lambda i: sims[i])
    lo = hi = best
    # absorb neighbors that carry nearly the same signal, up to ~12s total
    while lo - 1 >= 0 and sims[lo - 1] >= 0.8 * sims[best] \
            and segs[hi].end - segs[lo - 1].start <= 12.0:
        lo -= 1
    while hi + 1 < len(segs) and sims[hi + 1] >= 0.8 * sims[best] \
            and segs[hi + 1].end - segs[lo].start <= 12.0:
        hi += 1
    text = " ".join(s.text.strip() for s in segs[lo : hi + 1]).strip()
    return segs[lo].start, segs[hi].end, text


def _emotion_at(emotion_timeline: Optional[list[dict]], start: float, end: float) -> Optional[str]:
    if not emotion_timeline:
        return None
    counts: dict[str, int] = {}
    for e in emotion_timeline:
        if start - 3.0 <= e.get("t", -1) <= end + 3.0 and e.get("emotion"):
            counts[e["emotion"]] = counts.get(e["emotion"], 0) + 1
    return max(counts, key=counts.get) if counts else None


def find_moment(
    query: str,
    transcript: Transcript,
    emotion_timeline: Optional[list[dict]] = None,
    use_llm_fallback: bool = True,
    top_k: int = 5,
    hint_t: Optional[float] = None,
) -> Optional[Moment]:
    # Fast path: an explicitly quoted line pins the moment via word timestamps.
    quoted = _QUOTE_RE.search(query)
    if quoted:
        span = _find_phrase_span(quoted.group(1), transcript)
        if span:
            s, e, score = span
            moment = Moment(
                start=s, end=e, quote=quoted.group(1), score=score,
                dominant_emotion=_emotion_at(emotion_timeline, s, e),
                reasoning=f"quoted phrase matched in word stream (fuzzy={score:.2f})",
            )
            log.info("quoted-phrase match: [%.2f-%.2f] score=%.2f", s, e, score)
            return moment

    windows = _build_windows(transcript)
    if not windows:
        log.warning("no windows built from transcript")
        return None
    _attach_emotions(windows, emotion_timeline)

    from sentence_transformers import util

    model = _embedder()
    q_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    w_embs = model.encode(
        [w.text for w in windows], convert_to_tensor=True, normalize_embeddings=True
    )
    cos = util.cos_sim(q_emb, w_embs)[0].cpu().tolist()

    prior = _emotion_prior(query)
    scored: list[tuple[float, _Window, float]] = []
    for base, w in zip(cos, windows):
        bonus = 0.06 if (prior and w.emotion == prior) else 0.0
        if hint_t is not None:  # KB timestamp hint: prefer windows near it
            bonus += 0.1 * max(0.0, 1.0 - abs(w.start - hint_t) / 90.0)
        scored.append((base + bonus, w, base))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[:top_k]
    best_score, best_w, best_base = top[0]
    reasoning = f"semantic cosine={best_base:.3f}"
    if prior and best_w.emotion == prior:
        reasoning += f"; emotion prior '{prior}' matched"

    # LLM fallback when the field is ambiguous (weak or clustered top scores).
    ambiguous = best_base < 0.45 or (
        len(top) > 1 and abs(top[0][0] - top[1][0]) < 0.04
    )
    if use_llm_fallback and ambiguous:
        picked = _llm_rerank(query, [w for _, w, _ in top])
        if picked is not None:
            best_w = picked
            best_score = next(s for s, w, _ in top if w is picked)
            reasoning += "; LLM ranker resolved an ambiguous match"
            log.info("LLM ranker chose window [%.1f-%.1f]", best_w.start, best_w.end)

    # Tighten to the segment(s) that actually carry the moment.
    r_start, r_end, r_text = _refine_window(query, transcript, best_w, q_emb)
    if (r_start, r_end) != (best_w.start, best_w.end):
        reasoning += f"; refined {best_w.end - best_w.start:.0f}s window to [{r_start:.1f}-{r_end:.1f}]"

    moment = Moment(
        start=r_start,
        end=r_end,
        quote=r_text,
        score=float(best_score),
        dominant_emotion=_emotion_at(emotion_timeline, r_start, r_end) or best_w.emotion,
        reasoning=reasoning,
    )
    log.info("best moment: [%.1f-%.1f] score=%.3f", moment.start, moment.end, moment.score)
    return moment


def _llm_rerank(query: str, candidates: list[_Window]) -> Optional[_Window]:
    """Ask the LLM (provider-agnostic) to choose the best candidate window index."""
    try:
        from . import llm

        listing = "\n".join(
            f"[{i}] ({w.start:.0f}s-{w.end:.0f}s) {w.text}" for i, w in enumerate(candidates)
        )
        resp = llm.chat([
            {"role": "system", "content": "You pick which transcript window best matches the "
             "user's request. Reply with ONLY a JSON object: {\"index\": <int>}. No prose."},
            {"role": "user", "content": f"Request: {query}\n\nWindows:\n{listing}"},
        ])
        m = re.search(r'"index"\s*:\s*(\d+)', resp.text)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < len(candidates):
                return candidates[idx]
    except Exception as exc:
        log.warning("LLM rerank failed (%s); keeping semantic top-1", exc)
    return None


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Find a moment in a saved transcript")
    ap.add_argument("transcript_json")
    ap.add_argument("query")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    t = Transcript.from_dict(json.loads(open(args.transcript_json).read()))
    m = find_moment(args.query, t, use_llm_fallback=not args.no_llm)
    print(json.dumps(m.to_dict() if m else {"error": "no moment"}, indent=2))


if __name__ == "__main__":
    _cli()
