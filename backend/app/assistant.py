"""assistant.py — the orchestrator of the core interaction (provider-agnostic).

Given a user question about a movie/show/video, it runs a manual tool-use loop
over the LLM adapter (`llm.chat`, default = local Ollama): identify the source ->
fetch & transcribe the video (yt-dlp) -> locate the moment (moment_finder) ->
cut the clip (clipper) -> and return a STRUCTURED response:
{explanation, clip_url, source, quote, dominant_emotion, timestamp, citations}.

It yields SSE-style events (status / text / tool / clip / done). It HEDGES when
it can't confidently find the moment and CITES sources. Graceful errors never
crash the flow.

    python -m app.assistant "show me the scene where 'why so serious' is said"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Iterator, Optional

from . import source_id, verifier, websearch
from .clipper import make_clip
from .downloader import cached_download, download, search
from .llm import chat as llm_chat
from .llm import local_model, provider
from .logging_setup import get_logger
from .moment_finder import find_moment
from .session import STORE, LoadedVideo, Session
from .transcriber import release_model, transcribe

log = get_logger(__name__)

SYSTEM_PROMPT = """You are Scene Sense — a conversational clip finder and scene analyst.

You have TOOLS. To use a tool you MUST emit a real tool call — NEVER write a tool call
as text (e.g. writing "make_clip start=12" in your message does nothing).

Two kinds of request:

A) "show me / find / play the scene (or clip/moment/part) where ..." — the user wants
   the actual clip. You MUST, in order:
     1. find_and_fetch_video  (unless a video is already loaded) — use a specific query
        such as an official movie CLIP title (you may call identify_source first to get it).
     2. locate_moment  — find the exact window in the loaded video.
     3. make_clip  — cut it. Only AFTER a clip exists do you write the final answer.
   Do NOT call verify_quote for these. Do NOT describe the scene before the clip is made.

B) "did they really say X" / "what's the exact line" — call verify_quote only.

RULES:
- You identify the WORK (film/show/video), never a person by their face.
- Cite the source title. NEVER fabricate a quote, source, or scene.
- If you genuinely cannot find the clip, SAY SO and offer your best candidate + reasoning.
- Final answer: tight and vivid (2-4 sentences) — what the scene is, its tone, what ties
  it together. No step narration."""


_CLIP_WORDS = ("show", "find", "play", "clip", "scene", "moment", "the part", "where he",
               "where she", "where they")
_VERIFY_WORDS = ("did they really", "did he really", "did she really", "what's the exact",
                 "what is the exact", "misquote", "actually say", "real line")


def _is_clip_request(question: str) -> bool:
    q = question.lower()
    if any(w in q for w in _VERIFY_WORDS):
        return False
    return any(w in q for w in _CLIP_WORDS)


def _tools() -> list[dict]:
    def fn(name, desc, props, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        }

    return [
        fn("web_search", "Search the web for facts (source identification, exact quotes). "
           "Returns titles, urls, snippets.", {"query": {"type": "string"}}, ["query"]),
        fn("identify_source", "Identify the SOURCE (movie/show/video) of a scene from dialogue "
           "or description. Returns a cited best guess + a query to fetch the clip.",
           {"query": {"type": "string"}}, ["query"]),
        fn("find_and_fetch_video", "Search YouTube and download+transcribe the top match, "
           "loading it as the current video. Use a specific query (e.g. an official clip title).",
           {"query": {"type": "string"}}, ["query"]),
        fn("locate_moment", "Locate the best moment window for a query within the loaded "
           "video's transcript. Returns {start,end,quote,dominant_emotion,score}. Needs a loaded video.",
           {"query": {"type": "string"}}, ["query"]),
        fn("make_clip", "Cut a standalone playable mp4 clip [start,end] from the loaded video "
           "and return its clip_url. Call after locate_moment.",
           {"start": {"type": "number"}, "end": {"type": "number"},
            "quote": {"type": "string"}, "dominant_emotion": {"type": "string"}},
           ["start", "end"]),
        fn("verify_quote", "Verify/correct a possibly-misremembered quote. Returns the exact "
           "line + source + confidence; hedges when unverifiable.",
           {"quote": {"type": "string"}, "source_context": {"type": "string"}}, ["quote"]),
    ]


class _Result:
    def __init__(self) -> None:
        self.clip: Optional[dict] = None
        self.source: Optional[str] = None
        self.quote: Optional[str] = None
        self.dominant_emotion: Optional[str] = None
        self.timestamp: Optional[float] = None
        self.verification: Optional[dict] = None
        self.citations: list[dict] = []


def _run_tool(name: str, args: dict, session: Session, result: _Result,
              qid: str = "-", timings: Optional[dict] = None) -> dict:
    timings = timings if timings is not None else {}

    def _timed(phase: str, fn, *a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            timings[phase] = timings.get(phase, 0.0) + (time.perf_counter() - t0)

    if name == "web_search":
        results = websearch.search(args.get("query", ""))
        result.citations.extend(
            {"title": r["title"], "url": r["url"], "cited_text": r["snippet"][:200]} for r in results
        )
        return {"results": results}

    if name == "identify_source":
        data = source_id.identify_source(args["query"])
        if data.get("source_title"):
            result.source = data["source_title"]
        result.citations.extend(data.get("citations", []))
        return data

    if name == "find_and_fetch_video":
        query = args["query"]
        found = _timed("search", search, query, limit=3)
        if not found:
            return {"error": "no search results", "query": query}
        info = cached_download(found[0].id) if found[0].id else None
        if info:
            log.info("[%s] download cache hit for %s -> %s", qid, found[0].id, info.local_path)
        else:
            info = _timed("download", download, found[0].webpage_url)
        if not info.local_path:
            return {"error": "download failed", "query": query}
        transcript = _timed("transcribe", transcribe, info.local_path)
        release_model()  # free Whisper VRAM so the local LLM doesn't spill to CPU
        session.video = LoadedVideo(
            path=info.local_path, title=info.title, channel=info.channel,
            webpage_url=info.webpage_url, duration=info.duration, transcript=transcript,
            video_id=info.id, pinned=False,
        )
        result.source = result.source or info.title
        log.info("[%s] resolved source=%r id=%s url=%s transcript_segments=%d",
                 qid, info.title, info.id, info.webpage_url, len(transcript.segments))
        return {
            "loaded": True, "title": info.title, "channel": info.channel,
            "duration": info.duration, "webpage_url": info.webpage_url,
            "transcript_segments": len(transcript.segments),
            "transcript_preview": transcript.full_text()[:600],
        }

    if name == "locate_moment":
        if not session.video or not session.video.transcript:
            return {"error": "no video loaded; call find_and_fetch_video first"}
        moment = _timed("locate", find_moment, args["query"], session.video.transcript,
                        emotion_timeline=session.video.emotion_timeline or None)
        if not moment:
            log.info("[%s] locate_moment: no match in transcript of %r (%s)",
                     qid, session.video.title, session.video.path)
            return {"error": "no matching moment found"}
        result.quote = moment.quote
        result.dominant_emotion = moment.dominant_emotion
        result.timestamp = moment.start
        log.info("[%s] locate_moment searched transcript of %r (%s) -> chose [%.2f-%.2f]",
                 qid, session.video.title, session.video.path, moment.start, moment.end)
        return moment.to_dict()

    if name == "make_clip":
        if not session.video:
            return {"error": "no video loaded"}
        clip = _timed("clip", make_clip,
                      session.video.path, float(args["start"]), float(args["end"]),
                      quote=args.get("quote", result.quote or ""),
                      dominant_emotion=args.get("dominant_emotion", result.dominant_emotion))
        result.clip = clip.to_dict()
        if result.timestamp is None:
            result.timestamp = clip.start
        return {"clip_url": clip.url, "start": clip.start, "end": clip.end, "duration": clip.duration}

    if name == "verify_quote":
        data = verifier.verify_quote(args["quote"], args.get("source_context"))
        result.verification = data
        result.source = result.source or data.get("source")
        if data.get("citation"):
            result.citations.append(data["citation"])
        return data

    return {"error": f"unknown tool {name}"}


_STEP_LABELS = {
    "web_search": "searching the web",
    "identify_source": "identifying source",
    "find_and_fetch_video": "fetching video",
    "locate_moment": "locating moment",
    "make_clip": "cutting clip",
    "verify_quote": "verifying quote",
}


def run(question: str, session_id: str = "default") -> Iterator[dict]:
    """Yield events: status / text / tool / clip / done. The core interaction."""
    qid = uuid.uuid4().hex[:8]
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    session = STORE.get(session_id)

    # Per-query isolation: a video auto-fetched for a PREVIOUS question must not
    # leak into this one. Each new question resolves its source and transcript
    # from scratch. Only user-pinned videos (load_url/upload) survive turns.
    if session.video and not session.video.pinned:
        log.info("[%s] new query: clearing auto-fetched video %r from previous query",
                 qid, session.video.title)
        session.video = None

    log.info("[%s] question (session=%s): %s", qid, session_id, question)
    result = _Result()
    if session.video:
        result.source = session.video.title or None

    # Retrieval runs with a CLEAN context scoped to this question only — no prior
    # turns, no prior scene data.
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if session.video:  # pinned by the user — the one legitimate carry-over
        messages.insert(1, {"role": "system",
                            "content": f"A video is already loaded: {session.video.title!r} "
                                       f"({session.video.duration:.0f}s). Use locate_moment then make_clip."})

    clip_request = _is_clip_request(question)
    called: set[str] = set()
    explanation_parts: list[str] = []
    nudges = 0
    final_text = ""
    clip_emitted = False

    for _turn in range(9):
        try:
            t0 = time.perf_counter()
            resp = llm_chat(messages, tools=_tools())
            timings["llm"] = timings.get("llm", 0.0) + (time.perf_counter() - t0)
        except Exception as exc:
            log.exception("llm.chat failed")
            yield {"type": "error",
                   "message": f"LLM error ({provider()}/{local_model()}): {exc}"}
            return

        # record the assistant turn (with any tool calls) for history
        messages.append({
            "role": "assistant",
            "content": resp.text,
            "tool_calls": [
                {"id": tc["id"], "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in resp.tool_calls
            ],
        })

        if not resp.tool_calls:
            # Small models often stop early or narrate a fake tool call. If this is a
            # clip request and no clip exists yet, nudge toward the next real tool.
            if clip_request and result.clip is None and nudges < 3:
                nudges += 1
                if not session.video and "find_and_fetch_video" not in called:
                    step = ("You have NOT fetched a video yet. Call find_and_fetch_video now "
                            "with a specific query (e.g. an official movie CLIP title). "
                            "Do not answer in text — emit the tool call.")
                elif session.video and "locate_moment" not in called:
                    step = ("The video is loaded. Call locate_moment now to find the exact "
                            "window. Emit the tool call — do not describe it.")
                else:
                    step = ("Call make_clip now with the located start and end. "
                            "Emit the tool call — do not describe it.")
                yield {"type": "status", "step": "planning next step"}
                messages.append({"role": "user", "content": step})
                continue
            final_text = resp.text
            if resp.text:
                explanation_parts.append(resp.text)
                yield {"type": "text", "delta": resp.text}
            break

        # tool calls present: surface any pre-tool commentary but drop fake inline calls
        if resp.text and "make_clip" not in resp.text and "find_and_fetch_video" not in resp.text:
            explanation_parts.append(resp.text)
            yield {"type": "text", "delta": resp.text}

        for tc in resp.tool_calls:
            name, args, tid = tc["name"], tc["arguments"], tc["id"]
            called.add(name)
            yield {"type": "status", "step": _STEP_LABELS.get(name, name)}
            yield {"type": "tool", "name": name, "input": args}
            try:
                out = _run_tool(name, args, session, result, qid, timings)
            except Exception as exc:
                log.exception("[%s] tool %s failed", qid, name)
                out = {"error": str(exc)}

            # Batch locate -> clip: cutting is deterministic once the moment is
            # found, so skip the extra LLM round-trip on clip requests.
            if (name == "locate_moment" and clip_request and result.clip is None
                    and "start" in out and "end" in out):
                called.add("make_clip")
                yield {"type": "status", "step": _STEP_LABELS["make_clip"]}
                try:
                    clip_out = _run_tool("make_clip",
                                         {"start": out["start"], "end": out["end"]},
                                         session, result, qid, timings)
                    out = {**out, "clip": clip_out,
                           "note": "clip already cut — write the final 2-4 sentence "
                                   "answer now; do NOT call make_clip"}
                except Exception as exc:
                    log.exception("[%s] auto make_clip failed", qid)

            if result.clip and not clip_emitted:
                clip_emitted = True
                yield {"type": "clip", "clip": result.clip}
            messages.append({"role": "tool", "tool_call_id": tid, "name": name,
                             "content": json.dumps(out)[:4000]})

    # If we ran out of turns on a clip request without a clip, give an honest one-liner
    # instead of leaving the answer blank.
    if clip_request and result.clip is None and not final_text:
        hedge = ("I couldn't confidently pull that exact clip just now"
                 + (f" — best guess: {result.source}." if result.source else ".")
                 + " Try rephrasing with the movie name, or load a video and ask again.")
        explanation_parts.append(hedge)
        yield {"type": "text", "delta": hedge}

    # de-dupe citations
    seen, citations = set(), []
    for c in result.citations:
        key = c.get("url") or c.get("title")
        if key and key not in seen:
            seen.add(key)
            citations.append(c)

    total = time.perf_counter() - t_start
    spent = " ".join(f"{k}={v:.1f}s" for k, v in timings.items())
    log.info("[%s] answered: source=%r ts=%s clip=%s | timing: %s total=%.1fs",
             qid, result.source, result.timestamp,
             (result.clip or {}).get("url"), spent or "-", total)

    payload = {
        "request_id": qid,
        "explanation": "".join(explanation_parts).strip(),
        "clip_url": result.clip["url"] if result.clip else None,
        "clip": result.clip,
        "source": result.source,
        "quote": result.quote,
        "dominant_emotion": result.dominant_emotion,
        "timestamp": result.timestamp,
        "verification": result.verification,
        "citations": citations,
        "video": session.video.summary() if session.video else None,
    }
    yield {"type": "done", "payload": payload}


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Ask Scene Sense a question (standalone)")
    ap.add_argument("question")
    ap.add_argument("--session", default="default")
    args = ap.parse_args()
    final = None
    for event in run(args.question, args.session):
        t = event["type"]
        if t == "text":
            sys.stdout.write(event["delta"]); sys.stdout.flush()
        elif t == "status":
            print(f"\n  … {event['step']}", file=sys.stderr)
        elif t == "tool":
            print(f"  [tool] {event['name']}({json.dumps(event['input'])[:120]})", file=sys.stderr)
        elif t == "clip":
            print(f"  [clip] {event['clip']['url']}", file=sys.stderr)
        elif t == "error":
            print(f"\n  [error] {event['message']}", file=sys.stderr)
        elif t == "done":
            final = event["payload"]
    print("\n\n=== STRUCTURED PAYLOAD ===")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    _cli()
