"""source_id.py — identify the SOURCE of a scene from distinctive dialogue + metadata.

Web-searches (locally) over a quote/description, then asks the LLM to name the
movie/show/video and a query to fetch the clip. Returns cited results. Never
identifies people — only the work.

    python -m app.source_id "you can't handle the truth"
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

from . import llm, websearch
from .logging_setup import get_logger

log = get_logger(__name__)

_SYSTEM = (
    "You identify the SOURCE of a movie/TV/video scene from a line of dialogue or a "
    "description. You identify the WORK (film/show/video), never a person. You are given "
    "web search results — rely on them; do not invent a title you cannot support.\n"
    "Reply with a fenced ```json block with exactly these keys:\n"
    '{"source_title": string|null, "scene_summary": string, '
    '"confidence": "high"|"medium"|"low", "search_terms": string, '
    '"candidate_video_query": string|null}\n'
    "If the results don't clearly identify a source: source_title=null, confidence=\"low\"."
)


def identify_source(query: str, metadata: Optional[dict] = None) -> dict:
    results = websearch.search(f"movie tv show scene quote {query}", max_results=5)
    user = f"Scene query / dialogue: {query}\n"
    if metadata:
        meta = ", ".join(f"{k}={v}" for k, v in metadata.items() if v)
        if meta:
            user += f"Known metadata: {meta}\n"
    user += "\nWeb search results:\n" + websearch.format_results(results)
    user += "\n\nIdentify the source and return the JSON."

    resp = llm.chat([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}])
    data = llm.extract_json(resp.text) or {}
    citations = [{"title": r["title"], "url": r["url"], "cited_text": r["snippet"][:200]} for r in results]
    result = {
        "source_title": data.get("source_title"),
        "scene_summary": (data.get("scene_summary") or "").strip(),
        "confidence": data.get("confidence", "low"),
        "search_terms": data.get("search_terms") or query,
        "candidate_video_query": data.get("candidate_video_query"),
        "citations": citations,
    }
    log.info("source_id: %s (confidence=%s)", result["source_title"], result["confidence"])
    return result


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Identify a scene's source")
    ap.add_argument("query")
    args = ap.parse_args()
    print(json.dumps(identify_source(args.query), indent=2))


if __name__ == "__main__":
    _cli()
