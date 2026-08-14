"""verifier.py — misquote correction. Verbatim's core feature.

Web-searches (locally) for the exact wording of a public statement, then asks
the LLM to return a structured, cited verdict. HARD RULE: hedge when
unverifiable, never fabricate a "corrected" line.

    python -m app.verifier "read my lips, no new taxes ever"
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

from . import llm, websearch
from .logging_setup import get_logger

log = get_logger(__name__)

_SYSTEM = (
    "You are a careful quote verifier. Using the provided web search results, find the "
    "EXACT line as it appears in the source and correct common misquotes.\n"
    "HARD RULES: never fabricate a line or source; if the results don't confirm it, say so.\n"
    "Reply with a fenced ```json block with exactly these keys:\n"
    '{"user_version": string, "exact_line": string|null, "source": string|null, '
    '"is_misquote": boolean, "confidence": "high"|"medium"|"low", "note": string}\n'
    "If unverifiable: exact_line=null, confidence=\"low\", explain in note."
)


def verify_quote(quote: str, source_context: Optional[str] = None) -> dict:
    q = f'exact quote "{quote}"'
    if source_context:
        q += f" {source_context}"
    results = websearch.search(q, max_results=5)
    user = f'User\'s remembered quote: "{quote}"\n'
    if source_context:
        user += f"Source context: {source_context}\n"
    user += "\nWeb search results:\n" + websearch.format_results(results)
    user += "\n\nVerify the exact line and return the JSON."

    resp = llm.chat([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}])
    data = llm.extract_json(resp.text) or {}
    citations = [{"title": r["title"], "url": r["url"], "cited_text": r["snippet"][:200]} for r in results]
    result = {
        "user_version": data.get("user_version", quote),
        "exact_line": data.get("exact_line"),
        "source": data.get("source"),
        "is_misquote": bool(data.get("is_misquote", False)),
        "confidence": data.get("confidence", "low"),
        "note": (data.get("note") or "").strip() or "Could not verify confidently.",
        "citation": citations[0] if citations else None,
        "citations": citations,
        "in_video_timestamp": None,
    }
    log.info("verify: misquote=%s confidence=%s", result["is_misquote"], result["confidence"])
    return result


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Verify / correct a quote")
    ap.add_argument("quote")
    ap.add_argument("--source", default=None)
    args = ap.parse_args()
    print(json.dumps(verify_quote(args.quote, args.source), indent=2))


if __name__ == "__main__":
    _cli()
