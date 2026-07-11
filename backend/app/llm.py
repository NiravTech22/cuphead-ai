"""llm.py — provider-agnostic LLM adapter with tool-use.

Switch providers with env var LLM_PROVIDER = ollama | anthropic | openai
(default: ollama, model LOCAL_MODEL, e.g. qwen2.5:7b — a solid small tool-caller).

The rest of the app speaks ONE format (OpenAI-style messages + tools) and never
sees provider differences. `chat()` returns a normalized `LLMResponse`:
  .text        assistant text
  .tool_calls  [{id, name, arguments(dict)}]

Message format used throughout the app (OpenAI-style):
  {"role": "system"|"user", "content": str}
  {"role": "assistant", "content": str,
   "tool_calls": [{"id": str, "function": {"name": str, "arguments": dict}}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Tools are OpenAI-style:
  {"type": "function", "function": {"name", "description", "parameters": <schema>}}
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .logging_setup import get_logger

log = get_logger(__name__)


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "ollama").lower()


def local_model() -> str:
    return os.getenv("LOCAL_MODEL", "qwen2.5:7b")


class LLMResponse:
    """Normalized response so the app never sees provider differences."""

    def __init__(self, text: str = "", tool_calls: Optional[list[dict]] = None):
        self.text = text
        self.tool_calls = tool_calls or []  # [{id, name, arguments(dict)}]


def chat(messages: list[dict], tools: Optional[list[dict]] = None) -> LLMResponse:
    p = provider()
    if p == "ollama":
        return _ollama(messages, tools)
    if p == "anthropic":
        return _anthropic(messages, tools)
    if p == "openai":
        return _openai(messages, tools)
    raise ValueError(f"Unknown LLM_PROVIDER: {p}")


def _norm_args(args) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {}
    return {}


# ---------- LOCAL (Ollama) ----------
_OLLAMA_PS_LOGGED = False


def _log_ollama_residency(model: str) -> None:
    """Log once where Ollama actually placed the model (GPU vs CPU spill)."""
    global _OLLAMA_PS_LOGGED
    if _OLLAMA_PS_LOGGED:
        return
    _OLLAMA_PS_LOGGED = True
    try:
        import ollama

        ps = ollama.ps()
        models = ps.get("models", []) if isinstance(ps, dict) else (getattr(ps, "models", None) or [])
        for m in models:
            get = m.get if isinstance(m, dict) else lambda k, d=None: getattr(m, k, d)
            name = get("name", "") or ""
            if model.split(":")[0] in name:
                size = get("size", 0) or 0
                vram = get("size_vram", 0) or 0
                pct = (100 * vram / size) if size else 0
                log.info("ollama model %s resident: %.0f%% on GPU (%.1f/%.1f GB)",
                         name, pct, vram / 1e9, size / 1e9)
    except Exception as exc:
        log.debug("ollama ps failed: %s", exc)


def _ollama(messages, tools) -> LLMResponse:
    import ollama

    from . import config

    model = local_model()
    # Ollama accepts OpenAI-style messages & tools; assistant tool_calls want
    # arguments as a dict, tool results as role="tool" with content.
    msgs = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            msgs.append(
                {
                    "role": "assistant",
                    "content": m.get("content", "") or "",
                    "tool_calls": [
                        {"function": {"name": tc["function"]["name"],
                                      "arguments": _norm_args(tc["function"]["arguments"])}}
                        for tc in m["tool_calls"]
                    ],
                }
            )
        elif m["role"] == "tool":
            msgs.append({"role": "tool", "content": m["content"],
                         "tool_name": m.get("name", "")})
        else:
            msgs.append({"role": m["role"], "content": m.get("content", "") or ""})

    resp = ollama.chat(
        model=model,
        messages=msgs,
        tools=tools or None,
        options={"temperature": 0.2},  # low temp = steadier tool output
        keep_alive=config.OLLAMA_KEEP_ALIVE,  # stay resident between calls
    )
    _log_ollama_residency(model)
    msg = resp["message"]
    calls = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc["function"]
        calls.append({"id": tc.get("id") or fn["name"],
                      "name": fn["name"], "arguments": _norm_args(fn["arguments"])})
    return LLMResponse(text=msg.get("content", "") or "", tool_calls=calls)


# ---------- HOSTED FALLBACK: Anthropic ----------
def _anthropic(messages, tools) -> LLMResponse:
    import anthropic

    from . import config

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY or None)
    sys = next((m["content"] for m in messages if m["role"] == "system"), None)

    # Translate OpenAI-style history -> Anthropic content blocks.
    conv: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "user":
            conv.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in (m.get("tool_calls") or []):
                blocks.append({
                    "type": "tool_use", "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": _norm_args(tc["function"]["arguments"]),
                })
            conv.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif role == "tool":
            tr = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            # merge consecutive tool results into one user turn
            if conv and conv[-1]["role"] == "user" and isinstance(conv[-1]["content"], list):
                conv[-1]["content"].append(tr)
            else:
                conv.append({"role": "user", "content": [tr]})

    a_tools = [{"name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"]["parameters"]} for t in (tools or [])]

    resp = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", config.ASSISTANT_MODEL),
        max_tokens=1500,
        system=sys or anthropic.NOT_GIVEN,
        messages=conv,
        tools=a_tools or anthropic.NOT_GIVEN,
    )
    text, calls = "", []
    for block in resp.content:
        if block.type == "text":
            text += block.text
        elif block.type == "tool_use":
            calls.append({"id": block.id, "name": block.name, "arguments": block.input})
    return LLMResponse(text=text, tool_calls=calls)


# ---------- HOSTED FALLBACK: OpenAI ----------
def _openai(messages, tools) -> LLMResponse:
    from openai import OpenAI

    client = OpenAI()
    # OpenAI wants tool_calls.arguments as JSON strings and tool msgs with tool_call_id.
    msgs = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            msgs.append({
                "role": "assistant",
                "content": m.get("content", "") or None,
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["function"]["name"],
                                  "arguments": json.dumps(_norm_args(tc["function"]["arguments"]))}}
                    for tc in m["tool_calls"]
                ],
            })
        elif m["role"] == "tool":
            msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            msgs.append({"role": m["role"], "content": m.get("content", "")})

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=msgs,
        tools=tools or None,
    )
    msg = resp.choices[0].message
    calls = []
    for tc in (msg.tool_calls or []):
        calls.append({"id": tc.id, "name": tc.function.name,
                      "arguments": _norm_args(tc.function.arguments)})
    return LLMResponse(text=msg.content or "", tool_calls=calls)


# ---------- JSON helper (used by source_id / verifier) ----------
def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model response (handles ```json fences)."""
    import re

    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None
