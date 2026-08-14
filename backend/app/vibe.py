"""vibe.py — search the PROCESSED library by rhetorical tone (CPU, cache only).

"a moment that feels defiant" → top moments from already-cached videos.
Score = W_TEXT * cosine(query emb, scene-transcript emb)
            + W_EMO  * cosine(feeling target, scene emotion profile),
where the feeling target comes from backend/feelings.json (editable) — any
feeling words present in the query contribute their emotion vectors. If the
query names no known feeling word, scoring is text-only (W_EMO mass shifts
to text). Defaults documented in ADR-012.

Fingerprints (scene windows ≈ 22s of consecutive transcript segments, each
with a MiniLM embedding + mean emotion vector from emotions.json) are built
once per process and cached to data/vibe_index.json.
"""
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from . import filter as content_filter
from .emotion_data import EMOTIONS, path_for
from .knowledge.embedder import embed
from .logging_setup import get_logger

log = get_logger(__name__)

W_TEXT = 0.6           # semantic weight (default, ADR-012)
W_EMO = 0.4            # emotion-profile weight when feeling words matched
THRESHOLD = 0.42       # blended floor — below it we honestly report no match
SCENE_SECONDS = 22.0   # target scene-window length
FEELINGS_PATH = config.BACKEND_DIR / "feelings.json"
INDEX_PATH = config.DATA_DIR / "vibe_index.json"
THUMBS_DIR = config.DATA_DIR / "vibe_thumbs"
THUMBS_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_index: Optional[list[dict]] = None      # [{video_id,title,start,end,text,emb,emo}]


def _load_feelings() -> dict[str, dict[str, float]]:
    try:
        data = json.loads(FEELINGS_PATH.read_text())
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _video_title(video_id: str) -> str:
    try:
        info = json.loads((config.DOWNLOADS_DIR / f"{video_id}.info.json").read_text())
        return info.get("title", video_id)
    except (FileNotFoundError, json.JSONDecodeError):
        return video_id


def _scene_windows(video_id: str) -> list[dict]:
    """Group consecutive transcript segments into ~SCENE_SECONDS windows."""
    from .transcriber import transcribe
    video = config.DOWNLOADS_DIR / f"{video_id}.mp4"
    transcript = transcribe(str(video))
    scenes, cur, cur_start = [], [], None
    for s in transcript.segments:
        text = s.text.strip()
        if not text:
            continue
        if cur_start is None:
            cur_start = s.start
        cur.append(text)
        if s.end - cur_start >= SCENE_SECONDS:
            scenes.append({"start": cur_start, "end": s.end, "text": " ".join(cur)})
            cur, cur_start = [], None
    if cur and cur_start is not None:
        scenes.append({"start": cur_start,
                       "end": transcript.segments[-1].end, "text": " ".join(cur)})
    return [s for s in scenes if len(s["text"]) > 24]


def _emotion_profile(video_id: str, start: float, end: float) -> Optional[list[float]]:
    try:
        pts = json.loads(path_for(video_id).read_text())["points"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None
    rows = [[p["emotions"][e] for e in EMOTIONS] for p in pts if start <= p["t"] <= end]
    if not rows:
        return None
    return list(np.mean(np.array(rows), axis=0))


def _build_index() -> list[dict]:
    if INDEX_PATH.exists():
        try:
            idx = json.loads(INDEX_PATH.read_text())
            if idx:
                log.info("vibe index loaded: %d scenes", len(idx))
                return idx
        except json.JSONDecodeError:
            pass
    idx: list[dict] = []
    for mp4 in sorted(config.DOWNLOADS_DIR.glob("*.mp4")):
        vid = mp4.stem
        if not path_for(vid).exists():        # emotion contract required
            continue
        title = _video_title(vid)
        scenes = _scene_windows(vid)
        if not scenes:
            continue
        embs = embed([s["text"] for s in scenes])
        for s, e in zip(scenes, embs):
            emo = _emotion_profile(vid, s["start"], s["end"])
            if emo is None:
                continue
            idx.append({"video_id": vid, "title": title,
                        "start": round(s["start"], 2), "end": round(s["end"], 2),
                        "text": s["text"][:400], "emb": [round(float(x), 5) for x in e],
                        "emo": [round(float(x), 4) for x in emo]})
    INDEX_PATH.write_text(json.dumps(idx))
    log.info("vibe index built: %d scenes across %d videos",
             len(idx), len({s['video_id'] for s in idx}))
    return idx


def _get_index() -> list[dict]:
    global _index
    with _lock:
        if _index is None:
            _index = _build_index()
        return _index


def _feeling_target(query: str) -> Optional[np.ndarray]:
    q = query.lower()
    target = np.zeros(len(EMOTIONS))
    hit = False
    for word, vec in _load_feelings().items():
        if word in q:
            hit = True
            for e, w in vec.items():
                if e in EMOTIONS:
                    target[EMOTIONS.index(e)] += w
    if not hit:
        return None
    n = np.linalg.norm(target)
    return target / n if n > 0 else None


def _thumb(video_id: str, t: float) -> Optional[str]:
    name = f"{video_id}_{int(t)}.jpg"
    out = THUMBS_DIR / name
    if not out.exists():
        cp = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{t:.2f}",
             "-i", str(config.DOWNLOADS_DIR / f"{video_id}.mp4"),
             "-frames:v", "1", "-vf", "scale=320:-2,eq=saturation=0.8",
             "-q:v", "5", str(out)], capture_output=True)
        if cp.returncode != 0:
            return None
    return f"/vibe/{name}"


def search(query: str, k: int = 3) -> dict:
    """Top-k scenes matching a feeling. Honest empty state below THRESHOLD."""
    idx = _get_index()
    if not idx:
        return {"results": [], "note": "no processed videos with emotion data yet"}
    q_emb = embed([query])[0]
    target = _feeling_target(query)
    scored = []
    for s in idx:
        text_sim = float(np.dot(q_emb, np.array(s["emb"])))
        if target is not None:
            emo = np.array(s["emo"])
            n = np.linalg.norm(emo)
            emo_sim = float(np.dot(target, emo / n)) if n > 0 else 0.0
            score = W_TEXT * text_sim + W_EMO * emo_sim
        else:
            score = text_sim
        scored.append((score, text_sim, s))
    scored.sort(key=lambda x: -x[0])
    results = []
    for score, text_sim, s in scored:
        if score < THRESHOLD or len(results) >= k:
            break
        if any(r["video_id"] == s["video_id"] for r in results):
            continue                     # variety: one scene per video
        emo = np.array(s["emo"])
        top_emo = EMOTIONS[int(np.argmax(emo))]
        why = f"high {top_emo}, {'feeling-matched' if target is not None else 'semantic'} " \
              f"dialogue — {int(round(score * 100))}% vibe match"
        snippet = s["text"][:70].rsplit(" ", 1)[0]
        if (content_filter.check(s["title"]) or content_filter.check(why)
                or content_filter.check(snippet)):
            continue
        results.append({
            "video_id": s["video_id"], "title": s["title"],
            "start": s["start"], "end": s["end"],
            "why": why, "score": round(score, 3), "snippet": snippet,
            "thumb_url": _thumb(s["video_id"], (s["start"] + s["end"]) / 2),
        })
    if not results:
        return {"results": [],
                "note": "no processed moment clears the vibe threshold — "
                        "process more videos to widen the library"}
    log.info("vibe: %r -> %s", query,
             [(r["title"][:30], r["score"]) for r in results])
    return {"results": results}
