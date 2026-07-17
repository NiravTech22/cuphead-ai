"""emotion_data.py — the single emotion-data contract (see ADR-010).

One file per cached video: data/emotions/<videoId>.emotions.json
    { videoId, source, fps_sampled,
      points: [{t, emotions: {angry..neutral}, dominant, intensity}] }

`intensity` = 1 - P(neutral): how emotionally charged the moment is,
regardless of which emotion carries it. Documented choice (ADR-010).

Generator ("transcript-affect-v1"): emotion is scored from the dialogue —
each transcript segment is embedded with the EXISTING CPU MiniLM
(knowledge.embedder, zero VRAM) and compared to anchor sentences per
emotion; softmax over similarities gives the distribution. The facial
`fer` detector, when installed, can write the same contract; consumers
never care which generator produced the file.

    python -m app.emotion_data --backfill        # all cached videos
    python -m app.emotion_data <videoId>         # one video
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .knowledge.embedder import embed
from .logging_setup import get_logger

log = get_logger(__name__)

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SOURCE = "transcript-affect-v1"
_SOFTMAX_T = 0.06          # sharpens tiny cosine gaps into a usable distribution
MAX_POINTS = 200           # accessor render cap per window

# Anchor sentences per emotion — deliberately spoken-dialogue flavored.
_ANCHORS: dict[str, list[str]] = {
    "angry": ["I am furious at you.", "How dare you do this!",
              "This is a threat and I want you gone."],
    "disgust": ["That is disgusting and vile.", "This makes me sick to look at."],
    "fear": ["I'm terrified, please don't hurt me.", "Something is wrong, we need to run.",
             "I'm scared of what happens next."],
    "happy": ["I'm so happy to see you!", "This is wonderful news, we did it!",
              "I love this, what a joy."],
    "sad": ["I miss you so much it hurts.", "I'm sorry. I couldn't save them.",
            "I feel so alone and heartbroken."],
    "surprise": ["What?! I can't believe it!", "No way — that's impossible!"],
    "neutral": ["Here is the schedule for tomorrow.", "The car is parked outside.",
                "Please pass the report to the office."],
}


def path_for(video_id: str) -> Path:
    return config.EMOTIONS_DIR / f"{video_id}.emotions.json"


@functools.lru_cache(maxsize=1)
def _anchor_matrix() -> tuple[np.ndarray, list[str]]:
    """(A, 384) matrix of anchor embeddings + parallel emotion labels."""
    labels, texts = [], []
    for emo, sentences in _ANCHORS.items():
        for s in sentences:
            labels.append(emo)
            texts.append(s)
    return embed(texts), labels


def classify(texts: list[str]) -> list[dict[str, float]]:
    """Emotion distribution per text (softmax over max-anchor cosine sims)."""
    if not texts:
        return []
    vecs = embed(texts)                            # (N, 384), normalized
    anchors, labels = _anchor_matrix()
    sims = vecs @ anchors.T                        # (N, A)
    out = []
    for row in sims:
        per_emo = {e: max(row[i] for i, l in enumerate(labels) if l == e)
                   for e in EMOTIONS}
        vals = np.array([per_emo[e] for e in EMOTIONS], dtype=np.float64)
        exps = np.exp((vals - vals.max()) / _SOFTMAX_T)
        probs = exps / exps.sum()
        out.append({e: round(float(p), 4) for e, p in zip(EMOTIONS, probs)})
    return out


def _point(t: float, dist: dict[str, float]) -> dict:
    dominant = max(dist, key=dist.get)
    return {"t": round(t, 2), "emotions": dist, "dominant": dominant,
            "intensity": round(1.0 - dist.get("neutral", 0.0), 4)}


def build_for_video(video_id: str, force: bool = False) -> Optional[dict]:
    """Generate <videoId>.emotions.json from the cached transcript. CPU only."""
    out_path = path_for(video_id)
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())
    video = config.DOWNLOADS_DIR / f"{video_id}.mp4"
    if not video.is_file():
        log.warning("emotion_data: no cached video for %s", video_id)
        return None
    from .transcriber import transcribe          # cache-hit only in backfill use
    transcript = transcribe(str(video))
    segs = [(s.start, s.end, s.text.strip()) for s in transcript.segments
            if s.text.strip() and len(s.text.strip()) > 2]
    if not segs:
        log.warning("emotion_data: empty transcript for %s", video_id)
        return None
    dists = classify([t for _, _, t in segs])
    points = [_point((a + b) / 2, d) for (a, b, _), d in zip(segs, dists)]
    duration = max(transcript.duration, points[-1]["t"] + 1)
    data = {
        "videoId": video_id,
        "source": SOURCE,
        "fps_sampled": round(len(points) / duration, 4),
        "duration": round(duration, 2),
        "points": points,
    }
    out_path.write_text(json.dumps(data))
    log.info("emotions.json written: %s (%d points, %.0fs)",
             video_id, len(points), duration)
    return data


def get_timeline(video_id: str, start: Optional[float] = None,
                 end: Optional[float] = None, max_points: int = MAX_POINTS) -> Optional[dict]:
    """Plot-ready series for a window: evenly resampled (linear interp over the
    emotion vectors), capped at `max_points`."""
    try:
        data = json.loads(path_for(video_id).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    pts = data["points"]
    if not pts:
        return None
    t0 = data["points"][0]["t"] if start is None else float(start)
    t1 = data.get("duration", pts[-1]["t"]) if end is None else float(end)
    if t1 <= t0:
        t1 = t0 + 1.0
    n = int(min(max_points, max(40, (t1 - t0) * 4)))
    ts = np.array([p["t"] for p in pts])
    mat = np.array([[p["emotions"][e] for e in EMOTIONS] for p in pts])
    sample_ts = np.linspace(t0, t1, n)
    out_points = []
    for st in sample_ts:
        row = np.array([np.interp(st, ts, mat[:, i]) for i in range(len(EMOTIONS))])
        row = row / max(row.sum(), 1e-9)
        dist = {e: round(float(v), 4) for e, v in zip(EMOTIONS, row)}
        out_points.append(_point(float(st), dist))
    return {"videoId": video_id, "source": data["source"], "start": round(t0, 2),
            "end": round(t1, 2), "points": out_points}


def backfill(force: bool = False) -> int:
    n = 0
    for mp4 in sorted(config.DOWNLOADS_DIR.glob("*.mp4")):
        if build_for_video(mp4.stem, force=force):
            n += 1
    log.info("emotion backfill complete: %d/%d videos",
             n, len(list(config.DOWNLOADS_DIR.glob('*.mp4'))))
    return n


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Build/inspect emotions.json files")
    ap.add_argument("video_id", nargs="?")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.backfill:
        print(f"backfilled {backfill(force=args.force)} videos")
    elif args.video_id:
        d = build_for_video(args.video_id, force=args.force)
        print(json.dumps((d or {}).get("points", [])[:5], indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli()
