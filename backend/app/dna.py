"""dna.py — Scene DNA share card (1200×630 PNG, download-only, no hosting).

Composes, server-side with Pillow (ADR-013): the clip's key frame (subtly
desaturated, espresso gradient overlay), the quote in serif italic, source +
timestamp in tracked small-caps, the Phase-0 emotion waveform as the "DNA"
strip, and the wordmark. Palette = the site's dark tokens; no new colors.
The quote passes the content filter before it is drawn.
"""
from __future__ import annotations

import io
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import config, emotion_data
from . import filter as content_filter
from .logging_setup import get_logger

log = get_logger(__name__)

W, H = 1200, 630
S = 2                                   # supersample factor for crisp downscale
BG = (22, 19, 16)                       # --bg (dark)
FG = (236, 230, 218)                    # --fg
DIM = (151, 141, 125)                   # --dim
ACCENT = (181, 160, 129)                # --link
LINE = (43, 38, 32)                     # --line

SERIF_IT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"
SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _keyframe(source_path: str, t: float) -> Optional[Image.Image]:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        out = tmp.name
    cp = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", source_path, "-frames:v", "1",
         "-vf", f"scale={W}:-2,eq=saturation=0.72", "-q:v", "3", out],
        capture_output=True)
    try:
        if cp.returncode != 0:
            return None
        img = Image.open(out).convert("RGB")
    finally:
        Path(out).unlink(missing_ok=True)
    # cover-crop to W×H
    ratio = max(W / img.width, H / img.height)
    img = img.resize((round(img.width * ratio), round(img.height * ratio)),
                     Image.LANCZOS)
    x = (img.width - W) // 2
    y = (img.height - H) // 2
    return img.crop((x, y, x + W, y + H))


def _tracked(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, tracking: int):
    """Letter-spaced small-caps-style text; returns end x."""
    x, y = xy
    for ch in text.upper():
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking
    return x


def _wrap_quote(draw, text: str, font, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render_card(clip_id: str, title: Optional[str] = None) -> bytes:
    sidecar = json.loads((config.CLIPS_DIR / f"{clip_id}.json").read_text())
    src = sidecar["source_path"]
    vid = Path(src).stem
    start, end = float(sidecar["start"]), float(sidecar["end"])
    quote = (sidecar.get("quote") or "").strip()
    if quote and content_filter.check(quote):
        quote = ""                                   # never draw blocked text
    if title and content_filter.check(title):
        title = None
    if not title:
        try:
            info = json.loads((config.DOWNLOADS_DIR / f"{vid}.info.json").read_text())
            title = info.get("title", "Scene Sense")[:60]
        except (FileNotFoundError, json.JSONDecodeError):
            title = "Scene Sense"

    frame = _keyframe(src, (start + end) / 2)
    card = Image.new("RGB", (W, H), BG)
    if frame:
        card.paste(frame, (0, 0))
    # espresso gradient: readable lower half, cinematic top vignette
    overlay = Image.new("L", (1, H))
    for yy in range(H):
        a = int(60 if yy < H * 0.35 else 60 + (yy / H - 0.35) ** 1.1 * 300)
        overlay.putpixel((0, yy), min(a, 242))
    overlay = overlay.resize((W, H))
    card = Image.composite(Image.new("RGB", (W, H), BG), card, overlay)

    d = ImageDraw.Draw(card)
    MX = 84                                          # left margin

    # quote — serif italic, adaptive size, ≤3 lines
    if quote:
        qtext = f"“{quote[:140]}”"
        size = 54
        while size > 30:
            lines = _wrap_quote(d, qtext, _font(SERIF_IT, size), W - 2 * MX)
            if len(lines) <= 3:
                break
            size -= 6
        f = _font(SERIF_IT, size)
        y = 388 - len(lines) * (size + 12)
        for ln in lines:
            d.text((MX, y), ln, font=f, fill=FG)
            y += size + 12
    # metadata line — tracked caps
    t0 = start
    meta = f"{title}  ·  at {int(t0 // 60)}:{int(t0 % 60):02d}"
    emo = sidecar.get("dominant_emotion")
    if emo:
        meta += f"  ·  {emo}"
    _tracked(d, (MX, 412), meta[:76], _font(SANS, 17), DIM, 3)

    # DNA strip — the emotion waveform, accent on a hairline baseline
    # ±10s of context so the strip shows the scene's contour, autoscaled so
    # a uniformly intense clip still reads as a shape rather than a slab
    tl = emotion_data.get_timeline(vid, max(0.0, start - 10), end + 10)
    gx0, gx1, gy0, gy1 = MX, W - MX, 468, 552
    d.line([(gx0, gy1), (gx1, gy1)], fill=LINE, width=1)
    if tl and len(tl["points"]) >= 8:
        pts = tl["points"]
        ints = [p["intensity"] for p in pts]
        lo, hi = min(ints), max(ints)
        rng = max(hi - lo, 0.08)
        big = Image.new("RGBA", ((gx1 - gx0) * S, (gy1 - gy0) * S), (0, 0, 0, 0))
        bd = ImageDraw.Draw(big)
        n = len(pts)
        xy = [((i / (n - 1)) * big.width,
               big.height - 8 * S - ((p["intensity"] - lo) / rng) * (big.height - 18 * S))
              for i, p in enumerate(pts)]
        poly = [(x, big.height) for x, _ in xy[:1]] + xy + [(xy[-1][0], big.height)]
        bd.polygon(poly, fill=ACCENT + (46,))
        bd.line(xy, fill=ACCENT + (255,), width=S + 1, joint="curve")
        strip = big.resize((gx1 - gx0, gy1 - gy0), Image.LANCZOS)
        card.paste(strip, (gx0, gy0), strip)
        _tracked(d, (gx0, gy1 + 10), "scene dna — emotional intensity",
                 _font(SANS, 12), DIM, 2)

    # wordmark + restrained botanical dot ornament
    wm = "Scene Sense"
    fwm = _font(SERIF, 26)
    wm_w = d.textlength(wm, font=fwm)
    d.text((W - MX - wm_w, H - 62), wm, font=fwm, fill=FG)
    cx, cy = W - MX - wm_w - 26, H - 62 + 17
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], outline=ACCENT, width=1)
    d.line([(cx - 16, cy), (cx - 6, cy)], fill=ACCENT, width=1)

    buf = io.BytesIO()
    card.save(buf, "PNG")
    log.info("dna card rendered: %s (%d bytes)", clip_id, buf.tell())
    return buf.getvalue()
