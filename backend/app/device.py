"""GPU auto-detection and sensible model-size defaults.

The dev box is an RTX 3050 Laptop (4 GB). Frames + Whisper are the memory-heavy
parts, so defaults stay modest and everything degrades to CPU cleanly.
"""
from __future__ import annotations

import functools
import os

from .logging_setup import get_logger

log = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def detect() -> dict:
    """Return a device profile: {device, compute_type, whisper_model, vram_gb}."""
    device = "cpu"
    compute_type = "int8"
    vram_gb = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # float16 is well-supported on Ampere; int8_float16 saves VRAM on 4GB.
            compute_type = "int8_float16" if vram_gb < 6 else "float16"
    except Exception as exc:  # torch missing / broken driver -> CPU
        log.warning("torch/CUDA probe failed (%s); using CPU", exc)

    # Whisper size: config/env override wins, else 'base' — fast and good enough
    # for moment location, and it leaves VRAM headroom for the local LLM.
    whisper_model = os.getenv("VERBATIM_WHISPER_MODEL", "") or "base"

    profile = {
        "device": device,
        "compute_type": compute_type,
        "whisper_model": whisper_model,
        "vram_gb": round(vram_gb, 2),
    }
    log.info("device profile: %s", profile)
    return profile


def torch_device() -> str:
    return detect()["device"]
