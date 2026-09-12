"""Perception layer — capture, HUD reading, the pink parry prior, latent encoder.

Owner: the ``vision`` agent. See ``docs/SPEEDRUN_PLAN.md`` section 3.

Contracts this package must provide:

    grab() -> Frame                     capture, <= 4 ms, frame-counter stamped
    read_hud(frame) -> HudReading       template match, exact, <= 1 ms
    pink_mask(frame) -> Mask            HSV threshold, <= 1 ms
    encode(frames) -> latent            CNN/ViT, 256-d, <= 5 ms

Two design commitments, both load-bearing:

* The HUD is deterministic pixels, so it is read by template matching at 100%
  accuracy and never by a network. It is also the project's free label source —
  HP and card counts ground the encoder through auxiliary heads with no hand
  labelling anywhere.
* Every parryable object in Cuphead is pink by design. That invariant turns the
  hardest timing problem in the game into an HSV threshold.

Latency is a correctness property here, not a performance note: a perception
change that blows the budget is a regression even if its accuracy improves.

numpy/torch are imported lazily so the rest of the project stays importable on a
bare machine.
"""

from .capture import (
    Frame,
    FrameIntegrityError,
    FrameSource,
    IntegrityReport,
    IntegrityTracker,
    SyntheticFrameSource,
    frame_checksum,
    open_screen_source,
)

__all__ = [
    "Frame",
    "FrameIntegrityError",
    "FrameSource",
    "IntegrityReport",
    "IntegrityTracker",
    "SyntheticFrameSource",
    "frame_checksum",
    "open_screen_source",
]

from .frozen_jepa import FrozenJEPAEncoder
from .latent_encoder import FrozenVisualEncoder, LatentEncoder

__all__ += ['FrozenJEPAEncoder', 'FrozenVisualEncoder', 'LatentEncoder']
