"""Episodic memory — attempts, failures, and distilled lessons.

Owner: the ``data`` agent. See ``docs/SPEEDRUN_PLAN.md`` section 8.

Three stores:

* structured index over (boss, phase, loadout, outcome, ttk) for exact filters;
* embedding index over event traces, answering "have we seen this failure before?";
* distilled lessons — clusters of similar failures compacted into a short written
  finding with an evidence count.

The strategist reads the lessons, not the raw traces. A hundred raw traces will
not fit in context; ten lessons with counts will, and they carry more signal.
"""

from .replay import (
    Replay,
    ReplayAlignmentError,
    ReplayError,
    ReplayMeta,
    ReplayReader,
    ReplayWriter,
    read_replay,
    write_replay,
)

__all__ = [
    "Replay",
    "ReplayAlignmentError",
    "ReplayError",
    "ReplayMeta",
    "ReplayReader",
    "ReplayWriter",
    "read_replay",
    "write_replay",
]

from .latent_bank import Experience, LatentBank, Neighbor

__all__ += ['Experience', 'LatentBank', 'Neighbor']
