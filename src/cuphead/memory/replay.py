"""The replay record format: one attempt, frame-indexed, on disk.

Implements the acceptance criterion for task ``harness-replay-format``: a
round trip preserves frames, frame-indexed actions, events, HUD ground
truth, outcome, loadout and game build with zero differences, and the
loader raises on a misaligned action.

Layout on disk, one directory per attempt::

    <root>/<run_id>/
      meta.json       loadout, game build, fps, outcome, frame count
      actions.jsonl   one line per frame: {"frame": i, "action": {...}}
      events.json     the EventTrace, as produced by events.schema
      hud.jsonl       sparse per-frame HUD ground truth samples
      frames/         written by the capture pipeline, not by this module --
                       a replay references frame indices, not pixel data

Frame *content* is deliberately out of scope here. Embedding pixels would
require a video codec, which does not belong in a stdlib-only layer, and
every consumer of a replay (the evaluation harness, the world model's data
loader) reads frames by index against whatever `capture` wrote alongside
this record. What this module owns is the one invariant everything else
depends on: **every action is logged against the frame index it was
intended for, with no gaps and no repeats.** `ReplayWriter.finalize` checks
this fail-fast, at the end of the session that produced it; `ReplayReader`
checks it again on load, as defense against a file corrupted after the
fact. Violating it is the single most common silent killer of a world
model (see `docs/SPEEDRUN_PLAN.md` section 2.4).

Writes are atomic per file, following the same temp-then-``os.replace``
pattern as ``orchestration.state_store`` -- a crash mid-write must never
leave a half-written replay for a training run to silently ingest.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..events.schema import EventTrace


class ReplayAlignmentError(RuntimeError):
    """Raised when the action log does not have exactly one entry per frame."""


class ReplayError(RuntimeError):
    """Raised for any other structural problem with a replay on disk."""


def _assert_alignment(seen_frames: List[int], frame_count: int) -> None:
    """Shared by the writer (fail fast at ``finalize``) and the reader
    (defense in depth against a file corrupted after the fact -- see
    ``ReplayReader._assert_alignment``'s caller)."""
    if frame_count <= 0:
        return
    expected = list(range(frame_count))
    if seen_frames != expected:
        missing = sorted(set(expected) - set(seen_frames))
        extra = sorted(set(seen_frames) - set(expected))
        duplicated = sorted({f for f in seen_frames if seen_frames.count(f) > 1})
        detail = []
        if missing:
            detail.append(f"missing frames {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if extra:
            detail.append(f"unexpected frames {extra[:5]}{'...' if len(extra) > 5 else ''}")
        if duplicated:
            detail.append(f"duplicated frames {duplicated[:5]}{'...' if len(duplicated) > 5 else ''}")
        raise ReplayAlignmentError(
            f"action log is misaligned with frame_count={frame_count}: " + "; ".join(detail)
        )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class ReplayMeta:
    run_id: str
    boss: str
    loadout: Dict[str, str]
    game_build: str
    fps: float
    outcome: str
    frame_count: int
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "boss": self.boss,
            "loadout": self.loadout,
            "game_build": self.game_build,
            "fps": self.fps,
            "outcome": self.outcome,
            "frame_count": self.frame_count,
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ReplayMeta":
        return cls(
            run_id=raw["run_id"],
            boss=raw["boss"],
            loadout=raw.get("loadout", {}),
            game_build=raw.get("game_build", "unknown"),
            fps=float(raw.get("fps", 60.0)),
            outcome=raw.get("outcome", "INCOMPLETE"),
            frame_count=int(raw.get("frame_count", 0)),
            extras=raw.get("extras", {}),
        )


@dataclass(frozen=True)
class Replay:
    """A fully materialized replay, as returned by ``ReplayReader.read``."""

    meta: ReplayMeta
    actions: List[Dict[str, Any]]  # [{"frame": i, "action": {...}}, ...], frame-sorted
    events: EventTrace
    hud: List[Dict[str, Any]]  # [{"frame": i, ...hud fields}, ...]


class ReplayWriter:
    """Builds one replay directory, one attempt at a time.

    Usage::

        w = ReplayWriter(root, run_id="run_00001", boss="goopy_le_grande", ...)
        for frame_index, action in demonstration:
            w.log_action(frame_index, action.to_buttons())
        w.log_hud(frame_index, hp=3, super_cards=1.5, weapon="peashooter")
        w.finalize(events=trace, outcome="KNOCKOUT", frame_count=n)

    Actions are appended to ``actions.jsonl`` as they are logged, so a
    crashed recording session leaves a usable partial log rather than
    nothing -- ``finalize`` is what makes the replay complete and readable
    by ``ReplayReader``; an unfinalized directory is not a valid replay and
    is rejected.
    """

    def __init__(
        self,
        root: Path,
        *,
        run_id: str,
        boss: str,
        loadout: Optional[Dict[str, str]] = None,
        game_build: str = "unknown",
        fps: float = 60.0,
    ) -> None:
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "frames").mkdir(exist_ok=True)
        self.run_id = run_id
        self.boss = boss
        self.loadout = dict(loadout or {})
        self.game_build = game_build
        self.fps = fps
        self._actions_fh = open(self.dir / "actions.jsonl", "a", encoding="utf-8")
        self._hud_fh = open(self.dir / "hud.jsonl", "a", encoding="utf-8")
        self._logged_frames: set[int] = set()
        self._finalized = False

    def log_action(self, frame: int, action: Dict[str, Any]) -> None:
        if frame in self._logged_frames:
            raise ReplayAlignmentError(f"frame {frame} logged twice in the same writer session")
        self._logged_frames.add(frame)
        self._actions_fh.write(json.dumps({"frame": frame, "action": action}) + "\n")
        self._actions_fh.flush()

    def log_hud(self, frame: int, **fields: Any) -> None:
        self._hud_fh.write(json.dumps({"frame": frame, **fields}) + "\n")
        self._hud_fh.flush()

    def finalize(self, *, events: EventTrace, outcome: str, frame_count: int, extras: Optional[Dict[str, Any]] = None) -> ReplayMeta:
        if self._finalized:
            raise ReplayError(f"replay {self.run_id!r} already finalized")
        _assert_alignment(sorted(self._logged_frames), frame_count)
        self._actions_fh.close()
        self._hud_fh.close()

        meta = ReplayMeta(
            run_id=self.run_id,
            boss=self.boss,
            loadout=self.loadout,
            game_build=self.game_build,
            fps=self.fps,
            outcome=outcome,
            frame_count=frame_count,
            extras=extras or {},
        )
        _atomic_write_text(self.dir / "meta.json", json.dumps(meta.to_dict(), indent=2) + "\n")
        _atomic_write_text(self.dir / "events.json", json.dumps(events.to_dict(), indent=2) + "\n")
        self._finalized = True
        return meta

    def abandon(self) -> None:
        """Close file handles without finalizing -- for a session that errors out.

        Leaves the partial directory on disk (for post-mortem) but it will
        never satisfy ``ReplayReader``, which requires ``meta.json``.
        """
        if not self._actions_fh.closed:
            self._actions_fh.close()
        if not self._hud_fh.closed:
            self._hud_fh.close()


class ReplayReader:
    """Loads and validates one replay directory.

    ``read()`` is the only method that matters: it enforces frame-action
    alignment (every index in ``[0, frame_count)`` present exactly once, in
    order) and raises ``ReplayAlignmentError`` the moment that is violated,
    rather than handing back a silently corrupted action sequence.
    """

    def __init__(self, root: Path, run_id: str) -> None:
        self.dir = Path(root) / run_id
        if not (self.dir / "meta.json").exists():
            raise ReplayError(f"no finalized replay at {self.dir} (missing meta.json)")

    def read(self) -> Replay:
        meta = ReplayMeta.from_dict(json.loads((self.dir / "meta.json").read_text(encoding="utf-8")))
        actions = self._read_jsonl(self.dir / "actions.jsonl")
        hud = self._read_jsonl(self.dir / "hud.jsonl")
        events = EventTrace.from_dict(json.loads((self.dir / "events.json").read_text(encoding="utf-8")))

        actions.sort(key=lambda a: a["frame"])
        _assert_alignment([a["frame"] for a in actions], meta.frame_count)

        return Replay(meta=meta, actions=actions, events=events, hud=hud)

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        out = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


def write_replay(
    root: Path,
    *,
    run_id: str,
    boss: str,
    frames_and_actions: Iterable[tuple[int, Dict[str, Any]]],
    events: EventTrace,
    outcome: str,
    loadout: Optional[Dict[str, str]] = None,
    game_build: str = "unknown",
    fps: float = 60.0,
    hud_samples: Iterable[tuple[int, Dict[str, Any]]] = (),
) -> ReplayMeta:
    """Convenience one-shot writer for a complete, already-collected attempt."""
    writer = ReplayWriter(root, run_id=run_id, boss=boss, loadout=loadout, game_build=game_build, fps=fps)
    frame_count = 0
    try:
        for frame, action in frames_and_actions:
            writer.log_action(frame, action)
            frame_count = max(frame_count, frame + 1)
        for frame, fields in hud_samples:
            writer.log_hud(frame, **fields)
        return writer.finalize(events=events, outcome=outcome, frame_count=frame_count)
    except BaseException:
        writer.abandon()
        raise


def read_replay(root: Path, run_id: str) -> Replay:
    return ReplayReader(root, run_id).read()
