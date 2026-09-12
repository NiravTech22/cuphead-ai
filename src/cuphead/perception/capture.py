"""Frame capture with a monotonic counter and duplicate/drop detection.

Implements the acceptance criterion for task ``harness-frame-integrity``:
zero duplicate and zero dropped frames over a 10-minute capture, with the
detector unit-tested against a synthetic stream with injected gaps and
repeats.

A repeated frame is the most dangerous failure mode here, not a dropped one.
A drop is loud -- a gap in the frame index -- but a duplicate is silent: the
capture pipeline hands back the same pixels twice with two different frame
indices, and every downstream consumer reads it as "the world did nothing
for a frame," which is a fake transition that quietly poisons world-model
training data. So duplicate detection here is content-based (a cheap
checksum), not just index-based -- an index can advance correctly while the
backend still hands back a stale buffer.

Real backends (mss for screen capture) import lazily so this module -- and
everything that depends only on its abstractions -- stays importable with
nothing installed. ``SyntheticFrameSource`` is the stdlib-only source tests
and dry runs use in place of a real screen.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Protocol


@dataclass(frozen=True)
class Frame:
    """One captured frame.

    ``index`` is the source's own monotonic counter -- the thing frame-action
    alignment is checked against. ``t_capture`` is a ``time.perf_counter()``
    timestamp, good for latency measurement but never for alignment (clocks
    drift; indices don't). ``payload`` is left untyped here: a real backend
    fills it with pixel data (an ndarray); the synthetic source fills it with
    a plain Python object, which is all the integrity logic needs.
    """

    index: int
    t_capture: float
    payload: object
    checksum: int
    width: int | None = None
    height: int | None = None


def frame_checksum(payload: object) -> int:
    """A cheap, deterministic fingerprint of frame content.

    CRC32 over a stable byte representation. Good enough to catch "the same
    buffer handed back twice" -- it is not a security hash and does not need
    to be.
    """
    if isinstance(payload, (bytes, bytearray)):
        data = bytes(payload)
    elif hasattr(payload, "tobytes"):
        data = payload.tobytes()  # type: ignore[attr-defined]
    else:
        data = repr(payload).encode("utf-8")
    return zlib.crc32(data)


class FrameSource(Protocol):
    """What every capture backend provides: one frame per call, blocking."""

    def read(self) -> Frame: ...

    def close(self) -> None: ...


class SyntheticFrameSource:
    """Deterministic, stdlib-only frame source for tests and dry runs.

    Produces a monotonically increasing counter and content that changes
    every frame by default, so ``IntegrityTracker`` sees a clean stream
    unless duplicates/drops are explicitly injected -- which is exactly what
    the acceptance criterion asks the detector to be tested against.
    """

    def __init__(
        self,
        *,
        fps: float = 60.0,
        duplicate_at: frozenset[int] = frozenset(),
        drop_at: frozenset[int] = frozenset(),
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._dt = 1.0 / fps
        self._duplicate_at = duplicate_at
        self._drop_at = drop_at
        self._clock = clock
        self._next_index = 0
        self._last_frame: Optional[Frame] = None
        self._closed = False

    def read(self) -> Frame:
        if self._closed:
            raise RuntimeError("read() on a closed FrameSource")

        idx = self._next_index
        # A drop means the backend silently skipped emitting an index --
        # from the consumer's perspective the counter jumps.
        if idx in self._drop_at:
            self._next_index += 1
            idx = self._next_index

        if idx in self._duplicate_at and self._last_frame is not None:
            frame = Frame(
                index=idx,
                t_capture=self._clock(),
                payload=self._last_frame.payload,
                checksum=self._last_frame.checksum,
            )
        else:
            payload = f"frame-{idx}"
            frame = Frame(index=idx, t_capture=self._clock(), payload=payload, checksum=frame_checksum(payload))

        self._last_frame = frame
        self._next_index = idx + 1
        return frame

    def close(self) -> None:
        self._closed = True


class FrameIntegrityError(RuntimeError):
    """Raised when the capture stream has dropped or duplicated a frame."""


@dataclass
class IntegrityReport:
    frames_seen: int = 0
    duplicates: int = 0
    drops: int = 0

    @property
    def clean(self) -> bool:
        return self.duplicates == 0 and self.drops == 0


class IntegrityTracker:
    """Wraps a ``FrameSource``, detecting duplicate and dropped frames.

    Two independent checks, because they fail differently:

    * **Drop** -- the index does not advance by exactly 1. Loud and cheap:
      pure bookkeeping on the counter.
    * **Duplicate** -- the content checksum repeats even though the index
      advanced correctly. This is the dangerous one (see module docstring)
      and is why detection here is checksum-based, not index-based.

    By default a violation raises immediately, because a replay recorded
    over a corrupted stream is worse than no replay -- the world model would
    train on it and nobody would notice until hit-prediction AUC plateaus
    for no visible reason. Pass ``strict=False`` to instead accumulate a
    report, for tooling that wants to characterize *how* degraded a stream
    is (e.g. reporting duplicate-frame rate as a harness health metric).
    """

    def __init__(self, source: FrameSource, *, strict: bool = True) -> None:
        self._source = source
        self.strict = strict
        self.report = IntegrityReport()
        self._last_index: Optional[int] = None
        self._last_checksum: Optional[int] = None

    def read(self) -> Frame:
        frame = self._source.read()
        self.report.frames_seen += 1

        if self._last_index is not None:
            gap = frame.index - self._last_index
            if gap == 0:
                self._fail(f"frame index {frame.index} repeated")
            elif gap < 0:
                self._fail(f"frame index went backwards: {self._last_index} -> {frame.index}")
            elif gap > 1:
                self.report.drops += gap - 1
                if self.strict:
                    self._fail(f"dropped {gap - 1} frame(s) between index {self._last_index} and {frame.index}")

        if self._last_checksum is not None and frame.checksum == self._last_checksum and (
            self._last_index is None or frame.index != self._last_index
        ):
            self.report.duplicates += 1
            if self.strict:
                self._fail(f"frame {frame.index} duplicates the content of frame {self._last_index}")

        self._last_index = frame.index
        self._last_checksum = frame.checksum
        return frame

    def _fail(self, reason: str) -> None:
        raise FrameIntegrityError(reason)

    def close(self) -> None:
        self._source.close()

    def stream(self, n: Optional[int] = None) -> Iterator[Frame]:
        """Yield frames until ``n`` is reached or the source is exhausted."""
        count = 0
        while n is None or count < n:
            yield self.read()
            count += 1


def open_screen_source(*, monitor: int = 1, region: Optional[dict] = None) -> FrameSource:
    """Real screen capture via ``mss``, imported lazily.

    This is the backend a training session on the actual game uses. It is
    not exercised by the test suite -- there is no display in CI -- so its
    correctness rests on ``SyntheticFrameSource`` sharing the exact same
    ``FrameSource`` contract, which every test in this module exercises.
    """
    try:
        import mss  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "mss is not installed. Run `pip install -r requirements.txt` on the "
            "machine actually running Cuphead before using the real capture backend."
        ) from exc

    class _MSSFrameSource:
        def __init__(self) -> None:
            self._sct = mss.mss()
            self._monitor = region or self._sct.monitors[monitor]
            self._index = 0
            self._last_checksum: Optional[int] = None

        def read(self) -> Frame:
            shot = self._sct.grab(self._monitor)
            payload = shot.rgb
            frame = Frame(
                index=self._index,
                t_capture=time.perf_counter(),
                payload=payload,
                checksum=frame_checksum(payload),
                width=shot.width,
                height=shot.height,
            )
            self._index += 1
            return frame

        def close(self) -> None:
            self._sct.close()

    return _MSSFrameSource()
