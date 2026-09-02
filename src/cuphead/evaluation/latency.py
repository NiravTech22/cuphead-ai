"""Round-trip actuate -> observe latency measurement.

Backs the acceptance criterion for task ``harness-capture-latency``: p50 < 30 ms
and p99 < 50 ms over 200 trials, written to ``experiments/`` as a JSON record.
See ``docs/SPEEDRUN_PLAN.md`` section 2.1 -- every other number in the project
is meaningless until this passes, because it bounds what the 60 Hz reflex
layer and the 15 Hz planner can actually react to.

The measurement itself is backend-agnostic: send an action, then read frames
until one differs in content from the frame captured immediately before the
send, and report the gap between the send timestamp and that frame's own
capture timestamp. It works identically against the real screen-capture and
virtual-gamepad backends in ``perception.capture`` / ``control.actuator`` and
against the synthetic pair in this module, because both sides speak the same
``FrameSource`` / ``Actuator`` protocols.

The synthetic pair (``RespondingFrameSource`` + ``NotifyingActuator`` +
``FakeClock``) exists to validate the *measurement code*, not to stand in for
real latency evidence -- it drives a known, injected response delay in frame
counts rather than real time, so the test suite recovers it without sleeping.
A report produced in synthetic mode is tagged accordingly and is never
sufficient on its own to close the harness-capture-latency task; that
requires ``--real`` on the machine actually running Cuphead.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

from ..control.action_space import Action
from ..control.actuator import Actuation, Actuator
from ..perception.capture import Frame, FrameSource, frame_checksum


class LatencyTimeoutError(RuntimeError):
    """Raised when a trial sees no content change within the frame budget."""


def percentile(samples: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, no numpy required.

    The gate cares about p50 and p99 specifically because a mean hides
    exactly the tail spikes that break a 5-frame parry window.
    """
    if not samples:
        raise ValueError("percentile of an empty sample set is undefined")
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class LatencyReport:
    n: int
    p50_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    samples_ms: List[float] = field(default_factory=list)

    def within_budget(self, *, p50_ms: float, p99_ms: float) -> bool:
        return self.p50_ms < p50_ms and self.p99_ms < p99_ms

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "p50_ms": self.p50_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
            "max_ms": self.max_ms,
            "samples_ms": self.samples_ms,
        }


def measure_once(actuator: Actuator, frame_source: FrameSource, action: Action, *, max_frames: int = 600) -> float:
    """One actuate -> observe trial. Returns latency in milliseconds.

    ``max_frames`` bounds the search so a backend that never visibly responds
    (a misconfigured region, a paused game, a synthetic source wired up
    wrong) fails loudly with ``LatencyTimeoutError`` instead of hanging.
    """
    baseline = frame_source.read()
    act = actuator.send(action)
    for _ in range(max_frames):
        frame = frame_source.read()
        if frame.checksum != baseline.checksum:
            return (frame.t_capture - act.t_sent) * 1000.0
    raise LatencyTimeoutError(f"no visible response within {max_frames} frames of actuation")


def measure_many(
    actuator: Actuator,
    frame_source: FrameSource,
    actions: Sequence[Action],
    *,
    max_frames_per_trial: int = 600,
) -> LatencyReport:
    samples = [
        measure_once(actuator, frame_source, action, max_frames=max_frames_per_trial) for action in actions
    ]
    return LatencyReport(
        n=len(samples),
        p50_ms=percentile(samples, 50),
        p99_ms=percentile(samples, 99),
        mean_ms=statistics.fmean(samples),
        max_ms=max(samples),
        samples_ms=samples,
    )


# -- synthetic self-test backend -----------------------------------------


class FakeClock:
    """A monotonically advancing clock with no real waiting.

    Every call advances by exactly ``dt`` and returns the new value, so a
    tight measurement loop recovers a deterministic elapsed time without the
    test suite sleeping for real milliseconds.
    """

    def __init__(self, dt: float = 1.0 / 60.0) -> None:
        self.dt = dt
        self.t = 0.0

    def __call__(self) -> float:
        self.t += self.dt
        return self.t


class RespondingFrameSource:
    """Synthetic ``FrameSource`` whose content changes N frames after the most
    recent ``notify_action_sent()`` call, and is otherwise static.

    This exists only to validate ``measure_once``/``measure_many`` against a
    *known* injected delay (in frame counts, paired with a ``FakeClock`` for
    the time axis) -- it is not a stand-in for real latency evidence.
    """

    def __init__(self, respond_after_frames: int, clock: Callable[[], float]) -> None:
        if respond_after_frames < 1:
            raise ValueError("respond_after_frames must be >= 1")
        self._respond_after = respond_after_frames
        self._clock = clock
        self._index = 0
        self._stimulus = 0
        self._frames_since_send: int | None = None

    def notify_action_sent(self) -> None:
        self._frames_since_send = 0

    def read(self) -> Frame:
        if self._frames_since_send is not None:
            self._frames_since_send += 1
            if self._frames_since_send >= self._respond_after:
                self._stimulus += 1
                self._frames_since_send = None
        payload = f"stimulus-{self._stimulus}"
        frame = Frame(index=self._index, t_capture=self._clock(), payload=payload, checksum=frame_checksum(payload))
        self._index += 1
        return frame

    def close(self) -> None:
        pass


class NotifyingActuator:
    """Pairs with ``RespondingFrameSource``: every send starts its response timer."""

    def __init__(self, source: RespondingFrameSource, clock: Callable[[], float]) -> None:
        self._source = source
        self._clock = clock
        self.history: List[Actuation] = []

    def send(self, action: Action) -> Actuation:
        self._source.notify_action_sent()
        act = Actuation(action=action, t_sent=self._clock())
        self.history.append(act)
        return act

    def close(self) -> None:
        pass


def synthetic_round_trip(*, respond_after_frames: int = 2, fps: float = 60.0) -> tuple[NotifyingActuator, RespondingFrameSource]:
    """A matched actuator/source pair with a known, fixed response delay.

    Used by ``scripts/latency_canary.py --synthetic`` and by this module's
    own tests. Not real evidence -- see the module docstring.
    """
    clock = FakeClock(dt=1.0 / fps)
    source = RespondingFrameSource(respond_after_frames, clock)
    actuator = NotifyingActuator(source, clock)
    return actuator, source
