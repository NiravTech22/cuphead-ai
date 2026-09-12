"""Bounded, persistent, action-conditional episodic transition memory (stdlib)."""

from __future__ import annotations

import heapq
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

Latent = tuple[float, ...]


def checked_latent(values: Sequence[float], dimension: int) -> Latent:
    result = tuple(float(x) for x in values)
    if len(result) != dimension or not all(math.isfinite(x) for x in result):
        raise ValueError("latent has wrong dimension or non-finite values")
    return result


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("latent dimensions differ")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass(frozen=True)
class Experience:
    state: Latent
    action: str
    next_state: Latent
    scope: str
    reward: float = 0.0
    hit: bool = False
    terminal: bool = False
    frame: int = 0
    next_frame: int = 1
    elapsed: float = 1 / 15


@dataclass(frozen=True)
class Neighbor:
    distance: float
    experience: Experience


class LatentBank:
    def __init__(self, dimension: int, encoder_id: str, *, capacity: int = 4096):
        if dimension < 1 or capacity < 1 or not encoder_id:
            raise ValueError(
                "positive dimension/capacity and encoder identity required"
            )
        self.dimension, self.encoder_id, self.capacity = dimension, encoder_id, capacity
        self._rows: list[Experience] = []

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, experience: Experience) -> None:
        row = experience
        checked_latent(row.state, self.dimension)
        checked_latent(row.next_state, self.dimension)
        if not row.action or not row.scope:
            raise ValueError("action and scenario scope are required")
        if row.next_frame <= row.frame or row.frame < 0:
            raise ValueError("transition frames must advance")
        if (
            not math.isfinite(row.reward)
            or not math.isfinite(row.elapsed)
            or row.elapsed <= 0
        ):
            raise ValueError("reward and positive elapsed time must be finite")
        # Canonicalize caller-owned lists so subsequent mutation cannot change the bank.
        self._rows.append(
            Experience(
                checked_latent(row.state, self.dimension),
                row.action,
                checked_latent(row.next_state, self.dimension),
                row.scope,
                float(row.reward),
                bool(row.hit),
                bool(row.terminal),
                row.frame,
                row.next_frame,
                row.elapsed,
            )
        )
        if len(self._rows) > self.capacity:
            del self._rows[: len(self._rows) - self.capacity]

    def lookup(
        self,
        state: Sequence[float],
        action: str,
        scope: str,
        *,
        k: int = 5,
        radius: float = 0.25,
    ) -> tuple[Neighbor, ...]:
        state = checked_latent(state, self.dimension)
        if k < 1 or not math.isfinite(radius) or radius < 0:
            raise ValueError("positive k and finite nonnegative radius required")
        found = []
        for index, row in enumerate(self._rows):
            if row.action != action or row.scope != scope:
                continue
            d = distance(state, row.state)
            if d <= radius:
                found.append((d, index, row))
        return tuple(Neighbor(d, row) for d, _, row in heapq.nsmallest(k, found))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "dimension": self.dimension,
            "encoder_id": self.encoder_id,
            "capacity": self.capacity,
            "rows": [asdict(row) for row in self._rows],
        }
        name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, delete=False
            ) as stream:
                name = stream.name
                json.dump(data, stream, allow_nan=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)

    @classmethod
    def load(cls, path: str | Path, *, encoder_id: str) -> LatentBank:
        data = json.loads(Path(path).read_text())
        if data.get("version") != 1 or data.get("encoder_id") != encoder_id:
            raise ValueError(
                "bank version or encoder fingerprint mismatch; re-encode memory"
            )
        bank = cls(data["dimension"], encoder_id, capacity=data["capacity"])
        for row in data["rows"]:
            bank.add(Experience(**row))
        return bank
