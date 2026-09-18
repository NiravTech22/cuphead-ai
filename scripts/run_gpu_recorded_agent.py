#!/usr/bin/env python3
"""Runnable CUDA ring benchmark using synthetic clips; never actuates the game.

Use --repository and --checkpoint together to benchmark actual frozen V-JEPA
2.1. Without them this is an explicitly synthetic tiny encoder smoke test.
Banks are random benchmark fixtures, not learned game behavior.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cuphead.orchestration.gpu_recorded_agent import (
    InferenceResult, RingBufferedGPURecordedAgentEngine,
)

if TYPE_CHECKING:
    from torch import Tensor


def active_loop(
    engine: RingBufferedGPURecordedAgentEngine,
    next_frame: Callable[[], tuple[Tensor, int, float] | None],
    consume: Callable[[InferenceResult], None],
    *,
    count: int,
) -> None:
    """Bounded active loop; producer must return immediately or report no frame.

    Real integration supplies a latest-frame mailbox from a capture worker.
    No frames are fetched when the ring is full, preventing a stale CPU queue.
    ``consume`` must also be quick and handle action_id=None by neutral/release
    according to the existing controller policy. It must not act on neighbor IDs.
    No tensor/device synchronization or blocking capture occurs in this loop.
    """
    submitted = completed = 0
    while completed < count:
        result = engine.poll()
        if result is not None:
            consume(result)
            completed += 1
        if submitted < count and engine.in_flight < engine.capacity:
            frame = next_frame()
            if frame is not None:
                pixels, frame_id, captured_at = frame
                sequence = engine.try_submit(pixels, frame_id=frame_id, captured_at=captured_at)
                if sequence is not None:
                    submitted += 1
        # Yield the CPU scheduler; this is not a CUDA wait. A dedicated core can
        # omit this at the expense of CPU utilization. Measure polling jitter.
        if result is None:
            time.sleep(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--variant", choices=("base", "large", "giant", "gigantic"), default="base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument("--clip-frames", type=int, default=4)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if bool(args.repository) != bool(args.checkpoint):
        parser.error("--repository and --checkpoint must be provided together")
    if args.frames < 1 or args.bank_size < 8 or args.clip_frames < 2 or args.clip_frames % 2:
        parser.error("frames >= 1, bank-size >= 8 and positive even clip-frames required")
    import torch

    dimension = 384
    if args.repository:
        from cuphead.perception.frozen_jepa import FrozenJEPAEncoder
        frozen = FrozenJEPAEncoder(
            repository=args.repository, checkpoint=args.checkpoint,
            variant=args.variant, device=args.device, quantization="none",
            clip_frames=args.clip_frames, output_dim=dimension,
        )
        encoder = frozen.cuda_graph_module(precision=args.precision)
    else:
        class TinyEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = torch.nn.Linear(3, dimension)

            def forward(self, pixels: torch.Tensor) -> torch.Tensor:
                return self.projection(pixels.float().mean(dim=(2, 3, 4)) / 255.0)

        encoder = TinyEncoder()
    shape = (1, 3, args.clip_frames, 256, 256)
    generator = torch.Generator().manual_seed(17)
    keys = torch.randn(args.bank_size, dimension, generator=generator)
    labels = torch.arange(args.bank_size, dtype=torch.int64) % 8
    # Pre-generated clips isolate engine overhead from capture/preprocessing.
    clips = [torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator) for _ in range(4)]
    frame_id = 0

    def next_frame() -> tuple[torch.Tensor, int, float]:
        nonlocal frame_id
        frame_id += 1
        return clips[frame_id % len(clips)], frame_id, time.perf_counter()

    results: list[InferenceResult] = []
    with RingBufferedGPURecordedAgentEngine(
        encoder, keys, labels, input_shape=shape, device=args.device,
        slots=args.slots, compile_model=not args.no_compile,
    ) as engine:
        started = time.perf_counter()
        active_loop(engine, next_frame, results.append, count=args.frames)
        elapsed = time.perf_counter() - started
    latencies = sorted(r.latency_ms for r in results)
    report = {
        "encoder": "vjepa2.1" if args.repository else "synthetic_tiny",
        "input_and_bank": "synthetic_benchmark_only",
        "frames": len(results), "slots": args.slots,
        "throughput_fps": len(results) / elapsed,
        "median_latency_ms": statistics.median(latencies),
        "p99_latency_ms": latencies[min(len(latencies) - 1, int(0.99 * len(latencies)))],
        "max_latency_ms": max(latencies),
        "median_graph_ms": statistics.median(r.graph_ms for r in results),
        "deadline_misses": sum(r.deadline_missed for r in results),
        "invalid_embeddings": sum(not r.valid_embedding for r in results),
        "budget_passed": all(not r.deadline_missed and r.valid_embedding for r in results),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["budget_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
