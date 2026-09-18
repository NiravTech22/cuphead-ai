"""Optional, bounded CUDA inference runtime; importing this module needs no torch.

The engine owns every staging/output buffer and a private copy of the memory bank.
The caller transfers exclusive use of the encoder until ``close()``: never move,
train, mutate, or call it concurrently. Setup/capture and close may synchronize;
``try_submit`` and ``poll`` never wait for GPU completion. Use one CPU owner thread.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn


@dataclass(frozen=True)
class InferenceResult:
    """Owned CPU values; safe to retain after a slot is reused.

    ``action_id`` is suppressed for invalid embeddings or missed deadlines.
    End-to-end latency includes CPU staging, GPU queueing, H2D, graph, D2H and
    polling delay. Supply a capture timestamp to include capture/preprocessing.
    GPU timings measure stream intervals, not CPU dispatch or polling overhead.
    """

    sequence: int
    frame_id: int
    action_id: int | None
    neighbor_ids: tuple[int, ...]
    similarities: tuple[float, ...]
    valid_embedding: bool
    deadline_missed: bool
    latency_ms: float
    h2d_ms: float
    graph_ms: float
    d2h_ms: float


@dataclass
class _Slot:
    host_input: Tensor
    device_input: Tensor
    graph: torch.cuda.CUDAGraph
    outputs: tuple[Tensor, ...]
    host_outputs: tuple[Tensor, ...]
    copy_start: torch.cuda.Event
    copy_done: torch.cuda.Event
    graph_start: torch.cuda.Event
    graph_done: torch.cuda.Event
    read_start: torch.cuda.Event
    done: torch.cuda.Event
    sequence: int = -1
    frame_id: int = -1
    submitted_at: float = 0.0


class RingBufferedGPURecordedAgentEngine:
    """One manual CUDA graph per slot, with independent H2D/compute/D2H streams.

    ``encoder`` must be an eager, graph-safe nn.Module returning floating [1,D]
    latents from a fixed input. It must use PyTorch's current stream and must not
    mutate shared state. All inference (encoder, normalization, cosine top-k,
    similarity-weighted action vote) is captured. Compute runs serially, allowing
    next-slot H2D to overlap current-slot compute without concurrent model writes.
    Separate graph pools prevent one slot overwriting another's pending output.

    Inputs and labels are copied at setup; bank tensors stay in VRAM. ``keys``
    must be nonzero finite [N,D] embeddings in the encoder's exact feature space.
    ``action_ids`` are [N] nonnegative int64 labels. Use a single scenario-specific
    bank; this imitation vote is not the existing action-conditioned planner.

    The default 33ms deadline is strict (<, not <=). It is a measured deadline,
    not a hardware guarantee or a GPU cancellation mechanism. Full rings reject
    submissions immediately; stale results never produce an actionable ID.
    Actual transfer overlap requires hardware copy engines and enough bandwidth.
    Capture must run while no other thread is issuing CUDA work in this process.
    """

    def __init__(
        self,
        encoder: nn.Module,
        keys: Tensor,
        action_ids: Tensor,
        *,
        input_shape: tuple[int, ...],
        input_dtype: torch.dtype | None = None,
        device: str = "cuda:0",
        slots: int = 2,
        k: int = 8,
        budget_ms: float = 33.0,
        temperature: float = 0.1,
        compile_model: bool = True,
        warmup_steps: int = 3,
    ) -> None:
        if isinstance(slots, bool) or not isinstance(slots, int) or slots < 2:
            raise ValueError("slots must be an integer >= 2")
        if not input_shape or input_shape[0] != 1 or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in input_shape
        ):
            raise ValueError("input_shape must be positive static dimensions with batch 1")
        if not math.isfinite(budget_ms) or not 0 < budget_ms <= 33:
            raise ValueError("budget_ms must be finite and in (0, 33]")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 3:
            raise ValueError("at least three warmup steps required")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA-enabled PyTorch build and NVIDIA GPU are required")
        self._torch: Any = torch
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError("device must be CUDA")
        if self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        if not isinstance(encoder, torch.nn.Module) or hasattr(encoder, "_orig_mod"):
            raise TypeError("pass an eager nn.Module; this engine owns compilation")
        if keys.ndim != 2 or min(keys.shape) < 1 or not keys.is_floating_point():
            raise ValueError("keys must be nonempty floating [N,D]")
        if action_ids.shape != (keys.shape[0],) or action_ids.dtype != torch.int64:
            raise ValueError("action_ids must be int64 [N]")
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= keys.shape[0]:
            raise ValueError("k must be an integer in [1,N]")
        self._shape = tuple(input_shape)
        self._dtype = input_dtype if input_dtype is not None else torch.uint8
        if self._dtype not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("unsupported input dtype")
        self._owner = threading.get_ident()
        self._closed = False
        self._faulted = False
        self._pending: deque[int] = deque()
        self._next_slot = 0
        self._sequence = 0
        self._budget_ms = budget_ms
        self._k = k
        self._temperature = temperature
        self._slots: list[_Slot] = []

        # Lifecycle synchronization is intentional: caller-created weights/banks
        # may have outstanding work on other streams before ownership transfers.
        with torch.cuda.device(self._device), torch.inference_mode():
            torch.cuda.synchronize(self._device)
            self._keys = keys.detach().to(device=self._device, dtype=torch.float32).clone()
            norms = self._keys.norm(dim=1, keepdim=True)
            if not bool(torch.isfinite(self._keys).all()) or not bool(
                torch.isfinite(norms).all() & (norms > 1e-12).all()
            ):
                raise ValueError("bank rows must be finite with nonzero finite norms")
            self._keys.div_(norms)
            self._actions = action_ids.detach().to(self._device).clone()
            if int(self._actions.min()) < 0:
                raise ValueError("action IDs must be nonnegative")
            # Compress arbitrary labels, avoiding allocation proportional to max ID.
            self._labels, self._action_columns = torch.unique(
                self._actions, sorted=True, return_inverse=True
            )
            self._encoder = encoder.to(self._device).eval().requires_grad_(False)
            self._copy_stream = torch.cuda.Stream(device=self._device)
            self._compute_stream = torch.cuda.Stream(device=self._device)
            self._read_stream = torch.cuda.Stream(device=self._device)
            self._compute_stream.wait_stream(torch.cuda.current_stream(self._device))
            self._run: Any = self._inference
            if compile_model:
                self._run = torch.compile(
                    self._inference, fullgraph=True, dynamic=False,
                    options={"triton.cudagraphs": False},
                )
            try:
                self._capture(slots, warmup_steps)
            except BaseException:
                # Retain buffers until any partially submitted setup work finishes.
                torch.cuda.synchronize(self._device)
                raise

    def _inference(self, pixels: Tensor) -> tuple[Tensor, ...]:
        torch = self._torch
        latent = self._encoder(pixels).float()
        valid = torch.isfinite(latent).all() & (latent.norm() > 1e-12) & torch.isfinite(latent.norm())
        query = torch.nn.functional.normalize(torch.nan_to_num(latent), dim=1, eps=1e-12)
        similarities, indices = (query @ self._keys.T).topk(self._k, dim=1, sorted=True)
        weights = (similarities / self._temperature).softmax(dim=1)
        votes = torch.zeros((1, self._labels.numel()), device=pixels.device, dtype=torch.float32)
        votes.scatter_add_(1, self._action_columns[indices], weights)
        action = self._labels[votes.argmax(dim=1)]
        return indices, similarities, action, valid.reshape(1)

    def _capture(self, count: int, warmup_steps: int) -> None:
        """Allocate pinned storage once; compile and warm up before each capture."""
        torch = self._torch
        for _ in range(count):
            host = torch.zeros(self._shape, dtype=self._dtype, pin_memory=True)
            with torch.cuda.stream(self._compute_stream):
                static_input = torch.zeros(self._shape, device=self._device, dtype=self._dtype)
                probe = self._encoder(static_input)
                if not isinstance(probe, torch.Tensor) or probe.shape != (1, self._keys.shape[1]):
                    raise ValueError("encoder must return [1,D] matching bank width")
                if probe.device != self._device or not probe.is_floating_point():
                    raise ValueError("encoder must return floating latents on the selected GPU")
                for _ in range(warmup_steps):
                    self._run(static_input)
            self._compute_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            # Do not share pools: D2H for slot i may overlap replay of slot i+1.
            with torch.cuda.graph(graph, stream=self._compute_stream):
                outputs = self._run(static_input)
            host_outputs = tuple(torch.empty_like(t, device="cpu", pin_memory=True) for t in outputs)
            # Instantiate/upload the executable graph and touch transfer paths
            # during setup, not in the latency-sensitive first submission.
            with torch.cuda.stream(self._compute_stream):
                static_input.copy_(host, non_blocking=True)
                graph.replay()
                for destination, source in zip(host_outputs, outputs):
                    destination.copy_(source, non_blocking=True)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            # Materialize lazy event handles in setup rather than on first submit.
            for event in events:
                event.record(self._compute_stream)
            self._slots.append(_Slot(host, static_input, graph, outputs, host_outputs, *events))
        torch.cuda.synchronize(self._device)

    def _check_owner(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("engine methods must run on the constructing CPU thread")
        if self._closed:
            raise RuntimeError("engine is closed")
        if self._faulted:
            raise RuntimeError("CUDA submission failed; close and recreate the engine")

    @property
    def capacity(self) -> int:
        """Fixed ring size; queue capacity never grows under load."""
        self._check_owner()
        return len(self._slots)

    @property
    def in_flight(self) -> int:
        """Number of submitted results not yet consumed; no CUDA calls."""
        self._check_owner()
        return len(self._pending)

    def try_submit(
        self, frame: Tensor, *, frame_id: int, captured_at: float | None = None,
    ) -> int | None:
        """Copy a CPU frame into a free slot, enqueue work, return its sequence.

        Returns None on backpressure. The caller may reuse ``frame`` on return:
        its CPU-to-CPU staging copy is synchronous, while H2D is asynchronous.
        Never pass GPU tensors, mismatched layouts or mutate frame concurrently.
        ``captured_at`` uses time.perf_counter(), not Unix or device timestamps.
        No internal buffer is lent to callers, so a pending DMA source cannot be
        overwritten accidentally. No unbounded GPU submission queue is created.
        """
        self._check_owner()
        torch = self._torch
        if not isinstance(frame, torch.Tensor):
            raise TypeError("frame must be a CPU tensor")
        if frame.device.type != "cpu" or tuple(frame.shape) != self._shape or frame.dtype != self._dtype:
            raise ValueError("frame must match the configured CPU shape and dtype")
        if not frame.is_contiguous() or frame.requires_grad:
            raise ValueError("frame must be contiguous with requires_grad=False")
        now = time.perf_counter()
        if captured_at is not None and (not math.isfinite(captured_at) or captured_at > now):
            raise ValueError("captured_at must be a finite past perf_counter timestamp")
        if len(self._pending) == len(self._slots):
            return None
        slot_index = self._next_slot
        slot = self._slots[slot_index]
        slot.submitted_at = now if captured_at is None else captured_at
        slot.frame_id = frame_id
        slot.sequence = self._sequence
        try:
            with torch.cuda.device(self._device), torch.inference_mode():
                slot.host_input.copy_(frame)
                with torch.cuda.stream(self._copy_stream):
                    slot.copy_start.record()
                    slot.device_input.copy_(slot.host_input, non_blocking=True)
                    slot.copy_done.record()
                with torch.cuda.stream(self._compute_stream):
                    self._compute_stream.wait_event(slot.copy_done)
                    slot.graph_start.record()
                    slot.graph.replay()
                    slot.graph_done.record()
                with torch.cuda.stream(self._read_stream):
                    self._read_stream.wait_event(slot.graph_done)
                    slot.read_start.record()
                    for destination, source in zip(slot.host_outputs, slot.outputs):
                        destination.copy_(source, non_blocking=True)
                    slot.done.record()
        except BaseException:
            self._faulted = True
            raise
        self._pending.append(slot_index)
        self._next_slot = (slot_index + 1) % len(self._slots)
        self._sequence += 1
        return slot.sequence

    def poll(self) -> InferenceResult | None:
        """Consume the oldest completed result using only a nonblocking query.

        Read pinned outputs only after D2H's completion event is ready. Convert
        to owned Python values before releasing the slot. CPU tensor reads here
        do not synchronize CUDA; all timing events have already completed.
        Call frequently: slow polling is included in the end-to-end deadline.
        """
        self._check_owner()
        if not self._pending:
            return None
        slot = self._slots[self._pending[0]]
        if not slot.done.query():
            return None
        indices, similarities, action, valid = slot.host_outputs
        neighbor_ids = tuple(int(i) for i in indices[0].tolist())
        scores = tuple(float(s) for s in similarities[0].tolist())
        valid_embedding = bool(valid[0])
        action_id = int(action[0]) if valid_embedding else None
        h2d_ms = slot.copy_start.elapsed_time(slot.copy_done)
        graph_ms = slot.graph_start.elapsed_time(slot.graph_done)
        d2h_ms = slot.read_start.elapsed_time(slot.done)
        latency_ms = (time.perf_counter() - slot.submitted_at) * 1000
        missed = latency_ms >= self._budget_ms
        result = InferenceResult(
            slot.sequence, slot.frame_id, None if missed else action_id,
            neighbor_ids, scores, valid_embedding, missed, latency_ms,
            h2d_ms, graph_ms, d2h_ms,
        )
        self._pending.popleft()
        return result

    def close(self) -> None:
        """Lifecycle-only blocking drain. Discard pending results and release graphs.

        Always call in finally or use a context manager. No __del__: destructor
        timing/thread affinity is unsafe for explicit CUDA resource ownership.
        """
        if threading.get_ident() != self._owner:
            raise RuntimeError("close must run on the constructing CPU thread")
        if self._closed:
            return
        # Even a failed submission may have partially enqueued DMA or replay.
        self._copy_stream.synchronize()
        self._compute_stream.synchronize()
        self._read_stream.synchronize()
        self._pending.clear()
        self._slots.clear()
        self._run = None
        self._encoder = None
        self._keys = self._actions = self._labels = self._action_columns = None
        self._closed = True

    def __enter__(self) -> RingBufferedGPURecordedAgentEngine:
        self._check_owner()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
