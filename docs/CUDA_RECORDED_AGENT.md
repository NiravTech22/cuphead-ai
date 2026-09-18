# Ring-buffered CUDA inference

`RingBufferedGPURecordedAgentEngine` in
[`gpu_recorded_agent.py`](../src/cuphead/orchestration/gpu_recorded_agent.py)
is an optional GPU runtime for a frozen encoder and a fixed GPU-resident cosine
kNN bank. The default ring has two slots. It does not replace the existing
scenario verifier, action-conditioned consequence planner or controller.

## Scheduling and ownership

Each slot owns pinned CPU input/output tensors, static GPU input/output tensors,
six timing/completion events and a manually captured `torch.cuda.CUDAGraph`.
Pinned memory is allocated once with `pin_memory=True` (the allocation-time
equivalent of `Tensor.pin_memory()`), never inside the active loop.

```text
H2D stream:     input A -------- input B -------- input A(next)
                    |                |                  |
compute stream:     graph A ----------graph B -----------graph A(next)
                         |                  |
D2H stream:              results A          results B
                              |                  |
CPU:                    query/read/free A  query/read/free B
```

The three streams are independent; compute waits only on the corresponding
input-copy event, and readback waits only on the corresponding graph event.
`poll()` queries the D2H completion event, then reads CPU values and releases
the slot. Input A cannot be overwritten until A's result has been consumed.
Graph allocations use separate pools because a subsequent graph can run while
a previous slot's outputs are still being copied to the CPU. Sharing pools in
this schedule could overwrite pending results.

The bank is cloned, converted to FP32 and normalized once in VRAM. All-row cosine
similarity, top-k and softmax-weighted action voting run inside the graph. Only
top-k indices/scores, an action ID and an embedding-validity flag leave the GPU.
Sparse action IDs are compressed internally. A zero/NaN/infinite embedding
suppresses the action. Ties follow PyTorch top-k semantics; neighbor ordering
among exact ties is not promised.

The caller retains its source frame and bank tensors and may reuse them after
the relevant call returns. No engine-owned tensor is exposed. The engine has
exclusive use of the model while open; do not mutate weights, run training,
move the model, or run a concurrent forward. The model must be stateless in eval
mode, use current-stream CUDA operations and have no data-dependent Python
branches or CPU reads. Do not pass an already compiled model.

Construction, warmup, graph capture and `close()` may block. Compilation uses
`fullgraph=True`, `dynamic=False`, `options={"triton.cudagraphs": False}`.
Unsupported capture/compilation fails setup rather than silently falling back.
`compile_model=False` still uses manual CUDA graphs. Capture requires no other
threads issuing CUDA work in the process. All engine calls use one CPU owner
thread. A partially failed submission faults the engine: close and recreate it.

## V-JEPA 2.1 integration

Use the existing strict pretrained checkpoint loader:

```python
from cuphead.perception import FrozenJEPAEncoder
from cuphead.orchestration.gpu_recorded_agent import RingBufferedGPURecordedAgentEngine

frozen = FrozenJEPAEncoder(
    repository="checkpoints/vjepa2",
    checkpoint="checkpoints/vjepa2_1_vitb_dist_vitG_384.pt",
    variant="base", quantization="none", device="cuda:0",
    output_dim=384, clip_frames=4,
)
adapter = frozen.cuda_graph_module(precision="float32")
# keys: floating [N,384], action_ids: int64 [N], one verified scenario.
# These are supplied by your application, not fabricated action labels.
with RingBufferedGPURecordedAgentEngine(
    adapter, keys, action_ids,
    input_shape=(1, 3, 4, 256, 256), slots=2,
) as engine:
    # The active_loop example below handles submission and completion.
    active_loop(engine, next_frame, consume, count=300)
```

Import `active_loop` from the runnable
[`run_gpu_recorded_agent.py`](../scripts/run_gpu_recorded_agent.py) example, or
copy that small loop into your entry point. `next_frame` returns immediately
with `(cpu_tensor, frame_id, perf_counter_capture_timestamp)` or `None` when no
frame is ready. `consume` receives an immutable `InferenceResult`; execute only
valid, timely `action_id` values and apply the existing controller's neutral
policy otherwise. A capture worker should publish only its latest owned frame;
do not block the polling thread on `source.read()` or accumulate a CPU queue.
Always release controller input in the application's `finally` block.

The adapter expects RGB uint8 `[1,3,T,H,W]` clips, assembled and resized by the
producer. It performs ImageNet normalization, mean token pooling and the existing
seeded projection on GPU. Use a separate engine for image `[1,3,1,128,128]` and
video `[1,3,4,256,256]` branches; shape changes require recapture. Do not mix their
memory banks. Record checkpoint/source fingerprints, adapter precision,
preprocessing, projection seed and branch shape with a real bank. The engine
can validate tensor dimensions, but cannot prove their semantic compatibility.

The adapter shares the loader's backbone and changes its dtype when requested;
treat that loader as transferred to the adapter and recreate it before returning
to its CPU/tuple `encode()` API. FP16/BF16 must be validated separately against
FP32 on real observations. Dynamic CPU INT8 is not a CUDA backend.

## Latency contract and limits

The strict default deadline is **less than 33 ms** from submission (or supplied
capture timestamp) through CPU observation of the completed result. This includes
CPU staging, GPU queueing, H2D, encoder/retrieval, D2H and polling delay. Late
results carry `deadline_missed=True` and `action_id=None`. GPU work already
submitted cannot be cancelled; a full ring immediately returns `None` from
`try_submit`. The CPU stages a frame before enqueueing H2D, so this API avoids
GPU waits, but does not claim zero CPU copy cost or zero driver overhead.

CUDA graphs do not guarantee a hard real-time deadline, fully hide a PCIe copy,
or speed up an encoder beyond the GPU's compute capacity. In particular, two
queued requests can each have much higher latency than the completion interval.
Use measured tail latency, not throughput alone. Hardware copy-engine support,
memory bandwidth, GPU contention, OS scheduling and model/bank sizes determine
actual overlap and runtime. Per-slot graph pools also increase VRAM use. The
full exact search costs O(N*D) work and O(N) similarity scratch per graph.

The benchmark reports median/p99/max end-to-end latency, graph time, throughput,
invalid embeddings and deadline misses. It exits nonzero on any deadline miss
or invalid embedding. It uses pre-generated synthetic clips and a random bank:
even with real V-JEPA weights this is an inference benchmark, not a game-policy
or capture-to-actuation validation. The first graph replay and transfer paths
are warmed during construction; setup is excluded from measured runtime.

## Run and validate

Install an appropriate CUDA-enabled PyTorch build and `requirements-memory.txt`
in your GPU environment. Inductor additionally needs a supported compiler and
Triton setup; use `--no-compile` to test manual graphs without Inductor.

```powershell
# Tiny synthetic encoder: exercise the pipeline before loading large weights.
.venv\Scripts\python.exe scripts/run_gpu_recorded_agent.py --no-compile

# Actual V-JEPA encoder, synthetic clip/bank benchmark, manual graphs + Inductor.
.venv\Scripts\python.exe scripts/run_gpu_recorded_agent.py --repository checkpoints/vjepa2 --checkpoint checkpoints/vjepa2_1_vitb_dist_vitG_384.pt --frames 1000

# CPU ownership tests (no torch required).
.venv\Scripts\python.exe -m unittest discover -s tests -p test_gpu_recorded_agent.py -v

# Hardware-backed replay/slot-isolation tests.
$env:CUPHEAD_TEST_CUDA = "1"
.venv\Scripts\python.exe -m unittest discover -s tests -p test_gpu_recorded_agent.py -v
$env:CUPHEAD_TEST_CUDA_COMPILE = "1"
.venv\Scripts\python.exe -m unittest discover -s tests -p test_gpu_recorded_agent.py -v
```

Use NVIDIA Nsight Systems to confirm H2D overlaps compute and inspect CUDA API
waits; event timings alone do not prove physical overlap. Test under the actual
game workload and GPU contention before accepting the 33 ms budget. Real CUDA
replay, V-JEPA capture compatibility and performance remain unvalidated in the
development environment where PyTorch is not installed.

Implementation references: [PyTorch CUDA semantics and graph constraints](https://docs.pytorch.org/docs/stable/notes/cuda.html),
[torch.compile options](https://docs.pytorch.org/docs/stable/generated/torch.compile.html),
and [NVIDIA graph capture checklist](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/quick-checklist.html).
