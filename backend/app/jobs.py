"""jobs.py — job registry + TRUE cancellation for main pipeline queries.

Every main query is a Job (job_id == the query's qid). The registry tracks,
per job: status, spawned subprocesses, temp-file patterns, and abort
callbacks (e.g. closing the in-flight Ollama stream). `cancel()`:

  1. marks the job cancelled (checkpoints in the tool loop / whisper segment
     loop / yt-dlp progress hook raise JobCancelled at the next check),
  2. fires abort callbacks (LLM stream close ends Ollama generation),
  3. SIGTERMs registered subprocesses, SIGKILL after a 2s grace,
  4. deletes registered partial temp files (completed artifacts are
     UNregistered by their creators on success, so they survive).

Cancellation boundary (ADR-014): a single in-flight Whisper forward pass or
one Ollama stream chunk is not interruptible mid-call; cancellation takes
effect at the next checkpoint (≤ a segment / a token) — subprocesses die
immediately regardless.

Only ONE main pipeline runs at a time (4GB GPU): `pipeline_gate`.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .logging_setup import get_logger

log = get_logger(__name__)

# One video pipeline at a time, ever. Semaphore = thread-agnostic (the SSE
# generator's turns may run on different threadpool threads).
pipeline_gate = threading.BoundedSemaphore(1)


class JobCancelled(Exception):
    """Raised at cancellation checkpoints inside the pipeline."""


class Job:
    def __init__(self, job_id: str, query: str = ""):
        self.id = job_id
        self.query = query
        self.status = "running"            # running/cancelled/done/failed
        self.gate_held = False             # holds pipeline_gate (released once)
        self.resolved_title: Optional[str] = None
        self.procs: set[subprocess.Popen] = set()
        self.temps: set[str] = set()       # glob patterns of partial artifacts
        self.aborts: list[Callable] = []
        self.lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self.status == "cancelled"


_jobs: dict[str, Job] = {}
_current: Optional[Job] = None
_reg_lock = threading.Lock()


def start(job_id: str, query: str) -> Job:
    """Register a job. It does NOT become `current` until activate() — a job
    waiting on the pipeline gate must not hijack the running job's registry."""
    with _reg_lock:
        job = Job(job_id, query)
        _jobs[job_id] = job
        for k in list(_jobs)[:-24]:        # keep the registry bounded
            _jobs.pop(k, None)
    return job


def activate(job: Job) -> None:
    """Make this job `current` — call only after acquiring the pipeline gate."""
    global _current
    with _reg_lock:
        _current = job


def current() -> Optional[Job]:
    return _current


def get(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def finish(job: Job, status: str = "done") -> None:
    global _current
    if job.status == "running":
        job.status = status
    with _reg_lock:
        if _current is job:
            _current = None


def check(job: Optional[Job] = None) -> None:
    """Checkpoint: raise JobCancelled if the (given or current) job was cancelled."""
    j = job or _current
    if j is not None and j.cancelled:
        raise JobCancelled(j.id)


def register_temp(pattern: str | Path, job: Optional[Job] = None) -> None:
    j = job or _current
    if j:
        with j.lock:
            j.temps.add(str(pattern))


def unregister_temp(pattern: str | Path, job: Optional[Job] = None) -> None:
    j = job or _current
    if j:
        with j.lock:
            j.temps.discard(str(pattern))


def register_abort(cb: Callable, job: Optional[Job] = None) -> None:
    j = job or _current
    if j:
        with j.lock:
            j.aborts.append(cb)


def unregister_abort(cb: Callable, job: Optional[Job] = None) -> None:
    j = job or _current
    if j:
        with j.lock:
            if cb in j.aborts:
                j.aborts.remove(cb)


def release_gate(job: Job) -> None:
    """Release the pipeline gate exactly once per acquisition. Called from the
    pipeline's finally AND from cancel(): a client that aborts its SSE stream
    leaves the sync generator suspended (Starlette won't finalize it), so the
    gate must not depend on generator finalization."""
    with job.lock:
        if not job.gate_held:
            return
        job.gate_held = False
    pipeline_gate.release()


def run_proc(cmd: list[str]) -> subprocess.CompletedProcess:
    """subprocess.run replacement: the child is registered with the current
    job so cancel() can SIGTERM/SIGKILL it immediately."""
    job = _current
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    if job:
        with job.lock:
            job.procs.add(p)
    try:
        out, err = p.communicate()
        if job and job.cancelled:
            raise JobCancelled(job.id)
        return subprocess.CompletedProcess(cmd, p.returncode, out, err)
    finally:
        if job:
            with job.lock:
                job.procs.discard(p)


def _reap(job: Job, procs: list[subprocess.Popen]) -> None:
    deadline = time.time() + 2.0                   # 2s grace, then SIGKILL
    for p in procs:
        try:
            p.wait(timeout=max(0.05, deadline - time.time()))
        except Exception:
            try:
                p.kill()
                log.info("[%s] SIGKILL pid %s after grace", job.id, p.pid)
            except Exception:
                pass
    time.sleep(0.3)                                # let fds settle
    removed = 0
    with job.lock:
        patterns = list(job.temps)
    for pat in patterns:
        pp = Path(pat)
        for f in pp.parent.glob(pp.name):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    log.info("[%s] cancelled: %d proc(s) terminated, %d partial file(s) removed",
             job.id, len(procs), removed)
    # free the pipeline for the next job (the cancelled thread aborts at its
    # next checkpoint — overlap is bounded to one whisper segment / LLM chunk)
    release_gate(job)


def cancel(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        return {"ok": False, "status": "unknown"}
    if job.status != "running":
        return {"ok": True, "status": job.status}
    job.status = "cancelled"
    log.info("[%s] cancel requested by user", job.id)
    with job.lock:
        aborts, procs = list(job.aborts), list(job.procs)
    for cb in aborts:                              # close LLM stream etc.
        try:
            cb()
        except Exception:
            pass
    for p in procs:                                # SIGTERM now
        try:
            p.terminate()
        except Exception:
            pass
    threading.Thread(target=_reap, args=(job, procs), daemon=True).start()
    return {"ok": True, "status": "cancelled"}
