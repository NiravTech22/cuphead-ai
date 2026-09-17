"""Independent game-window video capture with a monotonic timestamp sidecar."""

from __future__ import annotations

import json
import platform
import threading
import time
from pathlib import Path


class WindowVideoRecorder:
    """Record X11 on Linux or the primary Windows desktop while planning blocks.

    Each thread owns its capture source. AVI uses a fixed playback rate;
    the JSONL sidecar preserves actual capture times and missed deadlines.
    """

    def __init__(self, path: str | Path, *, fps: int = 20):
        self.path = Path(path)
        if self.path.suffix.lower() != ".avi" or fps < 1:
            raise ValueError("video recording requires an .avi path and positive fps")
        self.fps = fps
        self.frames = 0
        self.missed_deadlines = 0
        self.error = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            self.close()
            raise RuntimeError("video recorder did not initialize within 10 seconds")
        self.check()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"video recorder failed: {self.error}")

    def _run(self):
        source = writer = None
        try:
            import cv2
            import numpy as np
            from .window_capture import open_game_source

            # DXcam caches cameras by output: do not share/release the agent's
            # camera from this independent recorder thread.
            if platform.system() == "Windows":
                from .capture import open_screen_source
                source = open_screen_source(backend="mss")
            else:
                source = open_game_source()
            first = source.read()
            writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), self.fps,
                                     (first.width, first.height))
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not open MJPG AVI writer")
            self._ready.set()
            deadline = time.perf_counter()
            with self.path.with_suffix(".frames.jsonl").open("w") as log:
                while not self._stop.is_set():
                    frame = first if self.frames == 0 else source.read()
                    if (frame.width, frame.height) != (first.width, first.height):
                        raise RuntimeError("game window resized during recording")
                    pixels = np.frombuffer(frame.payload, dtype=np.uint8).reshape(frame.height, frame.width, 3)
                    writer.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                    log.write(json.dumps({"video_frame": self.frames, "capture_t": frame.t_capture,
                                          "checksum": frame.checksum}) + "\n")
                    self.frames += 1
                    deadline += 1 / self.fps
                    now = time.perf_counter()
                    if now > deadline:
                        self.missed_deadlines += 1
                        deadline = now
                    self._stop.wait(max(0, deadline - now))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._ready.set()
            if writer is not None:
                writer.release()
            if source is not None:
                source.close()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("video recording thread failed to stop")
        self.check()

    def summary(self):
        return {"path": str(self.path), "fps": self.fps, "frames": self.frames,
                "missed_deadlines": self.missed_deadlines, "error": self.error,
                "timestamps": str(self.path.with_suffix(".frames.jsonl"))}
