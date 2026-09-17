"""Desktop Duplication capture, with owned RGB bytes and bounded fresh-frame waits."""
from __future__ import annotations

import math
import time

from .capture import Frame, frame_checksum


class DXCamFrameSource:
    """Monitor is one-based; region coordinates are relative to that output.

    DXcam may return None when no new desktop frame exists. Do not assign an
    index to that result or silently return an old buffer as a fresh frame.
    Indices count delivered samples, not the game's presentation counter.
    """
    def __init__(self, *, monitor=1, region=None, device_idx=0, timeout=2.0):
        if monitor < 1 or device_idx < 0:
            raise ValueError("DXcam requires monitor >= 1 and device_idx >= 0")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        rect = None
        if region is not None:
            left, top = region["left"], region["top"]
            width, height = region["width"], region["height"]
            if min(left, top) < 0 or min(width, height) <= 0:
                raise ValueError("DXcam region must be positive-sized and output-relative")
            rect = (left, top, left + width, top + height)
        try:
            import dxcam
        except ImportError as exc:
            raise RuntimeError("Install DXcam with `pip install -r requirements.txt`, or select backend='mss'.") from exc
        self._camera = dxcam.create(device_idx=device_idx, output_idx=monitor - 1,
                                    output_color="RGB")
        self._region = rect
        self._timeout = timeout
        self._index = 0

    def read(self):
        if self._camera is None:
            raise RuntimeError("read() on a closed FrameSource")
        deadline = time.perf_counter() + self._timeout
        while True:
            pixels = self._camera.grab(region=self._region)
            if pixels is not None:
                captured = time.perf_counter()
                height, width = pixels.shape[:2]
                raw = pixels.tobytes()
                frame = Frame(self._index, captured, raw, frame_checksum(raw), width, height)
                self._index += 1
                return frame
            if time.perf_counter() >= deadline:
                raise TimeoutError("DXcam did not deliver a new desktop frame before timeout")
            time.sleep(0.001)

    def close(self):
        if self._camera is not None:
            try:
                self._camera.release()
            finally:
                self._camera = None
