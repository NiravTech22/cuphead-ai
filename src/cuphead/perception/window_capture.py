"""Capture the X11 game drawable itself; XWayland's root can be entirely black."""

from __future__ import annotations

import time

from .capture import Frame, frame_checksum


class X11WindowSource:
    def __init__(self, title: str = "Cuphead", *, window_id: int | None = None):
        from Xlib import X, display

        self._X = X
        self._display = display.Display()
        self._index = 0
        try:
            if window_id is not None:
                self._window = self._display.create_resource_object("window", window_id)
            else:
                self._window = self._find(self._display.screen().root, title)
            if self._window is None:
                raise RuntimeError(f"no visible X11 game window named {title!r}")
            self._window.get_geometry()
        except BaseException:
            self._display.close()
            raise

    def _find(self, window, title):
        from Xlib.error import BadWindow

        try:
            if (
                window.get_wm_name() == title
                and window.get_attributes().map_state == self._X.IsViewable
            ):
                return window
            for child in window.query_tree().children:
                found = self._find(child, title)
                if found is not None:
                    return found
        except BadWindow:
            return None
        return None

    @property
    def window_id(self) -> int:
        return self._window.id

    def read(self) -> Frame:
        from PIL import Image

        geometry = self._window.get_geometry()
        shot = self._window.get_image(
            0, 0, geometry.width, geometry.height, self._X.ZPixmap, 0xFFFFFFFF
        )
        if shot is None or shot.depth not in (24, 32):
            raise RuntimeError("X11 capture requires a 24/32-bit drawable")
        order = (
            "BGRX"
            if self._display.display.info.image_byte_order == self._X.LSBFirst
            else "XRGB"
        )
        image = Image.frombytes(
            "RGB", (geometry.width, geometry.height), shot.data, "raw", order
        )
        raw = image.tobytes()
        frame = Frame(
            self._index,
            time.perf_counter(),
            raw,
            frame_checksum(raw),
            geometry.width,
            geometry.height,
        )
        self._index += 1
        return frame

    def close(self):
        self._display.close()
