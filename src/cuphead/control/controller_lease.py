"""One virtual controller owner per user, including across repository copies."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


class ControllerLease:
    def __init__(self, path=None):
        # Do not delete the file: replacing it would allow two distinct locks.
        self._file = open(path or Path(tempfile.gettempdir()) / "cuphead-ai-controller.lock", "a+b")
        try:
            self._file.seek(0, 2)
            if self._file.tell() == 0:
                self._file.write(b"\0")
                self._file.flush()
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._file.close()
            raise RuntimeError(
                "Another Cuphead AI controller session is active, or its lock is "
                "unavailable. Close the game and that session first. Run the agent "
                "with --launch; do not run launch_with_vgamepad alongside it."
            ) from exc

    def close(self):
        self._file.close()  # The OS releases the lock, including on process exit.
