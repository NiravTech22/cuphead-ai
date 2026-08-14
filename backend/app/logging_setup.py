"""Central logging configuration.

Every module gets a logger via `get_logger(__name__)`. We log the device used,
each pipeline step, and graceful-failure paths so the (sometimes long) waits are
legible and debuggable.
"""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def setup_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.getenv("VERBATIM_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(name)-18s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers[:] = [handler]
    # yt-dlp and httpx are noisy at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name.split(".")[-1])
