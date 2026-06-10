"""Structured logging for btc_autoresearch."""
from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "btc_autoresearch", level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide logger that writes to stdout exactly once.

    Idempotent: repeated calls with the same name reuse the same handler.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
