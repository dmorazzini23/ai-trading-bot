"""Helper for creating rotating log handlers."""

from logging.handlers import RotatingFileHandler
import logging
import os
import sys


def get_rotating_handler(
    path: str,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> logging.Handler:
    """Return a size rotating file handler.

    Falls back to ``stderr`` if the file cannot be created.
    """

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
    except OSError as exc:  # pragma: no cover - filesystem errors
        logging.getLogger(__name__).error("Cannot open log file %s: %s", path, exc)
        handler = logging.StreamHandler(sys.stderr)
    return handler
