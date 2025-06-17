"""Helper for creating rotating log handlers."""

from __future__ import annotations

from logging.handlers import RotatingFileHandler


def get_rotating_handler(path: str, max_bytes: int = 5_000_000, backup_count: int = 3) -> RotatingFileHandler:
    """Return a configured :class:`RotatingFileHandler`."""

    return RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
