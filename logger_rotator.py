"""Expose a helper to construct rotating log handlers for smoke tests."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler


def get_rotating_handler(
    filename: str,
    *,
    max_bytes: int = 10_000_000,
    backup_count: int = 3,
) -> logging.Handler:
    """Return a configured rotating file handler."""

    handler = RotatingFileHandler(
        filename,
        maxBytes=int(max_bytes),
        backupCount=int(backup_count),
    )
    handler.setLevel(logging.INFO)
    return handler
