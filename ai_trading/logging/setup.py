"""Helpers for configuring logging and tracking log file paths."""

from __future__ import annotations

import logging
from typing import List

# Internal list of log file paths used when configuring logging.
_logger_paths: list[str] | None = None


def get_logger_paths() -> list[str]:
    """Return a copy of the log file paths registered so far."""
    return list(_logger_paths or [])


def setup_logging(debug: bool = False, log_file: str | None = None) -> logging.Logger:
    """Configure logging and track any file handlers created.

    The ``debug`` flag is kept for compatibility but the effective log level is
    now driven by configuration or the ``LOG_LEVEL`` environment variable.
    Paths are tracked in ``_logger_paths`` which can be retrieved via
    :func:`get_logger_paths`.
    """
    from . import setup_logging as _setup_logging

    global _logger_paths
    if _logger_paths is None:
        _logger_paths = []

    if log_file and log_file not in _logger_paths:
        _logger_paths.append(log_file)

    # ``debug`` is intentionally ignored; callers should set ``LOG_LEVEL``.
    return _setup_logging(log_file=log_file)
