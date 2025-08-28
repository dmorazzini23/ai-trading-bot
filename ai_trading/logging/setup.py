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

    This is a thin wrapper around :func:`ai_trading.logging.setup_logging` that
    records the ``log_file`` argument whenever a ``RotatingFileHandler`` is
    requested.  Paths are tracked in ``_logger_paths`` which can be retrieved via
    :func:`get_logger_paths`.
    """
    from . import setup_logging as _setup_logging

    global _logger_paths
    if _logger_paths is None:
        _logger_paths = []

    if log_file and log_file not in _logger_paths:
        _logger_paths.append(log_file)

    return _setup_logging(debug=debug, log_file=log_file)
