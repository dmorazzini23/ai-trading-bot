"""Helpers for configuring logging and tracking log file paths."""

from __future__ import annotations

import logging


# Internal list of log file paths used when configuring logging.
_logger_paths: list[str] | None = None


def get_logger_paths() -> list[str]:
    """Return a copy of the log file paths registered so far."""
    return list(_logger_paths or [])


def _apply_library_filters() -> None:
    """Set log levels for noisy third-party libraries.

    Default filters target common verbose dependencies but can be adjusted via
    the ``LOG_QUIET_LIBRARIES`` environment variable which accepts comma
    separated ``logger=LEVEL`` pairs. Example: ``urllib3=WARNING,foo=ERROR``.
    """
    from ai_trading.config import management as config

    # ``charset_normalizer`` is particularly noisy at DEBUG level when used by
    # ``requests``. Elevate it to WARNING by default so debug logs from that
    # dependency do not clutter our output. Additional filters may be provided
    # via ``LOG_QUIET_LIBRARIES``.
    filters: dict[str, int] = {
        "charset_normalizer": logging.WARNING,
        # Peewee SQL debug statements are too verbose for production; keep at WARNING.
        "peewee": logging.WARNING,
    }
    raw = config.get_env("LOG_QUIET_LIBRARIES", "")
    for item in raw.split(","):
        name, _, level = item.partition("=")
        if name.strip() and level.strip():
            filters[name.strip()] = getattr(logging, level.strip().upper(), logging.INFO)
    from ai_trading.utils.logging import SuppressBelowLevelFilter

    for name, level in filters.items():
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addFilter(SuppressBelowLevelFilter(level))


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
    logger = _setup_logging(log_file=log_file)
    _apply_library_filters()
    return logger
