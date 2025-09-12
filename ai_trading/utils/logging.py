"""Utility filters for logging configuration."""
from __future__ import annotations

import logging


class SuppressBelowLevelFilter(logging.Filter):
    """Filter that drops records below ``min_level``.

    The logger's own level can remain ``DEBUG`` while this filter suppresses
    lower level messages, allowing callers to keep a debug-effective level but
    still quiet noisy third-party libraries.
    """

    def __init__(self, min_level: int) -> None:
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        return record.levelno >= self.min_level
