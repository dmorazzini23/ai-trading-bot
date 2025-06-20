"""Deprecated helper for creating rotating log handlers."""

import os
from logging.handlers import RotatingFileHandler


def get_rotating_handler(
    path: str,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> RotatingFileHandler:
    """Return a configured :class:`RotatingFileHandler`.

    .. deprecated:: 1.0
       File based logging is disabled; logs should be captured via ``stdout``.
    """

    raise NotImplementedError(
        "File logging is disabled; use centralized stdout logging instead"
    )
