"""Helper for creating rotating log handlers."""

from logging.handlers import RotatingFileHandler
import os


def get_rotating_handler(
    path: str,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> RotatingFileHandler:
    """Return a configured :class:`RotatingFileHandler`.

    Parameters
    ----------
    path : str
        File to write logs to.
    max_bytes : int, optional
        Maximum size of each log file before rotation. Defaults to ``10_000_000``.
    backup_count : int, optional
        Number of rotated log files to keep. Defaults to ``5``.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
