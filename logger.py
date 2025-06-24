"""Logging helpers for the AI trading bot."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Dict

_configured = False
_loggers: Dict[str, logging.Logger] = {}


def get_rotating_handler(
    path: str,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> logging.Handler:
    """Return a size-rotating file handler. Falls back to stderr on failure."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
    except OSError as exc:
        logging.getLogger(__name__).error("Cannot open log file %s: %s", path, exc)
        handler = logging.StreamHandler(sys.stderr)
    return handler


def setup_logging(debug: bool = False, log_file: str | None = None) -> None:
    """Configure the root logger once."""
    global _configured
    if _configured:
        return

    level_name = os.getenv("LOG_LEVEL", "DEBUG" if debug else "INFO").upper()
    level = logging.DEBUG if level_name == "DEBUG" else logging.INFO

    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s - %(message)s'
    )

    handlers = []
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    if log_file:
        try:
            file_handler = get_rotating_handler(log_file, max_bytes=10_485_760)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except OSError as exc:
            logging.getLogger(__name__).error(
                "Failed to configure log file %s: %s", log_file, exc
            )

    logger.handlers.clear()
    for h in handlers:
        logger.addHandler(h)

    file_part = f" with file {log_file}" if log_file else ""
    logger.info(
        "Logging initialized%s (level %s)",
        file_part,
        logging.getLevelName(level),
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring logging on first use."""
    if name not in _loggers:
        setup_logging()
        lg = logging.getLogger(name)
        if not lg.handlers:
            for h in logging.getLogger().handlers:
                lg.addHandler(h)
        lg.setLevel(logging.NOTSET)
        _loggers[name] = lg
    return _loggers[name]


logger = logging.getLogger(__name__)

__all__ = ["setup_logging", "get_logger", "logger"]

