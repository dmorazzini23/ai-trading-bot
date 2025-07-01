"""Logging helpers for the AI trading bot."""

import logging
import os
import queue
import sys
from logging.handlers import (
    QueueHandler,
    QueueListener,
    RotatingFileHandler,
    TimedRotatingFileHandler,
)
from typing import Dict

_configured = False
_loggers: Dict[str, logging.Logger] = {}
_log_queue: queue.Queue | None = None
_listener: QueueListener | None = None


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


def setup_logging(debug: bool = False, log_file: str | None = None) -> logging.Logger:
    """Configure the root logger in an idempotent way."""
    global _configured
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    # Attach console handler once
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        logger.addHandler(stream_handler)

    if log_file:
        # Always add rotating handler when log_file is provided
        rotating_handler = get_rotating_handler(log_file)
        rotating_handler.setFormatter(formatter)
        rotating_handler.setLevel(logging.INFO)
        logger.addHandler(rotating_handler)

    _configured = True
    return logger


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

def init_logger(log_file: str) -> logging.Logger:
    """Wrapper used by utilities to initialize logging."""
    # AI-AGENT-REF: provide simple alias for setup_logging
    return setup_logging(log_file=log_file)


__all__ = ["setup_logging", "get_logger", "init_logger", "logger"]


