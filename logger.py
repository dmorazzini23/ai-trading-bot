"""Logging helpers for the AI trading bot."""

from __future__ import annotations

import logging
import sys
from typing import Dict

_configured = False
_loggers: Dict[str, logging.Logger] = {}


def setup_logging() -> None:
    """Configure the root logger once."""
    global _configured
    if _configured:
        return

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s - %(message)s'
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # Clear existing handlers and set only stream handler
    logger.handlers.clear()
    logger.addHandler(stream_handler)

    logger.info(
        "Logging initialized: outputting only to stdout (systemd journal)"
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
        lg.setLevel(logging.INFO)
        _loggers[name] = lg
    return _loggers[name]


logger = logging.getLogger(__name__)

__all__ = ["setup_logging", "get_logger", "logger"]
