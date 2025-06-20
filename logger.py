"""Logging helpers for the AI trading bot."""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict

_configured = False
_loggers: Dict[str, logging.Logger] = {}


def setup_logging(debug: bool = False) -> None:
    """Configure the root logger once.

    Parameters
    ----------
    debug : bool, optional
        If ``True``, set log level to ``DEBUG`` regardless of the ``LOG_LEVEL``
        environment variable. Defaults to ``False``.
    """
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

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # Clear existing handlers and set only stream handler
    logger.handlers.clear()
    logger.addHandler(stream_handler)

    logger.info(
        "Logging initialized: outputting only to stdout (level %s)",
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
        # Propagate level from root logger
        lg.setLevel(logging.NOTSET)
        _loggers[name] = lg
    return _loggers[name]


logger = logging.getLogger(__name__)

__all__ = ["setup_logging", "get_logger", "logger"]
