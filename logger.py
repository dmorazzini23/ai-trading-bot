"""Utility helpers for consistent logging setup across the project."""

from __future__ import annotations

import logging
import os
import sys

from logger_rotator import get_rotating_handler


LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "bot.log")

logger = logging.getLogger("ai_trading_bot")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    _stream = logging.StreamHandler(sys.stdout)
    _stream.setFormatter(_formatter)
    _file_handler = get_rotating_handler(LOG_FILE)
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_stream)
    logger.addHandler(_file_handler)

logger.propagate = False


def log_uncaught_exceptions(ex_cls, ex, tb):
    """Log unhandled exceptions via the global logger."""

    logger.critical("Uncaught exception", exc_info=(ex_cls, ex, tb))


sys.excepthook = log_uncaught_exceptions


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the global logger (backwards compatibility)."""

    return logger
