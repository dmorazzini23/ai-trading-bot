"""Logging helpers for the AI trading bot."""

from __future__ import annotations

import logging
import os
from typing import Dict

from logger_rotator import get_rotating_handler

LOG_PATH = os.getenv("LOG_PATH", "logs/ai_trading_bot.log")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 10_000_000))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))

_configured = False
_loggers: Dict[str, logging.Logger] = {}


def setup_logging() -> None:
    """Configure the root logger once."""
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = get_rotating_handler(
        LOG_PATH, max_bytes=LOG_MAX_BYTES, backup_count=LOG_BACKUP_COUNT
    )
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    root_logger.info("Logging initialized. Writing logs to %s", LOG_PATH)
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
