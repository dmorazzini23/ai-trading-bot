"""Utility helpers for consistent logging setup across the project."""

from __future__ import annotations

import logging
import os
import sys

from logger_rotator import get_rotating_handler


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to ``log_file`` and stdout."""

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    handler = get_rotating_handler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a logger configured for the standard bot log file."""

    log_file = os.path.join(os.path.dirname(__file__), "logs", "bot.log")
    return setup_logger(name, log_file)


def configure_root_logger(log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger with a rotating file and stdout handler."""
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    handler = get_rotating_handler(log_file)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[handler, logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return logging.getLogger()


def log_uncaught_exceptions(ex_cls, ex, tb):
    """Log unhandled exceptions via the root logger."""

    logger = logging.getLogger()
    logger.critical("Uncaught exception", exc_info=(ex_cls, ex, tb))


sys.excepthook = log_uncaught_exceptions
