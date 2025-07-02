"""Logging helpers for the AI trading bot."""

import logging
import time
import os
import queue
import sys
import csv
from datetime import date

import pandas as pd
import numpy as np
import metrics_logger
from logging.handlers import (
    QueueHandler,
    QueueListener,
    RotatingFileHandler,
    TimedRotatingFileHandler,
)
from typing import Dict


class UTCFormatter(logging.Formatter):
    """Formatter with UTC timestamps and structured phase tags."""

    converter = time.gmtime

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

    formatter = UTCFormatter(
        "%(asctime)sZ %(levelname)s [%(bot_phase)s] %(name)s - %(message)s"
    )

    class _PhaseFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if not hasattr(record, "bot_phase"):
                record.bot_phase = "GENERAL"
            return True

    # Attach console handler once
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        stream_handler.addFilter(_PhaseFilter())
        logger.addHandler(stream_handler)

    if log_file:
        # Always add rotating handler when log_file is provided
        rotating_handler = get_rotating_handler(log_file)
        rotating_handler.setFormatter(formatter)
        rotating_handler.setLevel(logging.INFO)
        rotating_handler.addFilter(_PhaseFilter())
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


def log_performance_metrics(
    exposure_pct: float,
    equity_curve: list[float],
    regime: str,
    filename: str = "logs/performance.csv",
    *,
    as_of: date | None = None,
) -> None:
    """Log daily performance metrics to ``filename``."""

    if not equity_curve:
        return
    as_of = as_of or date.today()
    returns = pd.Series(equity_curve).pct_change().dropna()
    roll = returns.tail(20)
    if roll.empty:
        sharpe = sortino = realized_vol = 0.0
    else:
        sharpe = roll.mean() / (roll.std(ddof=0) or 1e-9) * np.sqrt(252 / 20)
        downside = roll[roll < 0]
        sortino = roll.mean() / (downside.std(ddof=0) or 1e-9) * np.sqrt(252 / 20)
        realized_vol = roll.std(ddof=0) * np.sqrt(252 / 20)
    max_dd = metrics_logger.compute_max_drawdown(equity_curve)
    rec = {
        "date": str(as_of),
        "exposure_pct": exposure_pct,
        "sharpe20": sharpe,
        "sortino20": sortino,
        "realized_vol": realized_vol,
        "max_drawdown": max_dd,
        "regime": regime,
    }
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        new = not os.path.exists(filename)
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rec.keys())
            if new:
                writer.writeheader()
            writer.writerow(rec)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to log performance metrics: %s", exc)


__all__ = [
    "setup_logging",
    "get_logger",
    "init_logger",
    "logger",
    "log_performance_metrics",
]


