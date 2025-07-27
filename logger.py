"""Logging helpers for the AI trading bot."""

import logging
import time
import os
import queue
import sys
import csv
import json
import traceback
from datetime import date, datetime
import atexit
import config

import metrics_logger

# Configure root formatting once in UTC
logging.Formatter.converter = time.gmtime
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
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


class JSONFormatter(logging.Formatter):
    """JSON log formatter with secret masking."""

    converter = time.gmtime

    def _json_default(self, obj):
        """Fallback serialization for unsupported types."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        omit = {
            "msg",
            "message",
            "args",
            "levelname",
            "levelno",
            "name",
            "created",
            "msecs",
            "relativeCreated",
            "asctime",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
        }
        for k, v in record.__dict__.items():
            if k in omit:
                continue
            if "key" in k.lower() or "secret" in k.lower():
                v = config.mask_secret(str(v))
            payload[k] = v
        if record.exc_info:
            exc_type, exc_value, _exc_tb = record.exc_info
            payload["exc"] = "".join(
                traceback.format_exception_only(exc_type, exc_value)
            ).strip()
        return json.dumps(payload, default=self._json_default)

_configured = False
_loggers: Dict[str, logging.Logger] = {}
_log_queue: queue.Queue | None = None
_listener: QueueListener | None = None


def get_rotating_handler(
    path: str,
    max_bytes: int = 5_000_000,
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
    global _configured, _log_queue, _listener
    logger = logging.getLogger()
    if _configured:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = JSONFormatter(
        "%(asctime)sZ"
    )

    class _PhaseFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if not hasattr(record, "bot_phase"):
                record.bot_phase = "GENERAL"
            return True

    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    stream_handler.addFilter(_PhaseFilter())
    handlers.append(stream_handler)

    if log_file:
        rotating_handler = get_rotating_handler(log_file)
        rotating_handler.setFormatter(formatter)
        rotating_handler.setLevel(logging.INFO)
        rotating_handler.addFilter(_PhaseFilter())
        handlers.append(rotating_handler)

    _log_queue = queue.Queue(-1)
    queue_handler = QueueHandler(_log_queue)
    queue_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    queue_handler.addFilter(_PhaseFilter())
    # AI-AGENT-REF: QueueHandler should enqueue raw records without formatting
    logger.handlers = [queue_handler]
    # AI-AGENT-REF: use background queue listener to reduce I/O blocking
    _listener = QueueListener(_log_queue, *handlers, respect_handler_level=True)
    _listener.start()

    def _stop_listener() -> None:
        if _listener:
            try:
                _listener.stop()
            except Exception:
                pass

    atexit.register(_stop_listener)

    _configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring logging on first use."""
    if name not in _loggers:
        root = setup_logging()
        logger = logging.getLogger(name)
        logger.handlers = root.handlers.copy()
        logger.setLevel(root.level)
        _loggers[name] = logger
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
    import pandas as pd
    import numpy as np

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


