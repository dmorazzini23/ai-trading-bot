"""Logging helpers for the AI trading bot.

These helpers centralize logger configuration and now include a sanitizer that
prevents collisions with reserved :class:`logging.LogRecord` attributes. Any
key in ``extra`` matching a reserved field is automatically prefixed with
``x_`` to keep structured logging safe. Use :func:`get_logger` to obtain a
sanitizing adapter for all modules.
"""  # AI-AGENT-REF: document extra key sanitization

import atexit
import csv
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from datetime import UTC, date, datetime
from typing import Any, Dict

# Reserved LogRecord attribute names that cannot be overridden by `extra`.
_RESERVED_LOGRECORD_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "asctime",
}  # AI-AGENT-REF: prevent reserved key collisions


def _sanitize_extra(extra: Dict[str, Any] | None) -> Dict[str, Any]:
    """Rename reserved LogRecord keys with ``x_`` prefix."""  # AI-AGENT-REF: sanitizer
    if not extra:
        return {}
    return {
        (k if k not in _RESERVED_LOGRECORD_KEYS else f"x_{k}"): v
        for k, v in extra.items()
    }


class SanitizingLoggerAdapter(logging.LoggerAdapter):
    """Adapter that sanitizes ``extra`` keys to avoid LogRecord collisions."""

    # AI-AGENT-REF: intercept and sanitize extra dicts
    def process(self, msg, kwargs):
        extra = kwargs.get("extra")
        if extra is not None:
            kwargs["extra"] = _sanitize_extra(extra)
        return msg, kwargs


_ROOT_LOGGER_NAME = "ai_trading"  # AI-AGENT-REF: default root logger name

# AI-AGENT-REF: structured logging helper to avoid stray kwargs
def with_extra(logger, level, msg: str, *, extra: dict | None = None, **_ignored):
    """Log `msg` with structured fields via `extra` only (no arbitrary kwargs)."""
    payload = {} if extra is None else dict(extra)
    logger.log(level, msg, extra=payload)


def info_kv(logger, msg: str, *, extra: dict | None = None):
    """Log at INFO level with structured key-value fields."""  # AI-AGENT-REF: wrapper
    with_extra(logger, logging.INFO, msg, extra=extra)


def warning_kv(logger, msg: str, *, extra: dict | None = None):
    """Log at WARNING level with structured key-value fields."""  # AI-AGENT-REF: wrapper
    with_extra(logger, logging.WARNING, msg, extra=extra)


def error_kv(logger, msg: str, *, extra: dict | None = None):
    """Log at ERROR level with structured key-value fields."""  # AI-AGENT-REF: wrapper
    with_extra(logger, logging.ERROR, msg, extra=extra)


# AI-AGENT-REF: Handle missing dependencies gracefully for testing
# AI-AGENT-REF: Lazy import config to avoid import-time dependencies
def _get_config():
    """Lazy import config management to avoid import-time dependencies."""
    from ai_trading.config import management as config
# AI-AGENT-REF: Import monitoring metrics (lazy load in functions if needed)
def _get_metrics_logger():
    """Lazy import metrics_logger to avoid import-time dependencies."""
    from ai_trading.telemetry import metrics_logger
# AI-AGENT-REF: Configure UTC formatting only, remove import-time basicConfig to prevent duplicates
logging.Formatter.converter = time.gmtime
from logging.handlers import (
    QueueHandler,
    QueueListener,
    RotatingFileHandler,
)


class UTCFormatter(logging.Formatter):
    """Formatter with UTC timestamps and structured phase tags."""

    converter = time.gmtime


class JSONFormatter(logging.Formatter):
    """JSON log formatter with secret masking."""

    converter = time.gmtime

    def _json_default(self, obj):
        """Fallback serialization for unsupported types."""
        if isinstance(obj, datetime | date):
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
                v = _get_config().mask_secret(str(v))
            payload[k] = v
        if record.exc_info:
            exc_type, exc_value, _exc_tb = record.exc_info
            payload["exc"] = "".join(
                traceback.format_exception_only(exc_type, exc_value)
            ).strip()
        return json.dumps(payload, default=self._json_default, ensure_ascii=False)


class CompactJsonFormatter(JSONFormatter):
    """Compact JSON log formatter that drops large extra payloads."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        
        # In compact mode, only include essential extra fields
        essential_fields = {
            "bot_phase",
            "present",  # For config verification logs
            "timestamp",  # For trading events
        }
        
        for k, v in record.__dict__.items():
            if k in essential_fields and k not in {
                "msg", "message", "args", "levelname", "levelno", "name",
                "created", "msecs", "relativeCreated", "asctime", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "thread", "threadName", "processName",
                "process", "taskName"
            }:
                if "key" in k.lower() or "secret" in k.lower():
                    v = _get_config().mask_secret(str(v))
                payload[k] = v
        
        if record.exc_info:
            exc_type, exc_value, _exc_tb = record.exc_info
            payload["exc"] = "".join(
                traceback.format_exception_only(exc_type, exc_value)
            ).strip()
        
        return json.dumps(payload, default=self._json_default, ensure_ascii=False, separators=(',', ':'))


class EmitOnceLogger:
    """Logger wrapper that tracks emitted messages to prevent duplicates."""
    
    def __init__(self, base_logger: logging.Logger):
        self._logger = base_logger
        self._emitted_keys: set[str] = set()
        self._lock = threading.Lock()
    
    def _emit_if_new(self, level: str, key: str, msg: str, *args, **kwargs) -> None:
        """Emit log message only if key hasn't been seen before."""
        with self._lock:
            if key not in self._emitted_keys:
                self._emitted_keys.add(key)
                log_method = getattr(self._logger, level.lower())
                log_method(msg, *args, **kwargs)
    
    def info(self, msg: str, key: str | None = None, *args, **kwargs) -> None:
        """Log info message once per key (defaults to message text as key)."""
        emit_key = key or msg
        self._emit_if_new("info", emit_key, msg, *args, **kwargs)
    
    def debug(self, msg: str, key: str | None = None, *args, **kwargs) -> None:
        """Log debug message once per key (defaults to message text as key)."""
        emit_key = key or msg
        self._emit_if_new("debug", emit_key, msg, *args, **kwargs)
    
    def warning(self, msg: str, key: str | None = None, *args, **kwargs) -> None:
        """Log warning message once per key (defaults to message text as key)."""
        emit_key = key or msg
        self._emit_if_new("warning", emit_key, msg, *args, **kwargs)
    
    def error(self, msg: str, key: str | None = None, *args, **kwargs) -> None:
        """Log error message once per key (defaults to message text as key)."""
        emit_key = key or msg
        self._emit_if_new("error", emit_key, msg, *args, **kwargs)


_configured = False
_loggers: dict[str, SanitizingLoggerAdapter] = {}  # AI-AGENT-REF: store adapters
_log_queue: queue.Queue | None = None
_listener: QueueListener | None = None
_LOGGING_LISTENER: QueueListener | None = None

# AI-AGENT-REF: Global thread-safe protection flag to prevent multiple logging setup calls
_LOGGING_LOCK = threading.Lock()
_LOGGING_CONFIGURED = False


def get_rotating_handler(
    path: str,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
) -> logging.Handler:
    """Return a size-rotating file handler. Falls back to stderr on failure."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count
        )
    except OSError as exc:
        logging.getLogger(__name__).error("Cannot open log file %s: %s", path, exc)
        handler = logging.StreamHandler(sys.stderr)
    return handler


def setup_logging(debug: bool = False, log_file: str | None = None) -> logging.Logger:
    """Configure the root logger in an idempotent way."""
    global _configured, _log_queue, _listener, _LOGGING_CONFIGURED

    # AI-AGENT-REF: Thread-safe check of global flag to prevent any duplicate setup
    with _LOGGING_LOCK:
        if _LOGGING_CONFIGURED:
            logging.getLogger(__name__).debug(
                "Logging already configured, skipping duplicate setup"
            )
            return logging.getLogger()

        logger = logging.getLogger()

        if _configured:
            return logger

        # AI-AGENT-REF: Clear any existing handlers to prevent duplicates from other logging configs
        logger.handlers.clear()

        logger.setLevel(logging.DEBUG)

        # Choose formatter based on LOG_COMPACT_JSON setting
        try:
            from ai_trading.config import get_settings
            S = get_settings()
            if S.log_compact_json:
                formatter = CompactJsonFormatter("%(asctime)sZ")
            else:
                formatter = JSONFormatter("%(asctime)sZ")
        except Exception:
            # Fallback to regular formatter if config not available
            formatter = JSONFormatter("%(asctime)sZ")

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
        try:
            if _listener._thread is not None:
                _listener._thread.daemon = True
        except Exception:
            pass
        _LOGGING_LISTENER = _listener
        atexit.register(_safe_shutdown_logging)

        _configured = True
        _LOGGING_CONFIGURED = (
            True  # AI-AGENT-REF: Set global flag to prevent setup in other modules
        )

        # Add validation logging
        logging.getLogger(__name__).info(
            "Logging configured successfully - no duplicates possible"
        )
        return logger


def _safe_shutdown_logging():
    try:
        listener = globals().get("_LOGGING_LISTENER", None)
        try:
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass
                try:
                    t = getattr(listener, "_thread", None)
                    if t is not None and hasattr(t, "join"):
                        t.join(timeout=1.0)
                except Exception:
                    pass
        except Exception:
            pass
        root = logging.getLogger()
        for h in list(root.handlers):
            try:
                root.removeHandler(h)
                try:
                    h.flush()
                except Exception:
                    pass
                try:
                    if hasattr(h, "close"):
                        h.close()
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass


def get_logger(name: str) -> SanitizingLoggerAdapter:
    """Return a named logger wrapped with :class:`SanitizingLoggerAdapter`."""
    if name not in _loggers:
        setup_logging()  # AI-AGENT-REF: ensure root configured once
        base = logging.getLogger(name or _ROOT_LOGGER_NAME)
        base.propagate = True
        base.setLevel(logging.NOTSET)
        _loggers[name] = SanitizingLoggerAdapter(base, {})
    return _loggers[name]


logger = get_logger(__name__)  # AI-AGENT-REF: use sanitizing adapter

# Create emit-once logger instance for preventing duplicate startup messages  
logger_once = EmitOnceLogger(logger)


def get_phase_logger(name: str, phase: str | None = None) -> logging.Logger:
    """
    Return a logger that prefixes messages with a trading 'phase' token so
    dedupe/filters in tests and structured logging can key off it.

    Parameters
    ----------
    name : str
        Logger name
    phase : Optional[str]
        Trading phase identifier (e.g., "ORDER_EXEC", "SIGNAL_GEN")

    Returns
    -------
    logging.Logger
        Logger with phase filtering enabled and propagation disabled
    """
    logger = logging.getLogger(name)
    logger.propagate = False  # Prevent duplicate logging
    if phase:
        # Lightweight filter to stamp phase once per record
        class _PhaseFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                if not hasattr(record, "bot_phase") or not record.bot_phase:
                    record.bot_phase = phase
                return True

        # avoid duplicate filters on repeated calls
        if not any(isinstance(f, _PhaseFilter) for f in logger.filters):
            logger.addFilter(_PhaseFilter())
    return logger


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
    import numpy as np
    import pandas as pd

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
    max_dd = _get_metrics_logger().compute_max_drawdown(equity_curve)
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


def log_trading_event(
    event_type: str,
    symbol: str,
    details: dict[str, any],
    level: str = "INFO",
    include_context: bool = True,
) -> None:
    """
    Log structured trading events with comprehensive context information.

    This function provides standardized logging for trading events with
    optional context capture for debugging and audit purposes. It ensures
    consistent formatting and includes relevant metadata for analysis.

    Parameters
    ----------
    event_type : str
        Type of trading event, such as:
        - 'TRADE_EXECUTED': Order execution events
        - 'SIGNAL_GENERATED': Signal generation events
        - 'RISK_LIMIT_HIT': Risk management events
        - 'DATA_FETCH_ERROR': Data retrieval issues
        - 'API_ERROR': External API failures
        - 'POSITION_UPDATE': Portfolio changes

    symbol : str
        Trading symbol associated with the event (e.g., 'AAPL', 'SPY')
        Use 'SYSTEM' for system-wide events not tied to specific symbols

    details : Dict[str, any]
        Event-specific details containing relevant information:
        - For trades: quantity, price, side, order_id
        - For signals: confidence, strategy, timeframe
        - For errors: error_code, retry_count, stack_trace

    level : str, optional
        Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        Default is 'INFO' for standard events

    include_context : bool, optional
        Whether to include additional system context:
        - Memory usage, CPU load
        - Active positions count
        - Market status
        - Bot configuration state

    Examples
    --------
    >>> # Log successful trade execution
    >>> log_trading_event(
    ...     'TRADE_EXECUTED',
    ...     'AAPL',
    ...     {
    ...         'side': 'buy',
    ...         'quantity': 100,
    ...         'price': 150.25,
    ...         'order_id': 'abc123',
    ...         'execution_time_ms': 250
    ...     }
    ... )

    >>> # Log risk limit violation
    >>> log_trading_event(
    ...     'RISK_LIMIT_HIT',
    ...     'SPY',
    ...     {
    ...         'limit_type': 'position_size',
    ...         'requested': 1000,
    ...         'max_allowed': 500,
    ...         'current_exposure': 0.85
    ...     },
    ...     level='WARNING'
    ... )

    >>> # Log API error with context
    >>> log_trading_event(
    ...     'API_ERROR',
    ...     'SYSTEM',
    ...     {
    ...         'provider': 'alpaca',
    ...         'endpoint': '/v2/orders',
    ...         'error_code': 429,
    ...         'retry_count': 3,
    ...         'error_message': 'Rate limit exceeded'
    ...     },
    ...     level='ERROR',
    ...     include_context=True
    ... )

    Notes
    -----
    - Logs are automatically timestamped with UTC timezone
    - Sensitive information (API keys, passwords) is automatically masked
    - Large objects are truncated to prevent log bloat
    - Context information is cached for performance
    """
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not isinstance(event_type, str) or not event_type.strip():
        logger.error("Invalid event_type: %s", event_type)
        return

    if not isinstance(symbol, str) or not symbol.strip():
        logger.error("Invalid symbol: %s", symbol)
        return

    if not isinstance(details, dict):
        logger.error("Details must be a dictionary, got: %s", type(details))
        return

    # Sanitize details to prevent logging sensitive data
    sanitized_details = _sanitize_log_data(details.copy())

    # Build structured log entry
    log_entry = {
        "event_type": event_type.upper(),
        "symbol": symbol.upper(),
        "timestamp": datetime.now(UTC).isoformat(),
        "details": sanitized_details,
    }

    # Add system context if requested
    if include_context:
        log_entry["context"] = _get_system_context()

    # Convert to JSON for structured logging
    try:
        json_message = json.dumps(log_entry, default=str, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.error("Failed to serialize log entry: %s", e)
        json_message = f"SERIALIZATION_ERROR: {event_type} {symbol}"

    # Log at appropriate level
    log_method = getattr(logger, level.lower(), logger.info)
    log_method("TRADING_EVENT: %s", json_message)


def _sanitize_log_data(data: dict[str, any]) -> dict[str, any]:
    """
    Remove or mask sensitive information from log data.

    Parameters
    ----------
    data : Dict[str, any]
        Raw data dictionary that may contain sensitive information

    Returns
    -------
    Dict[str, any]
        Sanitized data with sensitive fields masked or removed
    """
    sensitive_keys = {
        "api_key",
        "secret_key",
        "password",
        "token",
        "auth",
        "credentials",
        "private_key",
        "session_id",
    }

    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower()

        # Mask sensitive keys
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            sanitized[key] = "***MASKED***"
        # Keep long values intact for observability (no truncation)
        elif isinstance(value, str):
            sanitized[key] = value
        # Handle nested dictionaries
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_log_data(value)
        else:
            sanitized[key] = value

    return sanitized


def _get_system_context() -> dict[str, any]:
    """
    Gather relevant system context for debugging purposes.

    Returns
    -------
    Dict[str, any]
        System context information including performance metrics
    """
    # AI-AGENT-REF: optional psutil context
    try:
        import psutil  # type: ignore
    except Exception as e:
        return {"context_error": f"psutil missing: {e}"}

    try:
        context = {
            "memory_usage_mb": round(psutil.virtual_memory().used / 1024 / 1024, 1),
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "disk_usage_percent": psutil.disk_usage("/").percent,
        }

        context.update(
            {
                "python_version": sys.version.split()[0],
                "log_level": logging.getLogger().level,
                "handlers_count": len(logging.getLogger().handlers),
            }
        )

        return context

    except Exception as e:
        return {"context_error": str(e)}


def setup_enhanced_logging(
    log_file: str = None,
    level: str = "INFO",
    enable_json_format: bool = False,
    enable_performance_logging: bool = True,
    max_file_size_mb: int = 100,
    backup_count: int = 5,
) -> None:
    """
    Setup enhanced logging configuration for the trading bot.

    Configures multiple log handlers including file rotation, JSON formatting,
    and performance monitoring to provide comprehensive logging capabilities.

    Parameters
    ----------
    log_file : str, optional
        Path to the main log file. If None, uses console logging only.
    level : str, optional
        Minimum log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    enable_json_format : bool, optional
        Whether to use JSON formatting for structured logs
    enable_performance_logging : bool, optional
        Whether to enable performance and timing logs
    max_file_size_mb : int, optional
        Maximum size of log files before rotation (default: 100MB)
    backup_count : int, optional
        Number of backup log files to keep (default: 5)
    """
    global _LOGGING_CONFIGURED

    # AI-AGENT-REF: Thread-safe check of global flag to prevent duplicate enhanced logging setup
    with _LOGGING_LOCK:
        if _LOGGING_CONFIGURED:
            logging.getLogger(__name__).debug(
                "Enhanced logging already configured, skipping duplicate setup"
            )
            return

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Clear existing handlers
        root_logger.handlers.clear()

        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = UTCFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Setup file handler if log file specified
        if log_file:
            try:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)

                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_file_size_mb * 1024 * 1024,
                    backupCount=backup_count,
                    encoding="utf-8",
                )

                if enable_json_format:
                    file_formatter = JSONFormatter()
                else:
                    file_formatter = UTCFormatter(
                        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )

                file_handler.setFormatter(file_formatter)
                root_logger.addHandler(file_handler)

            except OSError as e:
                logging.error("Failed to setup file logging: %s", e)

        # Setup performance logging if enabled
        if enable_performance_logging:
            _setup_performance_logging()

        _LOGGING_CONFIGURED = (
            True  # AI-AGENT-REF: Set global flag to prevent duplicate setup
        )
        logging.info(
            "Enhanced logging configured - Level: %s, File: %s, JSON: %s",
            level,
            log_file or "console-only",
            enable_json_format,
        )


def _setup_performance_logging():
    """Setup performance-specific logging handlers."""
    perf_logger = logging.getLogger("performance")

    # Create separate performance log file
    perf_file = os.path.join(os.getenv("BOT_LOG_DIR", "logs"), "performance.log")

    try:
        os.makedirs(os.path.dirname(perf_file), exist_ok=True)
        perf_handler = RotatingFileHandler(
            perf_file, maxBytes=50 * 1024 * 1024, backupCount=3
        )
        perf_formatter = UTCFormatter(
            "%(asctime)s PERF %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        perf_handler.setFormatter(perf_formatter)
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)

    except OSError as e:
        logging.warning("Could not setup performance logging: %s", e)


def validate_logging_setup() -> dict[str, Any]:
    """
    Validate that logging is properly configured without duplicates.

    Returns
    -------
    Dict[str, Any]
        Validation results including handler count and potential issues
    """
    root_logger = logging.getLogger()
    handler_count = len(root_logger.handlers)

    validation_result = {
        "handlers_count": handler_count,
        "is_configured": _LOGGING_CONFIGURED,
        "expected_max_handlers": 2,  # Should be: 1 console + 1 file handler max
        "validation_passed": True,
        "issues": [],
    }

    # Check for too many handlers (potential duplicates)
    if handler_count > 2:
        validation_result["validation_passed"] = False
        validation_result["issues"].append(
            f"Too many handlers detected: {handler_count} (expected ≤ 2)"
        )
        logging.getLogger(__name__).warning(
            "WARNING: %d handlers detected - possible duplicate logging setup",
            handler_count,
        )

    # Check if logging is not configured at all
    if handler_count == 0 and not _LOGGING_CONFIGURED:
        validation_result["validation_passed"] = False
        validation_result["issues"].append("No logging handlers configured")

    # Log validation results
    if validation_result["validation_passed"]:
        logging.getLogger(__name__).info(
            "Logging validation passed - %d handlers configured", handler_count
        )
    else:
        logging.getLogger(__name__).error(
            "Logging validation failed: %s", validation_result["issues"]
        )

    return validation_result


__all__ = [
    "setup_logging",
    "get_logger",
    "get_phase_logger",
    "init_logger",
    "logger",
    "logger_once",
    "log_performance_metrics",
    "log_trading_event",
    "setup_enhanced_logging",
    "validate_logging_setup",
    "EmitOnceLogger",
    "CompactJsonFormatter",
    "with_extra",
    "info_kv",
    "warning_kv",
    "error_kv",
    "SanitizingLoggerAdapter",
]
