"""Logging helpers for the AI trading bot.

These helpers centralize logger configuration and now include a sanitizer that
prevents collisions with reserved :class:`logging.LogRecord` attributes. Any
key in ``extra`` matching a reserved field is automatically prefixed with
``x_`` to keep structured logging safe. Use :func:`get_logger` to obtain a
sanitizing adapter for all modules.
"""

import atexit
import contextlib
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
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from typing import Any
from ai_trading.exc import COMMON_EXC
from .json_formatter import JSONFormatter
from ai_trading.logging.redact import _ENV_MASK


def _ensure_finnhub_enabled_flag() -> None:
    """Set ENABLE_FINNHUB when a key is present; safe to call multiple times."""

    if os.getenv("FINNHUB_API_KEY") and os.getenv("ENABLE_FINNHUB") is None:
        os.environ["ENABLE_FINNHUB"] = "1"
        logging.getLogger(__name__).debug("ENABLE_FINNHUB_SET", extra={"enabled": True})


_ensure_finnhub_enabled_flag()


def _ensure_single_handler(log: logging.Logger, level: int | None = None) -> None:
    """Ensure no duplicate handler types and attach default if none exist."""

    seen_types: set[type[logging.Handler]] = set()
    handlers = getattr(log, "handlers", [])

    try:
        unique: list[logging.Handler] = list(handlers)
    except TypeError:  # AI-AGENT-REF: allow mocks without iterable handlers
        unique = []

    filtered: list[logging.Handler] = []
    for h in unique:
        h_type = type(h)
        if h_type in seen_types:
            continue
        seen_types.add(h_type)
        if not any(isinstance(f, ExtraSanitizerFilter) for f in h.filters):
            h.addFilter(ExtraSanitizerFilter())
        if _THROTTLE_FILTER not in h.filters:
            h.addFilter(_THROTTLE_FILTER)
        filtered.append(h)

    if not filtered:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        h.setFormatter(fmt)
        h.addFilter(ExtraSanitizerFilter())
        h.addFilter(_THROTTLE_FILTER)
        filtered.append(h)

    log.handlers = filtered

    if level is not None:
        log.setLevel(level)


_RESERVED_LOGRECORD_KEYS = {
    "name",
    "msg",
    "message",
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
}


def _sanitize_extra(extra: dict[str, Any] | None) -> dict[str, Any]:
    """Rename reserved ``LogRecord`` keys with ``x_`` prefix."""
    if not extra:
        return {}
    return {k if k not in _RESERVED_LOGRECORD_KEYS else f"x_{k}": v for k, v in extra.items()}


_SENSITIVE_EXTRA_KEYS = ("api_key", "secret")


def sanitize_extra(extra: dict[str, Any] | None) -> dict[str, Any]:
    """Sanitize ``extra`` mapping.

    Reserved ``LogRecord`` keys are prefixed with ``x_`` and values of keys
    containing sensitive tokens (``api_key``, ``secret`` or ``url``) are
    redacted.
    """
    cleaned = _sanitize_extra(extra)
    out: dict[str, Any] = {}
    for k, v in cleaned.items():
        if any(tok in k.lower() for tok in _SENSITIVE_EXTRA_KEYS):
            out[k] = _ENV_MASK
        else:
            out[k] = v
    return out


class ExtraSanitizerFilter(logging.Filter):
    """Filter that sanitizes arbitrary ``LogRecord`` extras."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - lightweight
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED_LOGRECORD_KEYS}
        if extras:
            sanitized = sanitize_extra(extras)
            for k in extras:
                record.__dict__.pop(k, None)
            record.__dict__.update(sanitized)
        return True


class MessageThrottleFilter(logging.Filter):
    """Suppress identical log messages emitted within a short window."""

    SUMMARY_INTERVAL = 30.0

    def __init__(self, throttle_seconds: float | None = None) -> None:
        super().__init__()
        self.throttle_seconds = self._resolve_throttle_seconds(throttle_seconds)
        self._lock = threading.Lock()
        self._state: dict[str, dict[str, float | int]] = {}

    @staticmethod
    def _resolve_throttle_seconds(value: float | None) -> float:
        """Resolve throttle interval honouring millisecond and legacy knobs."""

        if value is not None:
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                return 0.5

        raw_ms = os.getenv("LOG_TIMING_THROTTLE_MS")
        if raw_ms is not None:
            try:
                return max(float(raw_ms) / 1000.0, 0.0)
            except (TypeError, ValueError):
                return 0.5

        raw_seconds = os.getenv("LOG_THROTTLE_SECONDS")
        try:
            return max(float(raw_seconds), 0.0) if raw_seconds is not None else 0.5
        except (TypeError, ValueError):
            return 0.5

    def _now(self) -> float:
        return time.monotonic()

    @staticmethod
    def _quote_message(message: str) -> str:
        escaped = message.replace('"', '\\"')
        return f'"{escaped}"'

    def _normalize_stage_timing(self, record: logging.LogRecord) -> None:
        if record.msg == "STAGE_TIMING":
            stage = getattr(record, "stage", None)
            elapsed_ms = getattr(record, "elapsed_ms", None)
            if stage is not None and elapsed_ms is not None:
                try:
                    ms_value = int(elapsed_ms)
                except (TypeError, ValueError):
                    ms_value = elapsed_ms
                record.msg = f"STAGE_TIMING | stage={stage} ms={ms_value}"
                record.args = ()

    def _emit_summary(self, record: logging.LogRecord, message: str, suppressed: int) -> None:
        summary = f"LOG_THROTTLE_SUMMARY | suppressed={suppressed} message={self._quote_message(message)}"
        logging.getLogger(record.name or _ROOT_LOGGER_NAME).info(summary)

    def filter(self, record: logging.LogRecord) -> bool:
        if self.throttle_seconds <= 0:
            return True

        if isinstance(record.msg, str) and record.msg.startswith("LOG_THROTTLE_SUMMARY"):
            return True

        self._normalize_stage_timing(record)
        message = record.getMessage()
        if not isinstance(message, str) or not message:
            return True

        now = self._now()
        with self._lock:
            state = self._state.get(message)
            if state is None:
                self._state[message] = {
                    "last_emit": now,
                    "suppressed": 0,
                    "last_summary": now,
                }
                return True

            last_emit = float(state.get("last_emit", 0.0))
            suppressed = int(state.get("suppressed", 0))
            last_summary = float(state.get("last_summary", 0.0))

            if now - last_emit < self.throttle_seconds:
                suppressed += 1
                state["suppressed"] = suppressed
                if now - last_summary >= self.SUMMARY_INTERVAL and suppressed:
                    self._emit_summary(record, message, suppressed)
                    state["suppressed"] = 0
                    state["last_summary"] = now
                return False

            if suppressed and now - last_summary >= self.SUMMARY_INTERVAL:
                self._emit_summary(record, message, suppressed)
                state["suppressed"] = 0
                state["last_summary"] = now

            state["last_emit"] = now
            return True


_THROTTLE_FILTER = MessageThrottleFilter()


class SanitizingLoggerAdapter(logging.LoggerAdapter):
    """Adapter that sanitizes ``extra`` keys to avoid LogRecord collisions."""

    def __getattribute__(self, name):
        """Delegate missing attributes to the underlying logger."""
        try:
            return super().__getattribute__(name)
        except AttributeError:
            logger = super().__getattribute__("logger")
            return getattr(logger, name)

    @property
    def handlers(self) -> list[logging.Handler]:
        """Delegate handler list to the underlying logger."""
        # If the named logger has no handlers, mirror root handlers for parity in tests
        return self.logger.handlers or logging.getLogger().handlers

    @handlers.setter
    def handlers(self, value: list[logging.Handler]) -> None:
        """Replace handlers on the underlying logger."""
        self.logger.handlers = value

    @property
    def filters(self) -> list[logging.Filter]:
        """Delegate filter list to the underlying logger."""
        return self.logger.filters

    @filters.setter
    def filters(self, value: list[logging.Filter]) -> None:
        """Replace filters on the underlying logger."""
        self.logger.filters = value

    @property
    def propagate(self) -> bool:
        """Whether the underlying logger propagates messages."""
        return self.logger.propagate

    @propagate.setter
    def propagate(self, value: bool) -> None:
        """Set propagation behaviour for the underlying logger."""
        self.logger.propagate = value

    def process(self, msg, kwargs):
        extra = kwargs.get("extra")
        if extra is not None:
            kwargs["extra"] = sanitize_extra(extra)
        return (msg, kwargs)


_ROOT_LOGGER_NAME = "ai_trading"


def with_extra(logger, level, msg: str, *, extra: dict | None = None, **_ignored):
    """Log `msg` with structured fields via `extra` only (no arbitrary kwargs)."""
    payload = {} if extra is None else dict(extra)
    logger.log(level, msg, extra=payload)


def info_kv(logger, msg: str, *, extra: dict | None = None):
    """Log at INFO level with structured key-value fields."""
    with_extra(logger, logging.INFO, msg, extra=extra)


def warning_kv(logger, msg: str, *, extra: dict | None = None):
    """Log at WARNING level with structured key-value fields."""
    with_extra(logger, logging.WARNING, msg, extra=extra)


def error_kv(logger, msg: str, *, extra: dict | None = None):
    """Log at ERROR level with structured key-value fields."""
    with_extra(logger, logging.ERROR, msg, extra=extra)


def _get_config():
    """Lazy import config management to avoid import-time dependencies.

    Returns the imported module so callers can access attributes safely.
    """
    from ai_trading.config import management as config

    return config


def _get_metrics_logger():
    """Lazy import metrics_logger to avoid import-time dependencies.

    Returns the module which provides log_* helpers.
    """
    from ai_trading.telemetry import metrics_logger

    return metrics_logger


def _mask_secret(value: str) -> str:
    """Non-throwing redactor for secret-like values (config-independent).

    Short values fully masked; longer values keep a tiny prefix/suffix.
    """
    try:
        s = "" if value is None else str(value)
        n = len(s)
        if n == 0:
            return ""
        if n <= 6:
            return "***"
        return f"{s[:2]}***{s[-2:]}"
    except COMMON_EXC:
        return "***"


logging.Formatter.converter = time.gmtime


class UTCFormatter(logging.Formatter):
    """Formatter with UTC timestamps and structured phase tags."""

    converter = time.gmtime


class CompactJsonFormatter(JSONFormatter):
    """Compact JSON log formatter that drops large extra payloads."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        essential_fields = {"bot_phase", "present", "timestamp"}
        for k, v in record.__dict__.items():
            if k in essential_fields and k not in {
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
            }:
                if "key" in k.lower() or "secret" in k.lower():
                    v = _mask_secret(v)
                payload[k] = v
        if record.exc_info:
            exc_type, exc_value, _exc_tb = record.exc_info
            payload["exc"] = "".join(traceback.format_exception_only(exc_type, exc_value)).strip()
        return json.dumps(payload, default=self._json_default, ensure_ascii=False, separators=(",", ":"))


class EmitOnceLogger:
    """Logger wrapper that tracks emitted messages to prevent duplicates."""

    def __init__(self, base_logger: logging.Logger):
        self._logger = base_logger
        self._emitted_keys: dict[str, tuple[date, int]] = {}
        self._lock = threading.Lock()

    def _emit_if_new(self, level: str, key: str, msg: str, *args, **kwargs) -> None:
        """Emit log message only once per key each day."""
        today = date.today()
        with self._lock:
            last_date, count = self._emitted_keys.get(key, (None, 0))
            if last_date != today:
                count = 0
            count += 1
            self._emitted_keys[key] = (today, count)
            if count > 1:
                return
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
_loggers: dict[str, SanitizingLoggerAdapter] = {}
_log_queue: queue.Queue | None = None
_listener: QueueListener | None = None
_LOGGING_LISTENER: QueueListener | None = None
# Use a re-entrant lock so setup_logging can be safely invoked if it is
# indirectly re-entered during module import (e.g. config -> logging).
# A standard Lock would deadlock when setup_logging imports modules that in
# turn call get_logger(), which also attempts to acquire this lock.
_LOGGING_LOCK = threading.RLock()
_LOGGING_CONFIGURED = False


def ensure_logging_configured(level: int | None = None) -> None:
    """Configure root logging once."""
    global _LOGGING_CONFIGURED
    root = logging.getLogger()
    if _LOGGING_CONFIGURED or root.handlers:
        return
    logging.basicConfig(level=level or logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    _LOGGING_CONFIGURED = True


def get_rotating_handler(path: str, max_bytes: int = 5000000, backup_count: int = 5) -> logging.Handler:
    """Return a size-rotating file handler. Falls back to stderr on failure."""
    try:
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    except PermissionError as exc:
        logging.getLogger(__name__).warning("Cannot create log directory %s: %s", os.path.dirname(path), exc)
        return logging.StreamHandler(sys.stderr)
    try:
        handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
    except OSError as exc:
        logging.getLogger(__name__).error("Cannot open log file %s: %s", path, exc)
        handler = logging.StreamHandler(sys.stderr)
    return handler


def setup_logging(debug: bool = False, log_file: str | None = None) -> logging.Logger:
    """Configure the root logger in an idempotent way.

    The ``debug`` flag is retained for backward compatibility but the log
    level should now be controlled via settings or the ``LOG_LEVEL``
    environment variable.
    """
    global _configured, _log_queue, _listener, _LOGGING_CONFIGURED
    _ensure_finnhub_enabled_flag()
    if _LOGGING_CONFIGURED and _configured:
        return logging.getLogger()
    if debug:
        # Deprecated: retain for callers that still pass ``debug=True``.
        # The effective log level is derived from configuration instead.
        pass
    with _LOGGING_LOCK:
        if _LOGGING_CONFIGURED and _configured:
            return logging.getLogger()
        if _listener is not None:
            _listener = None
        if _LOGGING_CONFIGURED and _configured:
            return logging.getLogger()
        logger = logging.getLogger()
        if _configured:
            return logger
        _configured = True
        _ensure_single_handler(logger)
        logger.handlers.clear()
        S = None
        level_name_env_default = os.getenv("LOG_LEVEL", "INFO")
        try:
            from ai_trading.config import get_settings, management as config

            S = get_settings()
            level_name = getattr(S, "log_level", level_name_env_default)
            yf_level_name = getattr(
                S,
                "log_level_yfinance",
                config.get_env("LOG_LEVEL_YFINANCE", "WARNING"),
            )
            formatter = (
                CompactJsonFormatter("%Y-%m-%dT%H:%M:%SZ")
                if bool(getattr(S, "log_compact_json", False))
                else JSONFormatter("%Y-%m-%dT%H:%M:%SZ")
            )
        except COMMON_EXC:
            from ai_trading.config import management as config

            level_name = level_name_env_default
            yf_level_name = config.get_env("LOG_LEVEL_YFINANCE", "WARNING")
            formatter = JSONFormatter("%Y-%m-%dT%H:%M:%SZ")
        level = getattr(logging, str(level_name).upper(), logging.INFO)
        logger.setLevel(level)
        yf_level = getattr(logging, str(yf_level_name).upper(), logging.WARNING)
        logging.getLogger("yfinance").setLevel(yf_level)
        # Reduce noisy HTTP libraries unless explicitly requested
        try:
            http_env_default = config.get_env("LOG_LEVEL_HTTP", "WARNING")
        except Exception:
            http_env_default = os.getenv("LOG_LEVEL_HTTP", "WARNING")
        http_level_name = getattr(S, "log_level_http", http_env_default)
        http_level = getattr(logging, str(http_level_name).upper(), logging.WARNING)
        for _name in ("urllib3", "requests"):
            logging.getLogger(_name).setLevel(http_level)

        class _PhaseFilter(logging.Filter):

            def filter(self, record: logging.LogRecord) -> bool:
                if not hasattr(record, "bot_phase"):
                    record.bot_phase = "GENERAL"
                return True

        handlers: list[logging.Handler] = []
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        stream_handler.addFilter(_PhaseFilter())
        from ai_trading.logging_filters import SecretFilter

        secret_filter = SecretFilter()
        extra_filter = ExtraSanitizerFilter()
        stream_handler.addFilter(secret_filter)
        stream_handler.addFilter(extra_filter)
        handlers.append(stream_handler)
        if log_file:
            rotating_handler = get_rotating_handler(log_file)
            rotating_handler.setFormatter(formatter)
            rotating_handler.setLevel(logging.INFO)
            rotating_handler.addFilter(_PhaseFilter())
            rotating_handler.addFilter(secret_filter)
            rotating_handler.addFilter(extra_filter)
            handlers.append(rotating_handler)
        # In tests, avoid background queue threads to prevent timeout interference
        if os.getenv("PYTEST_RUNNING") == "1" or os.getenv("LOG_DISABLE_QUEUE") == "1":
            logger.handlers = handlers
            _listener = None
        else:
            _log_queue = queue.Queue(-1)
            queue_handler = QueueHandler(_log_queue)
            queue_handler.setLevel(level)
            queue_handler.addFilter(_PhaseFilter())
            queue_handler.addFilter(secret_filter)
            queue_handler.addFilter(extra_filter)
            logger.handlers = [queue_handler]
            _listener = QueueListener(_log_queue, *handlers, respect_handler_level=True)
            _listener.start()
        atexit.register(_safe_shutdown_logging)
        atexit.register(logging.shutdown)
        _LOGGING_CONFIGURED = True
        logging.getLogger(__name__).info("Logging configured successfully - no duplicates possible")
        for h in handlers:
            with contextlib.suppress(Exception):
                h.flush()
        # Apply filters for noisy third-party libraries after configuration is complete.
        from ai_trading.logging.setup import _apply_library_filters

        _apply_library_filters()
        return logger


def configure_logging(debug: bool = False, log_file: str | None = None) -> SanitizingLoggerAdapter:
    """Configure logging and return a sanitizing adapter.

    The function is resilient to configuration/setting import failures and
    guarantees that both global configuration flags are set.  The returned
    logger is always wrapped with :class:`SanitizingLoggerAdapter` so callers
    can safely attach structured extras.
    """
    global _configured, _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        _configured = True
        return get_logger(_ROOT_LOGGER_NAME)
    try:
        setup_logging(debug=debug, log_file=log_file)
    except COMMON_EXC:
        ensure_logging_configured()
    _configured = True
    _LOGGING_CONFIGURED = True
    return get_logger(_ROOT_LOGGER_NAME)


def _safe_shutdown_logging():
    try:
        listener = globals().get("_LOGGING_LISTENER", None)
        try:
            if listener is not None:
                try:
                    listener.stop()
                except COMMON_EXC:
                    pass
                try:
                    t = getattr(listener, "_thread", None)
                    if t is not None and hasattr(t, "join"):
                        t.join(timeout=1.0)
                except COMMON_EXC:
                    pass
        except COMMON_EXC:
            pass
        root = logging.getLogger()
        for h in list(root.handlers):
            try:
                root.removeHandler(h)
                try:
                    h.flush()
                except COMMON_EXC:
                    pass
                try:
                    if hasattr(h, "close"):
                        h.close()
                except COMMON_EXC:
                    pass
            except COMMON_EXC:
                pass
    except COMMON_EXC:
        pass


def get_logger(name: str) -> SanitizingLoggerAdapter:
    "Return a named logger wrapped with :class:`SanitizingLoggerAdapter`."
    if _LOGGING_CONFIGURED and name in _loggers:
        return _loggers[name]
    if not _LOGGING_CONFIGURED:
        configure_logging()
    _ensure_single_handler(logging.getLogger())
    if name not in _loggers:
        base = logging.getLogger(name or _ROOT_LOGGER_NAME)
        base.propagate = True
        base.setLevel(logging.NOTSET)
        _loggers[name] = SanitizingLoggerAdapter(base, {})
    return _loggers[name]


logger = SanitizingLoggerAdapter(logging.getLogger(__name__), {})
logger_once = EmitOnceLogger(logger)


def log_compact_json(*_a: Any, **_k: Any) -> None:  # pragma: no cover - stub
    """Stub to avoid ``AttributeError`` when compact JSON logging is disabled."""
    return None


def log_market_fetch(*_a: Any, **_k: Any) -> None:  # pragma: no cover - stub
    """Stub to avoid ``AttributeError`` when market fetch logging is disabled."""
    return None


def log_finnhub_disabled(symbol: str) -> None:
    """Debug once when Finnhub is disabled for ``symbol``."""
    logger_once.debug(
        "FINNHUB_DISABLED",
        key=f"FINNHUB_DISABLED:{symbol}",
        extra={"symbol": symbol},
    )


def warn_finnhub_disabled_no_data(
    symbol: str,
    *,
    timeframe: str | None = None,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
) -> None:
    """Log once when Finnhub is disabled and a fetch produced no data.

    The ``EmitOnceLogger`` originally deduped purely on *symbol*, which meant
    separate requests for different windows could silence each other when
    tests (or runtime checks) exercised multiple scenarios in the same
    process. Extend the key with the request window so distinct fetch attempts
    during the same run still produce the expected audit log.
    """

    def _normalize_range_component(value: datetime | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    key_parts: list[str] = [symbol]
    if timeframe:
        key_parts.append(str(timeframe))
    start_key = _normalize_range_component(start)
    end_key = _normalize_range_component(end)
    if start_key or end_key:
        key_parts.append(f"{start_key or ''}->{end_key or ''}")
    dedupe_key = ":".join(key_parts)
    extra = {
        "symbol": symbol,
        "recommendation": "set ENABLE_FINNHUB=1 and provide FINNHUB_API_KEY",
    }
    if timeframe:
        extra["timeframe"] = str(timeframe)
    if start_key:
        extra["start"] = start_key
    if end_key:
        extra["end"] = end_key

    logger_once.info(
        "FINNHUB_DISABLED_NO_DATA",
        key=f"FINNHUB_DISABLED_NO_DATA:{dedupe_key}",
        extra=extra,
    )


def log_fetch_attempt(provider: str, *, status: int | None = None, error: str | None = None, **extra: Any) -> None:
    """Log a market data fetch attempt and its outcome.

    Parameters
    ----------
    provider : str
        Data source name, e.g. ``"alpaca"`` or ``"yfinance"``.
    status : Optional[int]
        HTTP status code returned by the provider, if available.
    error : Optional[str]
        Error message when the attempt fails or returns an unexpected payload.
    **extra : dict
        Additional context about the request (symbol, feed, timeframe, request
        parameters, correlation IDs, remaining retries, backoff delay, etc.).

    Notes
    -----
    This helper centralizes fetch attempt logging so callers can capture
    success, empty responses, and error codes with consistent structured
    metadata.
    """
    payload: dict[str, Any] = {"provider": provider, **extra}
    if status is not None:
        payload["status"] = status
    if error is not None:
        payload["error"] = error
        logger.warning("FETCH_ATTEMPT", extra=payload)
    else:
        logger.info("FETCH_ATTEMPT", extra=payload)


def log_backup_provider_used(
    provider: str,
    *,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> None:
    """Log and record when the backup data provider serves a window."""
    payload: dict[str, Any] = {
        "provider": provider,
        "symbol": symbol,
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    try:
        from ai_trading.data.metrics import backup_provider_used as _backup_counter

        _backup_counter.labels(provider=provider, symbol=symbol).inc()
    except COMMON_EXC:
        logger.debug(
            "METRIC_BACKUP_PROVIDER_FAILED",
            extra={"provider": provider, "symbol": symbol},
        )
    logger.info("BACKUP_PROVIDER_USED", extra=payload)


def log_empty_retries_exhausted(
    provider: str,
    *,
    symbol: str,
    timeframe: str,
    feed: str | None = None,
    retries: int | None = None,
) -> None:
    """Log when repeated data fetches yield empty results."""
    payload: dict[str, Any] = {
        "provider": provider,
        "symbol": symbol,
        "timeframe": timeframe,
        "remaining_retries": 0,
    }
    if feed is not None:
        payload["feed"] = feed
    if retries is not None:
        payload["retries"] = retries
    logger.error("EMPTY_RETRIES_EXHAUSTED", extra=payload)


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
    logger = get_logger(name)
    logger.propagate = False
    if phase:

        class _PhaseFilter(logging.Filter):

            def filter(self, record: logging.LogRecord) -> bool:
                if not hasattr(record, "bot_phase") or not record.bot_phase:
                    record.bot_phase = phase
                return True

        if not any((isinstance(f, _PhaseFilter) for f in logger.filters)):
            logger.addFilter(_PhaseFilter())
    return logger


def init_logger(log_file: str) -> logging.Logger:
    """Wrapper used by utilities to initialize logging."""
    return setup_logging(log_file=log_file)


def log_performance_metrics(
    exposure_pct: float,
    equity_curve: list[float],
    regime: str,
    filename: str | None = None,
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
        sharpe = roll.mean() / (roll.std(ddof=0) or 1e-09) * np.sqrt(252 / 20)
        downside = roll[roll < 0]
        sortino = roll.mean() / (downside.std(ddof=0) or 1e-09) * np.sqrt(252 / 20)
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
    if filename is None:
        from pathlib import Path

        from ai_trading.paths import LOG_DIR

        filename = str((LOG_DIR / "performance.csv").resolve())
    try:
        os.makedirs(os.path.dirname(filename), mode=0o700, exist_ok=True)
    except PermissionError as exc:
        logger.warning("Failed to log performance metrics: %s", exc)
        return
    try:
        new = not os.path.exists(filename)
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rec.keys())
            if new:
                writer.writeheader()
            writer.writerow(rec)
    except COMMON_EXC as exc:
        logger.warning("Failed to log performance metrics: %s", exc)


def log_trading_event(
    event_type: str, symbol: str, details: dict[str, any], level: str = "INFO", include_context: bool = True
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

    Notes
    -----
    - Logs are automatically timestamped with UTC timezone
    - Sensitive information (API keys, passwords) is automatically masked
    - Large objects are truncated to prevent log bloat
    - Context information is cached for performance
    """
    logger = get_logger(__name__)
    if not isinstance(event_type, str) or not event_type.strip():
        logger.error("Invalid event_type: %s", event_type)
        return
    if not isinstance(symbol, str) or not symbol.strip():
        logger.error("Invalid symbol: %s", symbol)
        return
    if not isinstance(details, dict):
        logger.error("Details must be a dictionary, got: %s", type(details))
        return
    sanitized_details = _sanitize_log_data(details.copy())
    log_entry = {
        "event_type": event_type.upper(),
        "symbol": symbol.upper(),
        "timestamp": datetime.now(UTC).isoformat(),
        "details": sanitized_details,
    }
    if include_context:
        log_entry["context"] = _get_system_context()
    try:
        json_message = json.dumps(log_entry, default=str, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.error("Failed to serialize log entry: %s", e)
        json_message = f"SERIALIZATION_ERROR: {event_type} {symbol}"
    base_logger = getattr(logger, "logger", logger)
    log_method = getattr(base_logger, level.lower(), base_logger.info)
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
    sensitive_keys = {"api_key", "secret_key", "password", "token", "auth", "credentials", "private_key", "session_id"}
    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any((sensitive in key_lower for sensitive in sensitive_keys)):
            sanitized[key] = "***MASKED***"
        elif isinstance(value, str):
            sanitized[key] = value
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
    try:
        import psutil
    except COMMON_EXC as e:
        return {"context_error": f"psutil missing: {e}"}
    try:
        context = {
            "memory_usage_mb": round(psutil.virtual_memory().used / 1024 / 1024, 1),
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "disk_usage_percent": psutil.disk_usage("/").percent,
        }
        root = logging.getLogger()
        handlers = getattr(root, "handlers", [])
        try:
            handler_count = len(handlers)
        except TypeError:
            handler_count = 0
        context.update(
            {
                "python_version": sys.version.split()[0],
                "log_level": root.level,
                "handlers_count": handler_count,
            }
        )
        return context
    except COMMON_EXC as e:
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
    with _LOGGING_LOCK:
        if _LOGGING_CONFIGURED:
            logging.getLogger(__name__).debug("Enhanced logging already configured, skipping duplicate setup")
            return
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        root_logger.handlers.clear()
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = UTCFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        from ai_trading.logging_filters import SecretFilter

        secret_filter = SecretFilter()
        console_handler.addFilter(secret_filter)
        root_logger.addHandler(console_handler)
        if log_file:
            try:
                os.makedirs(os.path.dirname(log_file), mode=0o700, exist_ok=True)
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
                        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
                    )
                file_handler.setFormatter(file_formatter)
                file_handler.addFilter(secret_filter)
                root_logger.addHandler(file_handler)
            except PermissionError as e:
                logging.warning("Failed to setup file logging: %s", e)
            except OSError as e:
                logging.error("Failed to setup file logging: %s", e)
        if enable_performance_logging:
            _setup_performance_logging()
        _LOGGING_CONFIGURED = True
        logging.info(
            "Enhanced logging configured - Level: %s, File: %s, JSON: %s",
            level,
            log_file or "console-only",
            enable_json_format,
        )


def _setup_performance_logging():
    """Setup performance-specific logging handlers."""
    perf_logger = get_logger("performance")
    perf_file = os.path.join(os.getenv("BOT_LOG_DIR", "logs"), "performance.log")
    try:
        os.makedirs(os.path.dirname(perf_file), mode=0o700, exist_ok=True)
    except PermissionError as e:
        logging.warning("Could not setup performance logging: %s", e)
        return
    try:
        perf_handler = RotatingFileHandler(perf_file, maxBytes=50 * 1024 * 1024, backupCount=3)
        perf_formatter = UTCFormatter("%(asctime)s PERF %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        perf_handler.setFormatter(perf_formatter)
        from ai_trading.logging_filters import SecretFilter

        perf_handler.addFilter(SecretFilter())
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False
    except OSError as e:
        logging.warning("Could not setup performance logging: %s", e)


def dedupe_stream_handlers(log: logging.Logger) -> int:
    """Public helper to dedupe duplicate ``StreamHandler`` instances on a logger.

    Returns the final total number of handlers on the logger after dedupe.
    """
    before = len(log.handlers)
    _ensure_single_handler(log)
    after = len(log.handlers)
    if after < before:
        get_logger(__name__).info("LOGGING_SETUP_DEDUPED", extra={"before": before, "after": after})
    return after


def validate_logging_setup(logger: logging.Logger | None = None, *, dedupe: bool = False) -> dict[str, Any]:
    """Validate logging configuration and (optionally) dedupe handlers.

    Parameters
    ----------
    logger: Optional[logging.Logger]
        Logger to validate; defaults to the root logger when ``None``.
    dedupe: bool
        When ``True``, duplicates are removed via :func:`dedupe_stream_handlers`.

    Returns
    -------
    dict[str, Any]
        Validation results including handler counts and issues.
    """
    log = logging.getLogger() if logger is None else logger
    before_count = len(log.handlers)
    if dedupe:
        after_count = dedupe_stream_handlers(log)
        did_dedupe = after_count < before_count
    else:
        after_count = before_count
        did_dedupe = False
    validation_result = {
        "handlers_count": after_count,
        "before_handlers_count": before_count,
        "deduped": did_dedupe,
        "is_configured": _LOGGING_CONFIGURED,
        "expected_max_handlers": 2,
        "validation_passed": True,
        "issues": [],
    }
    if after_count > 2:
        validation_result["validation_passed"] = False
        validation_result["issues"].append(f"Too many handlers detected: {after_count} (expected ≤ 2)")
        get_logger(__name__).warning("WARNING: %d handlers detected - possible duplicate logging setup", after_count)
    if after_count == 0 and (not _LOGGING_CONFIGURED):
        validation_result["validation_passed"] = False
        validation_result["issues"].append("No logging handlers configured")
    if validation_result["validation_passed"]:
        get_logger(__name__).info("Logging validation passed - %d handlers configured", after_count)
    else:
        get_logger(__name__).error("Logging validation failed: %s", validation_result["issues"])
    return validation_result


__all__ = [
    "setup_logging",
    "configure_logging",
    "get_logger",
    "get_phase_logger",
    "init_logger",
    "logger",
    "logger_once",
    "log_fetch_attempt",
    "log_backup_provider_used",
    "log_empty_retries_exhausted",
    "log_performance_metrics",
    "log_trading_event",
    "log_finnhub_disabled",
    "warn_finnhub_disabled_no_data",
    "log_compact_json",
    "log_market_fetch",
    "setup_enhanced_logging",
    "validate_logging_setup",
    "dedupe_stream_handlers",
    "EmitOnceLogger",
    "CompactJsonFormatter",
    "with_extra",
    "info_kv",
    "warning_kv",
    "error_kv",
    "SanitizingLoggerAdapter",
    "sanitize_extra",
]
