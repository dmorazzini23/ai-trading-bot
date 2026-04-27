from __future__ import annotations

import itertools
import logging
import queue
import sys
from datetime import datetime, UTC
from logging.handlers import QueueHandler
from types import SimpleNamespace
from typing import Any

import pytest

import ai_trading.logging as logmod


class _FakeAdapter:
    def __init__(self) -> None:
        self.logger = logging.getLogger("ai_trading.tests.fake78g")
        self.calls: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = []
        self.filters: list[logging.Filter] = []

    def addFilter(self, filter_: logging.Filter) -> None:  # noqa: N802 - logging API shape
        self.filters.append(filter_)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("debug", msg, args, kwargs))

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("info", msg, args, kwargs))

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("warning", msg, args, kwargs))

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("error", msg, args, kwargs))

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((logging.getLevelName(level).lower(), msg, args, kwargs))


class _SecretFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.secret_filter_seen = True
        return True


@pytest.fixture(autouse=True)
def _restore_logging_globals():
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    original_configured = logmod._configured
    original_logging_configured = logmod._LOGGING_CONFIGURED
    original_listener = logmod._listener
    original_logging_listener = logmod._LOGGING_LISTENER
    original_log_queue = logmod._log_queue
    original_loggers = dict(logmod._loggers)
    original_last_monotonic = logmod._LAST_MONOTONIC
    original_summary_last_emit = dict(logmod._SUMMARY_LAST_EMIT)
    original_throttle_state = dict(logmod._THROTTLE_FILTER._state)
    try:
        yield
    finally:
        logmod.shutdown_queue_listener()
        root.handlers = original_handlers
        root.setLevel(original_level)
        logmod._configured = original_configured
        logmod._LOGGING_CONFIGURED = original_logging_configured
        logmod._listener = original_listener
        logmod._LOGGING_LISTENER = original_logging_listener
        logmod._log_queue = original_log_queue
        logmod._loggers.clear()
        logmod._loggers.update(original_loggers)
        logmod._LAST_MONOTONIC = original_last_monotonic
        logmod._SUMMARY_LAST_EMIT.clear()
        logmod._SUMMARY_LAST_EMIT.update(original_summary_last_emit)
        logmod._THROTTLE_FILTER._state.clear()
        logmod._THROTTLE_FILTER._state.update(original_throttle_state)
        logmod.reset_provider_log_dedupe()


def test_runtime_env_monotonic_and_pytest_queue_bridge(monkeypatch) -> None:
    monkeypatch.setitem(
        logmod.sys.modules,
        "ai_trading.config.management",
        SimpleNamespace(get_env=lambda name, default=None: "" if name == "EMPTY" else 42),
    )
    assert logmod._runtime_env("EMPTY", "fallback") == "fallback"
    assert logmod._runtime_env("NUMBER") == "42"

    logmod._LAST_MONOTONIC = None
    monkeypatch.setattr(logmod.time, "monotonic", lambda: 123.5)
    assert logmod._monotonic_time() == 123.5

    def raise_runtime() -> float:
        raise RuntimeError("timer unavailable")

    monkeypatch.setattr(logmod.time, "monotonic", raise_runtime)
    assert logmod._monotonic_time() == 123.5

    logmod._LAST_MONOTONIC = None
    monkeypatch.setattr(logmod.time, "time", lambda: 77.0)
    assert logmod._monotonic_time() == 77.0

    root = logging.getLogger()
    stream = logging.StreamHandler()
    qh = QueueHandler(queue.Queue())
    root.handlers = [qh, stream]
    monkeypatch.setattr(logmod, "_runtime_env", lambda name, default=None: "1")
    logmod._ensure_pytest_logging_bridge()
    assert root.handlers == [stream]


def test_single_handler_dedupes_filters_and_handles_non_iterable_handlers(monkeypatch) -> None:
    monkeypatch.setitem(logmod.sys.modules, "ai_trading.logging_filters", SimpleNamespace(SecretFilter=_SecretFilter))

    logger = logging.getLogger("ai_trading.tests.single_handler78g")
    first_stream = logging.StreamHandler()
    duplicate_stream = logging.StreamHandler()
    null_handler = logging.NullHandler()
    logger.handlers = [first_stream, duplicate_stream, null_handler]

    logmod._ensure_single_handler(logger, level=logging.DEBUG)

    assert logger.level == logging.DEBUG
    assert logger.handlers == [first_stream, null_handler]
    assert any(isinstance(f, _SecretFilter) for f in first_stream.filters)
    assert any(isinstance(f, logmod.ExtraSanitizerFilter) for f in first_stream.filters)
    assert logmod._THROTTLE_FILTER in first_stream.filters

    odd_logger = logging.getLogger("ai_trading.tests.odd_handlers78g")
    odd_logger.handlers = object()  # type: ignore[assignment]
    logmod._ensure_single_handler(odd_logger)
    assert len(odd_logger.handlers) == 1
    assert isinstance(odd_logger.handlers[0], logging.StreamHandler)


def test_configure_logging_falls_back_and_logger_cache(monkeypatch) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    logmod._configured = False
    logmod._LOGGING_CONFIGURED = False
    logmod._loggers.clear()
    calls: list[str] = []

    def fail_setup(*args: Any, **kwargs: Any) -> None:
        raise ValueError("settings unavailable")

    def fallback_basic_config(level: int | None = None) -> None:
        calls.append(f"fallback:{level}")
        root.handlers = [logging.NullHandler()]

    monkeypatch.setattr(logmod, "setup_logging", fail_setup)
    monkeypatch.setattr(logmod, "ensure_logging_configured", fallback_basic_config)

    adapter = logmod.configure_logging(debug=True, log_file="ignored.log")

    assert calls == ["fallback:None"]
    assert logmod._configured is True
    assert logmod._LOGGING_CONFIGURED is True
    assert adapter is logmod.get_logger("ai_trading")
    assert logmod.get_logger("ai_trading") is adapter


def test_setup_logging_queue_branch_and_idempotent_return(monkeypatch, tmp_path) -> None:
    import ai_trading.config as config_pkg
    from ai_trading.config import management as config_management
    import ai_trading.logging.setup as logging_setup

    root = logging.getLogger()
    root.handlers.clear()
    logmod._configured = False
    logmod._LOGGING_CONFIGURED = False
    logmod._listener = None
    logmod._LOGGING_LISTENER = None

    class _Settings:
        log_level = "ERROR"
        log_level_yfinance = "INFO"
        log_level_http = "DEBUG"
        log_compact_json = True

    class _Listener:
        def __init__(self, log_queue: queue.Queue, *handlers: logging.Handler, respect_handler_level: bool) -> None:
            self.log_queue = log_queue
            self.handlers = handlers
            self.respect_handler_level = respect_handler_level
            self.started = False
            self._thread = None

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False

    listeners: list[_Listener] = []

    def listener_factory(*args: Any, **kwargs: Any) -> _Listener:
        listener = _Listener(*args, **kwargs)
        listeners.append(listener)
        return listener

    monkeypatch.setattr(config_pkg, "get_settings", lambda: _Settings())
    monkeypatch.setattr(config_management, "get_env", lambda name, default=None, **kwargs: default)
    monkeypatch.setattr(logmod, "_runtime_env", lambda name, default=None: None if name == "PYTEST_RUNNING" else default)
    monkeypatch.setattr(logmod, "QueueListener", listener_factory)
    monkeypatch.setattr(logmod.atexit, "register", lambda func: None)
    monkeypatch.setattr(logging_setup, "_apply_library_filters", lambda: None)

    configured = logmod.setup_logging(debug=True, log_file=str(tmp_path / "queue.log"))

    assert configured is root
    assert root.level == logging.ERROR
    assert logging.getLogger("yfinance").level == logging.INFO
    assert logging.getLogger("urllib3").level == logging.DEBUG
    assert logging.getLogger("requests").level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], QueueHandler)
    assert listeners[0].started is True
    assert listeners[0].respect_handler_level is True
    assert len(listeners[0].handlers) == 2
    assert logmod._LOGGING_LISTENER is listeners[0]
    assert logmod.setup_logging() is root


def test_rate_limit_tracker_reset_and_summary_report_branches(monkeypatch, caplog) -> None:
    tracker = logmod.RateLimitedEventTracker(window_seconds=10)
    tracker.update_window(-1)
    assert tracker.record("disabled", logger_name="ai_trading.tests.rate78g")[0] is True

    tracker.update_window(10)
    monotonic_values = itertools.chain(
        [0.0, 1.0, 2.0, 3.0, 4.0, 11.0, 100.0, 100.0, 3701.0, 3701.0],
        itertools.repeat(3701.0),
    )
    monkeypatch.setattr(logmod, "_monotonic_time", lambda: next(monotonic_values))

    assert tracker.record("window report", logger_name="ai_trading.tests.rate78g")[0] is True
    for symbol in ("AAPL", "MSFT", "NVDA"):
        should_log, summaries = tracker.record(
            "window report",
            logger_name="ai_trading.tests.rate78g",
            extra={"symbol": symbol, "feed": "iex"},
        )
        assert should_log is False
        assert summaries == []

    assert tracker.flush(force=False) == []
    expired = tracker.flush(force=False)
    assert len(expired) == 1
    assert expired[0].key == "WINDOW_REPORT"
    assert expired[0].suppressed == 3
    assert expired[0].sample_symbol == "NVDA"
    assert tracker.flush(force=True) == []

    logmod._SUMMARY_LAST_EMIT.clear()
    too_small = logmod.RateLimitedSummary("small", 2, "ai_trading.tests.rate78g", 10.0)
    emitted = logmod.RateLimitedSummary("again", 3, "ai_trading.tests.rate78g", 10.0)
    with caplog.at_level(logging.INFO, logger="ai_trading.tests.rate78g"):
        logmod._emit_rate_limit_summaries([too_small, emitted])
        logmod._emit_rate_limit_summaries([emitted])
        logmod._emit_rate_limit_summaries([emitted])
    messages = [record.getMessage() for record in caplog.records if "LOG_THROTTLE_SUMMARY" in record.getMessage()]
    assert messages == [
        'LOG_THROTTLE_SUMMARY | key="again" suppressed=3 window_s=10.0',
        'LOG_THROTTLE_SUMMARY | key="again" suppressed=3 window_s=10.0',
    ]

    tracker.reset()
    assert tracker.flush(force=True) == []


def test_flush_log_throttle_summaries_respects_namespace_and_provider_threshold(monkeypatch, caplog) -> None:
    monkeypatch.setattr(logmod, "_runtime_env", lambda name, default=None: "1" if name == "PYTEST_RUNNING" else default)
    logmod._THROTTLE_FILTER._state.clear()
    logmod._THROTTLE_FILTER._state.update(
        {
            "keep message": {
                "suppressed": 3,
                "logger_name": "ai_trading.keep.child",
                "last_summary": 0.0,
                "seen_in_cycle": True,
            },
            "skip message": {
                "suppressed": 4,
                "logger_name": "ai_trading.skip",
                "last_summary": 0.0,
                "seen_in_cycle": True,
            },
            "too small": {
                "suppressed": 1,
                "logger_name": "ai_trading.keep",
                "last_summary": 0.0,
                "seen_in_cycle": True,
            },
        }
    )

    for _ in range(2):
        logmod.record_provider_log_suppressed("provider below threshold")
    for _ in range(3):
        logmod.record_provider_log_suppressed("provider outage | body")

    with caplog.at_level(logging.INFO):
        logmod.flush_log_throttle_summaries(logging.getLogger("ai_trading.keep"))

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert 'key="KEEP_MESSAGE"' in text
    assert 'key="SKIP_MESSAGE"' not in text
    assert 'key="TOO_SMALL"' not in text
    assert 'key="PROVIDER_OUTAGE"' in text
    assert "provider below threshold" not in text
    assert logmod._THROTTLE_FILTER._state["skip message"]["suppressed"] == 4


def test_finnhub_once_per_window_data_quality_and_performance_edges(monkeypatch, tmp_path) -> None:
    fake_once = _FakeAdapter()
    monkeypatch.setattr(logmod, "logger_once", fake_once)
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)

    logmod.log_finnhub_disabled("AAPL")
    logmod.warn_finnhub_disabled_no_data("MSFT", timeframe="1Min", start=start, end=end)

    assert fake_once.calls[0] == (
        "debug",
        "FINNHUB_DISABLED",
        (),
        {"key": "FINNHUB_DISABLED:AAPL", "extra": {"symbol": "AAPL"}},
    )
    assert fake_once.calls[1][0] == "info"
    assert fake_once.calls[1][3]["key"] == (
        "FINNHUB_DISABLED_NO_DATA:MSFT:1Min:2024-01-02T14:30:00+00:00->2024-01-02T15:30:00+00:00"
    )

    fake_logger = _FakeAdapter()
    monkeypatch.setattr(logmod, "get_logger", lambda name: fake_logger)

    class _BadSymbols:
        def __iter__(self):
            raise ValueError("no symbols")

    logmod.log_data_quality_event(
        "bad-symbols",
        provider="alpaca",
        severity="INFO",
        symbols=_BadSymbols(),
    )
    assert fake_logger.calls[-1] == (
        "info",
        "DATA_QUALITY_EVENT",
        (),
        {"extra": {"event": "bad-symbols", "provider": "alpaca"}},
    )

    metrics_called = False

    def fail_if_called() -> None:
        nonlocal metrics_called
        metrics_called = True

    monkeypatch.setattr(logmod, "_get_metrics_logger", fail_if_called)
    logmod.log_performance_metrics(0.5, [], "flat", str(tmp_path / "unused.csv"))
    assert metrics_called is False


def test_backup_provider_phase_logger_shutdown_and_system_context_edges(monkeypatch) -> None:
    metric_calls: list[tuple[str, str, bool]] = []

    def record_metric(provider: str, symbol: str, *, increment: bool) -> None:
        metric_calls.append((provider, symbol, increment))

    monkeypatch.setitem(
        sys.modules,
        "ai_trading.data.fetch.metrics",
        SimpleNamespace(inc_backup_provider_used=record_metric),
    )
    active_logger = _FakeAdapter()
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = datetime(2024, 1, 2, 15, 30, tzinfo=UTC)

    payload = logmod.log_backup_provider_used(
        "backup",
        symbol="AAPL",
        timeframe="1Min",
        start=start,
        end=end,
        extra={"reason": "configured_source_override", "drop": None},
        logger=active_logger,
    )

    assert metric_calls == [("backup", "AAPL", True)]
    assert payload["reason"] == "configured_source_override"
    assert "drop" not in payload
    assert active_logger.calls[-1][0] == "info"

    def fail_metric(provider: str, symbol: str, *, increment: bool) -> None:
        raise ValueError("metrics down")

    monkeypatch.setitem(
        sys.modules,
        "ai_trading.data.fetch.metrics",
        SimpleNamespace(inc_backup_provider_used=fail_metric),
    )
    active_logger.calls.clear()
    logmod.log_backup_provider_used(
        "backup",
        symbol="MSFT",
        timeframe="1Day",
        start=start,
        end=end,
        extra={"fallback_reason": "unexpected"},
        logger=active_logger,
    )
    assert [call[0] for call in active_logger.calls] == ["debug", "warning"]

    phase_logger = _FakeAdapter()
    monkeypatch.setattr(logmod, "get_logger", lambda name: phase_logger)
    returned = logmod.get_phase_logger("ai_trading.tests.phase78g", phase="ORDER")
    record = logging.LogRecord("ai_trading.tests.phase78g", logging.INFO, __file__, 1, "msg", (), None)
    assert returned.filters[-1].filter(record) is True
    assert record.bot_phase == "ORDER"

    class _BadListener:
        _thread = SimpleNamespace(join=lambda timeout=None: (_ for _ in ()).throw(ValueError("join failed")))

        def stop(self) -> None:
            raise ValueError("stop failed")

    logmod._LOGGING_LISTENER = _BadListener()  # type: ignore[assignment]
    logmod.shutdown_queue_listener()
    assert logmod._LOGGING_LISTENER is None

    class _Memory:
        used = 10 * 1024 * 1024
        percent = 12.5

    class _Disk:
        percent = 55.0

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            virtual_memory=lambda: _Memory(),
            cpu_percent=lambda interval=None: 3.25,
            disk_usage=lambda path: _Disk(),
        ),
    )
    root = logging.getLogger()

    class _BadLenHandlers(list[logging.Handler]):
        def __len__(self) -> int:
            raise TypeError("no length")

    root.handlers = _BadLenHandlers()
    context = logmod._get_system_context()
    assert context["memory_usage_mb"] == 10.0
    assert context["handlers_count"] == 0

    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: (_ for _ in ()).throw(ValueError("psutil bad"))),
    )
    assert logmod._get_system_context() == {"context_error": "psutil bad"}


def test_enhanced_and_performance_logging_file_setup_failures(monkeypatch, tmp_path) -> None:
    logmod._LOGGING_CONFIGURED = False
    root = logging.getLogger()
    root.handlers.clear()
    warnings: list[tuple[str, tuple[Any, ...]]] = []
    errors: list[tuple[str, tuple[Any, ...]]] = []
    monkeypatch.setattr(logmod.logging, "warning", lambda msg, *args, **kwargs: warnings.append((msg, args)))
    monkeypatch.setattr(logmod.logging, "error", lambda msg, *args, **kwargs: errors.append((msg, args)))
    monkeypatch.setattr(logmod.Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("deny")))

    logmod.setup_enhanced_logging(
        log_file=str(tmp_path / "blocked" / "app.log"),
        enable_performance_logging=False,
    )

    assert logmod._LOGGING_CONFIGURED is True
    assert warnings[-1][0] == "Failed to setup file logging: %s"
    assert str(warnings[-1][1][0]) == "deny"
    assert len(root.handlers) == 1

    warnings.clear()
    logmod._LOGGING_CONFIGURED = True
    monkeypatch.setattr(logmod.Path, "mkdir", lambda *args, **kwargs: None)

    def fail_rotating_handler(*args: Any, **kwargs: Any) -> logging.Handler:
        raise OSError("disk full")

    monkeypatch.setattr(logmod, "RotatingFileHandler", fail_rotating_handler)
    monkeypatch.setattr(logmod, "_runtime_env", lambda name, default=None: str(tmp_path))

    logmod._setup_performance_logging()

    assert warnings[-1][0] == "Could not setup performance logging: %s"
    assert str(warnings[-1][1][0]) == "disk full"

    errors.clear()
    logmod._LOGGING_CONFIGURED = False
    root.handlers.clear()
    monkeypatch.setattr(logmod.Path, "mkdir", lambda *args, **kwargs: None)
    logmod.setup_enhanced_logging(
        log_file=str(tmp_path / "oserror" / "app.log"),
        enable_performance_logging=False,
    )
    assert errors[-1][0] == "Failed to setup file logging: %s"
    assert str(errors[-1][1][0]) == "disk full"
