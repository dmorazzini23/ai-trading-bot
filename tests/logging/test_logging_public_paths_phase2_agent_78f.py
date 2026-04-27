from __future__ import annotations

import csv
import json
import logging
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

import ai_trading.logging as logmod


class _FakeAdapter:
    def __init__(self, name: str = "ai_trading.tests.fake") -> None:
        self.name = name
        self.logger = logging.getLogger(name)
        self.calls: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = []

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((logging.getLevelName(level).lower(), msg, args, kwargs))

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("info", msg, args, kwargs))

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("warning", msg, args, kwargs))

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("error", msg, args, kwargs))

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("debug", msg, args, kwargs))

    def isEnabledFor(self, level: int) -> bool:  # noqa: N802 - logging API shape
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
        logmod.reset_provider_log_dedupe()


def _record(msg: object, *, name: str = "ai_trading.tests.public") -> logging.LogRecord:
    return logging.LogRecord(name, logging.INFO, __file__, 1, msg, (), None)


def test_env_setup_and_summary_helpers_are_deterministic(monkeypatch, caplog) -> None:
    def raising_get_env(name: str, default: str | None = None) -> str | None:
        raise RuntimeError(f"boom {name} {default}")

    monkeypatch.setitem(
        logmod.sys.modules,
        "ai_trading.config.management",
        SimpleNamespace(get_env=raising_get_env),
    )
    assert logmod._runtime_env("MISSING", "fallback") == "fallback"

    monkeypatch.setattr(
        logmod,
        "_runtime_env",
        lambda name, default=None: {"FINNHUB_API_KEY": "key", "ENABLE_FINNHUB": None}.get(name, default),
    )
    with caplog.at_level(logging.DEBUG, logger="ai_trading.logging"):
        logmod._ensure_finnhub_enabled_flag()
    assert any(record.getMessage() == "ENABLE_FINNHUB_DEFAULTED" for record in caplog.records)

    caplog.clear()
    summary_logger = logging.getLogger("ai_trading.tests.summary")
    with caplog.at_level(logging.INFO, logger=summary_logger.name):
        logmod._emit_throttle_summary(summary_logger, "TOO_FEW", 2)
        logmod._emit_throttle_summary(summary_logger, "ENOUGH", 3)
    assert [record.getMessage() for record in caplog.records] == [
        'LOG_THROTTLE_SUMMARY | suppressed=3 key="ENOUGH"'
    ]


def test_message_throttle_emits_inline_and_flush_summaries(monkeypatch, caplog) -> None:
    assert logmod.MessageThrottleFilter._resolve_throttle_seconds("bad") == 5.0

    throttle = logmod.MessageThrottleFilter(throttle_seconds=10)
    throttle.SUMMARY_INTERVAL = 30.0
    times = iter([100.0, 101.0, 142.0, 143.0, 144.0, 145.0, 175.0])
    monkeypatch.setattr(throttle, "_now", lambda: next(times))

    with caplog.at_level(logging.INFO, logger="ai_trading.tests.throttle.inline"):
        assert throttle.filter(_record("inline, noisy", name="ai_trading.tests.throttle.inline")) is True
        throttle._state["inline, noisy"]["suppressed"] = 2
        throttle._state["inline, noisy"]["last_summary"] = 0.0
        assert throttle.filter(_record("inline, noisy", name="ai_trading.tests.throttle.inline")) is False

    assert any('key="INLINE"' in record.getMessage() for record in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ai_trading.tests.throttle.expire"):
        assert throttle.filter(_record("expire | detail", name="ai_trading.tests.throttle.expire")) is True
        assert throttle.filter(_record("expire | detail", name="ai_trading.tests.throttle.expire")) is False
        assert throttle.filter(_record("expire | detail", name="ai_trading.tests.throttle.expire")) is False
        assert throttle.filter(_record("expire | detail", name="ai_trading.tests.throttle.expire")) is False
        assert throttle.filter(_record("expire | detail", name="ai_trading.tests.throttle.expire")) is True
    assert any('key="EXPIRE"' in record.getMessage() for record in caplog.records)


def test_rate_limited_event_logging_and_flush(monkeypatch, caplog) -> None:
    logger = logging.getLogger("ai_trading.tests.rate.public")

    monkeypatch.setattr(logmod, "_runtime_env", lambda name, default=None: "0")
    with caplog.at_level(logging.INFO, logger=logger.name):
        logmod.log_throttled_event(logger, "RATE_DISABLED", level=logging.INFO, extra={"symbol": "AAPL"})
    assert any(record.getMessage() == "RATE_DISABLED" for record in caplog.records)

    caplog.clear()
    logmod._SUMMARY_LAST_EMIT.clear()
    tracker = logmod.RateLimitedEventTracker(window_seconds=10)
    monkeypatch.setattr(logmod, "_get_rate_limit_tracker", lambda: tracker)
    monkeypatch.setattr(logmod, "_monotonic_time", lambda: 100.0)

    for _ in range(4):
        logmod.log_throttled_event(
            logger,
            "provider empty response",
            level=logging.WARNING,
            extra={"provider": "alpaca", "feed": "iex"},
            message="PROVIDER_EMPTY",
        )

    with caplog.at_level(logging.INFO, logger=logger.name):
        logmod._emit_rate_limit_summaries(tracker.flush(force=True))
    summaries = [record.getMessage() for record in caplog.records if "LOG_THROTTLE_SUMMARY" in record.getMessage()]
    assert summaries == [
        'LOG_THROTTLE_SUMMARY | key="PROVIDER_EMPTY_RESPONSE" suppressed=3 '
        'window_s=10.0 sample_symbol="alpaca" sample_feed="iex"'
    ]


def test_data_quality_event_normalizes_symbols_and_redacts_context(monkeypatch) -> None:
    fake = _FakeAdapter()
    monkeypatch.setattr(logmod, "get_logger", lambda name: fake)

    logmod.log_data_quality_event(
        "missing-bars",
        provider="alpaca",
        severity="not-a-level",
        reason="empty",
        symbols=[" msft ", "AAPL", "msft", ""],
        context={"name": "reserved", "api_key": "secret", "has_secret": True},
    )

    level, msg, _args, kwargs = fake.calls[-1]
    assert level == "warning"
    assert msg == "DATA_QUALITY_EVENT"
    assert kwargs["extra"] == {
        "event": "missing-bars",
        "provider": "alpaca",
        "reason": "empty",
        "symbols": ["AAPL", "MSFT"],
        "context": {"x_name": "reserved", "api_key": logmod._ENV_MASK, "has_secret": True},
    }


def test_trading_event_validation_serialization_and_recursive_redaction(monkeypatch, caplog) -> None:
    fake = _FakeAdapter("ai_trading.logging")
    monkeypatch.setattr(logmod, "get_logger", lambda name: fake)

    logmod.log_trading_event("", "AAPL", {})
    logmod.log_trading_event("fill", "", {})
    logmod.log_trading_event("fill", "AAPL", ["bad"], include_context=False)  # type: ignore[arg-type]

    assert [call[0] for call in fake.calls[:3]] == ["error", "error", "error"]
    assert [call[1] for call in fake.calls[:3]] == [
        "Invalid event_type: %s",
        "Invalid symbol: %s",
        "Details must be a dictionary, got: %s",
    ]

    fake.calls.clear()
    with caplog.at_level(logging.WARNING, logger="ai_trading.logging"):
        logmod.log_trading_event(
            "fill",
            "aapl",
            {
                "api_key": "abc",
                "plain": "visible",
                "nested": {"token": "tok", "quantity": 3},
            },
            level="warning",
            include_context=False,
        )

    records = [
        record for record in caplog.records if record.name == "ai_trading.logging" and record.levelno == logging.WARNING
    ]
    assert records[-1].msg == "TRADING_EVENT: %s"
    payload = json.loads(records[-1].args[0])
    assert payload["event_type"] == "FILL"
    assert payload["symbol"] == "AAPL"
    assert payload["details"] == {
        "api_key": "***MASKED***",
        "plain": "visible",
        "nested": {"token": "***MASKED***", "quantity": 3},
    }
    assert "context" not in payload


def test_performance_metrics_writes_csv_and_handles_failures(monkeypatch, tmp_path) -> None:
    filename = tmp_path / "logs" / "performance.csv"
    monkeypatch.setattr(
        logmod,
        "_get_metrics_logger",
        lambda: SimpleNamespace(compute_max_drawdown=lambda curve: -0.25),
    )

    logmod.log_performance_metrics(
        0.42,
        [100.0, 102.0, 101.0, 104.0],
        "risk-on",
        str(filename),
        as_of=date(2024, 1, 2),
    )

    with filename.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["date"] == "2024-01-02"
    assert rows[0]["exposure_pct"] == "0.42"
    assert rows[0]["max_drawdown"] == "-0.25"
    assert rows[0]["regime"] == "risk-on"

    fake_logger = _FakeAdapter()
    monkeypatch.setattr(logmod, "logger", fake_logger)
    monkeypatch.setattr(logmod.Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("nope")))
    logmod.log_performance_metrics(0.1, [1.0, 2.0], "risk-off", str(tmp_path / "blocked" / "perf.csv"))
    assert fake_logger.calls[-1][0] == "warning"
    assert fake_logger.calls[-1][1] == "Failed to log performance metrics: %s"


def test_setup_enhanced_logging_configures_console_file_and_idempotent_debug(tmp_path, caplog) -> None:
    logmod._LOGGING_CONFIGURED = False
    root = logging.getLogger()
    root.handlers.clear()
    log_file = tmp_path / "enhanced.log"

    logmod.setup_enhanced_logging(
        log_file=str(log_file),
        level="DEBUG",
        enable_json_format=True,
        enable_performance_logging=False,
        max_file_size_mb=1,
        backup_count=1,
    )

    assert logmod._LOGGING_CONFIGURED is True
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 2
    assert isinstance(root.handlers[1].formatter, logmod.JSONFormatter)

    handlers_after_first_setup = list(root.handlers)
    with caplog.at_level(logging.DEBUG, logger="ai_trading.logging"):
        logmod.setup_enhanced_logging(enable_performance_logging=False)
    assert root.handlers == handlers_after_first_setup
