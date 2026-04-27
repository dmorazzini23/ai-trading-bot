from __future__ import annotations

import datetime as dt
import io
import itertools
import json
import logging
import sys
from enum import Enum
from types import MappingProxyType, SimpleNamespace
from uuid import UUID

import pytest

import ai_trading.logging as logmod
import ai_trading.utils.base as base


class _Side(Enum):
    BUY = "buy"


class _FakeLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[object, ...], dict[str, object]]] = []

    def debug(self, msg: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("debug", msg, args, kwargs))

    def info(self, msg: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("info", msg, args, kwargs))

    def warning(self, msg: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("warning", msg, args, kwargs))

    def error(self, msg: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("error", msg, args, kwargs))

    def critical(self, msg: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("critical", msg, args, kwargs))

    def log(self, level: int, msg: str, *args: object, **kwargs: object) -> None:
        self.calls.append((logging.getLevelName(level).lower(), msg, args, kwargs))


def _record(
    name: str = "ai_trading.tests.logging",
    msg: str = "event",
    *,
    level: int = logging.INFO,
) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, msg, (), None)


def test_sanitize_adapter_formatter_and_emit_once(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = logmod.sanitize_extra(
        {
            "name": "reserved",
            "api_key": "abc123",
            "has_secret": True,
            "nested_secret": "hidden",
            "plain": 7,
        }
    )
    assert payload == {
        "x_name": "reserved",
        "api_key": logmod._ENV_MASK,
        "has_secret": True,
        "nested_secret": logmod._ENV_MASK,
        "plain": 7,
    }

    adapter = logmod.SanitizingLoggerAdapter(logging.getLogger("ai_trading.tests.adapter"), {})
    msg, kwargs = adapter.process("hello", {"extra": {"msg": "reserved", "secret": "hidden"}})
    assert msg == "hello"
    assert kwargs["extra"]["x_msg"] == "reserved"
    assert kwargs["extra"]["secret"] == logmod._ENV_MASK
    assert adapter.handlers == (adapter.logger.handlers or logging.getLogger().handlers)

    record = _record(msg="compact")
    record.bot_phase = "TEST"
    record.present = True
    record.timestamp = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    record.exc_info = (ValueError, ValueError("bad"), None)
    formatted = json.loads(logmod.CompactJsonFormatter("%Y-%m-%dT%H:%M:%SZ").format(record))
    assert formatted["msg"] == "compact"
    assert formatted["bot_phase"] == "TEST"
    assert formatted["present"] is True
    assert formatted["exc"] == "ValueError: bad"

    fake = _FakeLogger()
    once = logmod.EmitOnceLogger(fake)
    days = iter(
        [
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 1),
            dt.date(2024, 1, 2),
            dt.date(2024, 1, 2),
            dt.date(2024, 1, 2),
            dt.date(2024, 1, 2),
            dt.date(2024, 1, 2),
        ]
    )
    monkeypatch.setattr(logmod, "_utc_today", lambda: next(days))
    once.info("same", key="k")
    once.info("same", key="k")
    once.info("same", key="k")
    once.debug("debug")
    once.warning("warning")
    once.error("error")
    once.critical("critical")
    assert [call[0] for call in fake.calls] == [
        "info",
        "info",
        "debug",
        "warning",
        "error",
        "critical",
    ]


def test_message_and_rate_limit_throttling_paths(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    throttle = logmod.MessageThrottleFilter(throttle_seconds=10)
    times = iter([0.0, 1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(throttle, "_now", lambda: next(times))

    first = _record("ai_trading.tests.throttle", "STAGE_TIMING")
    first.stage = "fetch"
    first.elapsed_ms = "12.8"
    assert throttle.filter(first) is True
    assert first.getMessage() == "STAGE_TIMING | stage=fetch ms=12.8"

    for _ in range(3):
        repeat = _record("ai_trading.tests.throttle", "STAGE_TIMING")
        repeat.stage = "fetch"
        repeat.elapsed_ms = "12.8"
        assert throttle.filter(repeat) is False

    with caplog.at_level(logging.INFO, logger="ai_trading.tests.throttle"):
        throttle.flush_cycle(namespace="ai_trading.tests")
    assert any("LOG_THROTTLE_SUMMARY" in rec.getMessage() for rec in caplog.records)
    assert logmod.MessageThrottleFilter._quote_message('a "quote"') == '"a \\"quote\\""'
    assert logmod.MessageThrottleFilter(throttle_seconds=0).filter(_record()) is True

    tracker = logmod.RateLimitedEventTracker(window_seconds=2)
    monotonic = itertools.chain([10.0, 10.5, 11.0, 12.1], itertools.repeat(12.1))
    monkeypatch.setattr(logmod, "_monotonic_time", lambda: next(monotonic))
    assert tracker.record("Fetch attempt, noisy", logger_name="ai_trading.tests.rate")[0] is True
    assert tracker.record(
        "Fetch attempt, noisy",
        logger_name="ai_trading.tests.rate",
        extra={"symbol": "AAPL", "timeframe": "1Min"},
    )[0] is False
    assert tracker.record("Fetch attempt, noisy", logger_name="ai_trading.tests.rate")[0] is False
    should_log, summaries = tracker.record("Fetch attempt, noisy", logger_name="ai_trading.tests.rate")
    assert should_log is True
    assert summaries[0].key == "FETCH_ATTEMPT"
    assert summaries[0].suppressed == 2
    assert tracker.flush(force=True) == []

    logmod._SUMMARY_LAST_EMIT.clear()
    summary = logmod.RateLimitedSummary(
        key="manual key",
        suppressed=3,
        logger_name="ai_trading.tests.rate",
        window_s=2.0,
        sample_symbol="MSFT",
        sample_feed="1Day",
    )
    with caplog.at_level(logging.INFO, logger="ai_trading.tests.rate"):
        logmod._emit_rate_limit_summaries([summary])
        logmod._emit_rate_limit_summaries([summary])
    summary_messages = [rec.getMessage() for rec in caplog.records if "manual key" in rec.getMessage()]
    assert len(summary_messages) == 1
    assert 'sample_symbol="MSFT"' in summary_messages[0]


def test_logging_misc_helpers_and_event_wrappers(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(logmod, "_runtime_env", lambda name, default=None: "bad")
    assert logmod.MessageThrottleFilter().throttle_seconds == 5.0
    monkeypatch.setattr(
        logmod,
        "_runtime_env",
        lambda name, default=None: "2500" if name == "LOG_TIMING_THROTTLE_MS" else None,
    )
    assert logmod.MessageThrottleFilter().throttle_seconds == 2.5
    monkeypatch.setattr(
        logmod,
        "_runtime_env",
        lambda name, default=None: "7" if name == "LOG_THROTTLE_SECONDS" else None,
    )
    assert logmod.MessageThrottleFilter().throttle_seconds == 7.0

    deduper = logmod.LogDeduper()
    assert deduper.should_log("k", ttl_s=0, now=1.0) is True
    assert deduper.should_log("k", ttl_s=10, now=2.0) is False
    assert deduper.should_log("k", ttl_s=10, now=12.0) is True
    deduper.reset()
    assert deduper.should_log("k", ttl_s=10, now=13.0) is True

    logmod.reset_provider_log_dedupe()
    for _ in range(3):
        logmod.record_provider_log_suppressed("provider unavailable, retrying")
    with caplog.at_level(logging.INFO, logger="ai_trading"):
        logmod._flush_provider_log_summaries()
    assert any("PROVIDER_UNAVAILABLE" in rec.getMessage() for rec in caplog.records)

    fake = _FakeLogger()
    logmod.info_kv(fake, "info", extra={"a": 1})
    logmod.warning_kv(fake, "warning")
    logmod.error_kv(fake, "error")
    assert [call[0] for call in fake.calls] == ["info", "warning", "error"]

    assert logmod._truthy_flag(" yes ") is True
    assert logmod._truthy_flag(None) is False
    assert logmod._logger_namespace(logging.LoggerAdapter(logging.getLogger("ai_trading.tests.ns"), {})) == (
        "ai_trading.tests.ns"
    )
    assert logmod._namespace_matches("ai_trading.tests", "ai_trading.tests.child") is True
    assert logmod._namespace_matches("ai_trading.other", "ai_trading.tests.child") is False
    assert logmod._mask_secret("") == ""
    assert logmod._mask_secret("short") == "***"
    assert logmod._mask_secret("long-secret") == "lo***et"

    captured: list[tuple[str, int, dict[str, object] | None, str | None]] = []

    def fake_throttled(
        logger: logging.Logger | logging.LoggerAdapter,
        key: str,
        *,
        level: int = logging.WARNING,
        extra: dict[str, object] | None = None,
        message: str | None = None,
    ) -> None:
        captured.append((key, level, extra, message))

    monkeypatch.setattr(logmod, "log_throttled_event", fake_throttled)
    logmod.log_fetch_attempt("alpaca", status=500, error="empty body", symbol="AAPL")
    assert captured == [
        (
            "FETCH_ATTEMPT_alpaca_500_empty_body",
            logging.WARNING,
            {"provider": "alpaca", "symbol": "AAPL", "status": 500, "error": "empty body"},
            "FETCH_ATTEMPT",
        )
    ]

    fake.calls.clear()
    logmod.log_empty_retries_exhausted("alpaca", symbol="AAPL", timeframe="1Min", feed="iex", retries=2)
    assert any(rec.getMessage() == "EMPTY_RETRIES_EXHAUSTED" for rec in caplog.records)

    class _Listener:
        def __init__(self) -> None:
            self.stopped = False
            self._thread = SimpleNamespace(join=lambda timeout=None: None)

        def stop(self) -> None:
            self.stopped = True

    listener = _Listener()
    logmod._listener = listener  # type: ignore[assignment]
    logmod.shutdown_queue_listener()
    assert listener.stopped is True
    assert logmod._listener is None


def test_logging_setup_fallbacks_and_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(logmod.Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("no")))
    assert isinstance(logmod.get_rotating_handler(str(tmp_path / "no" / "app.log")), logging.StreamHandler)

    monkeypatch.setattr(logmod.Path, "mkdir", lambda *args, **kwargs: None)

    def fail_rotating_handler(*args: object, **kwargs: object) -> logging.Handler:
        raise OSError("disk")

    monkeypatch.setattr(logmod, "RotatingFileHandler", fail_rotating_handler)
    assert isinstance(logmod.get_rotating_handler(str(tmp_path / "app.log")), logging.StreamHandler)

    log = logging.getLogger("ai_trading.tests.validation")
    log.handlers = [logging.StreamHandler(), logging.StreamHandler(), logging.NullHandler()]
    with caplog.at_level(logging.WARNING, logger="ai_trading.logging"):
        result = logmod.validate_logging_setup(log, dedupe=False)
    assert result["validation_passed"] is False
    assert result["handlers_count"] == 3

    deduped = logmod.validate_logging_setup(log, dedupe=True)
    assert deduped["deduped"] is True
    assert deduped["handlers_count"] <= 2

    class _BadContext:
        def keys(self) -> list[str]:
            raise ValueError("bad mapping")

    assert logmod._safe_context_payload(_BadContext()) == {}


def test_base_logging_order_stale_and_time_helpers(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = base.PhaseLoggerAdapter(logging.getLogger("ai_trading.tests.phase"), {"bot_phase": "SCAN"})
    _, kwargs = adapter.process("phase", {})
    assert kwargs["extra"]["bot_phase"] == "SCAN"
    assert kwargs["extra"]["timestamp"].tzinfo is dt.UTC

    with caplog.at_level(logging.DEBUG, logger="ai_trading.utils.base"):
        base.log_warning("HEALTH_STALE_DATA", exc=ValueError("old"), extra={"symbol": "AAPL"})
        base.log_warning("NORMAL_WARNING", extra={"symbol": "MSFT"})
    assert any(rec.levelno == logging.DEBUG and "HEALTH_STALE_DATA" in rec.getMessage() for rec in caplog.records)
    assert any(rec.levelno == logging.WARNING and rec.getMessage() == "NORMAL_WARNING" for rec in caplog.records)

    phase_logger = base.get_phase_logger("ai_trading.tests.phase.factory", "TRADE")
    _, phase_kwargs = phase_logger.process("factory", {})
    assert phase_kwargs["extra"]["bot_phase"] == "TRADE"

    fake_debug = _FakeLogger()

    class _HealthModule:
        @staticmethod
        def snapshot_basic() -> dict[str, float]:
            return {"cpu_percent": 12.34}

    monkeypatch.setitem(sys.modules, "ai_trading.monitoring.system_health", _HealthModule)
    base.log_cpu_usage(fake_debug, note="unit")
    assert fake_debug.calls[-1][1] == "CPU_USAGE%s: %.2f%%"

    class _NotException:
        AlpacaOrderHTTPError = int

    monkeypatch.setitem(sys.modules, "ai_trading.alpaca_api", _NotException)
    assert base.AlpacaOrderHTTPError in base._alpaca_http_error_types()

    base._STALE_CACHE.clear()
    tick = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    now = iter([100.0, 101.0, 500.0])
    monkeypatch.setattr(base.time, "time", lambda: next(now))
    assert base.should_log_stale("AAPL", tick, ttl=300) is True
    assert base.should_log_stale("AAPL", tick, ttl=300) is False
    assert base.should_log_stale("AAPL", tick, ttl=300) is True

    monkeypatch.setattr(base.random, "uniform", lambda low, high: high)
    assert base.backoff_delay(3, base=2.0, cap=5.0, jitter=0.1) == pytest.approx(5.5)
    assert base.backoff_delay(3, base=2.0, cap=5.0, jitter=0.0) == 5.0

    order = SimpleNamespace(
        created=tick,
        side=_Side.BUY,
        order_id=UUID("00000000-0000-0000-0000-000000000001"),
        qty=3,
        active=True,
        note={"nested": "value"},
    )
    text = base.format_order_for_log(order)
    assert "created=2024-01-01T00:00:00+00:00" in text
    assert "side=buy" in text
    assert "order_id=00000000-0000-0000-0000-000000000001" in text
    assert base.format_order_for_log(None) == ""

    lock = base._CallableLock()
    assert lock.acquire(blocking=False) is True
    assert lock.locked() is True
    lock.release()
    with lock() as held:
        assert held.locked() is True
    assert lock.locked() is False

    assert base.ensure_utc(dt.date(2024, 1, 2)) == dt.datetime(2024, 1, 2, tzinfo=dt.UTC)
    assert base.ensure_utc(dt.datetime(2024, 1, 2, 12)).tzinfo is dt.UTC
    with pytest.raises(TypeError):
        base.ensure_utc("2024-01-02")  # type: ignore[arg-type]

    class _FakeStamp:
        def to_pydatetime(self) -> dt.datetime:
            return dt.datetime(2024, 1, 6, 12, tzinfo=dt.UTC)

    assert base.is_weekend(_FakeStamp()) is True
    assert base.is_market_holiday(dt.datetime(2024, 12, 25, tzinfo=dt.UTC)) is True


def test_base_dataframe_datetime_and_column_helpers(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pd = pytest.importorskip("pandas")
    import numpy as np

    assert base.get_latest_close(None) == 0.0
    assert base.get_latest_close(pd.DataFrame({"c": ["bad", 10.5]})) == 10.5
    assert base.get_latest_close(pd.DataFrame({"close": [np.inf]})) == 0.0
    assert base.get_latest_close(pd.DataFrame({"open": [1]})) == 0.0

    assert list(base.safe_to_datetime(None)) == []
    millis = base.safe_to_datetime([1_700_000_000_000, 0])
    assert millis.tz is not None
    naive = base.safe_to_datetime(["2024-01-01T00:00:00Z"], utc=False)
    assert naive.tz is None

    logmod.logger_once._emitted_keys.clear()
    invalid = base.safe_to_datetime(["not-a-date"], context="unit")
    assert invalid.isna().all()

    valid = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"],
            "open": [1, 2],
            "high": [2, 3],
            "low": [0, 1],
            "close": [1.5, 2.5],
            "volume": [100, 200],
        }
    )
    base.validate_ohlcv(valid)
    with pytest.raises(ValueError, match="missing columns"):
        base.validate_ohlcv(valid.drop(columns=["volume"]))
    with pytest.raises(ValueError, match="not monotonic"):
        base.validate_ohlcv(valid.iloc[::-1].reset_index(drop=True))
    with pytest.raises(ValueError, match="OHLC columns incomplete"):
        base.validate_ohlcv(valid[["timestamp", "open", "high", "low", "volume"]], required=["timestamp"])

    columns = pd.DataFrame(
        {
            "Open": [1],
            "High": [2],
            "Low": [0],
            "Close": [1.5],
            "Volume": [100],
            "symbol": ["AAPL"],
            "returns": [0.1],
            "Datetime": pd.to_datetime(["2024-01-01T00:00:00Z"]),
        }
    )
    assert base.get_ohlcv_columns(columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert base.get_symbol_column(columns) == "symbol"
    assert base.get_return_column(columns) == "returns"
    assert base.get_datetime_column(columns) == "Datetime"
    assert base.get_order_column(pd.DataFrame({"SIDE": ["buy"]}), "side") == "SIDE"

    with pytest.raises(TypeError):
        base.get_column(pd.DataFrame({"bad": [1]}), ["bad"], "bad", dtype="O")
    with pytest.raises(ValueError):
        base.get_column(pd.DataFrame({"dup": [1, 1]}), ["dup"], "dup", must_be_unique=True)

    with caplog.at_level(logging.WARNING, logger="ai_trading.utils.base"):
        assert base.get_symbol_column(pd.DataFrame({"symbol": ["AAPL", "AAPL"]})) is None
    assert "_safe_get_column failed for symbol" in caplog.text

    assert base.validate_ohlcv_basic(columns) is True
    assert base.validate_ohlcv_basic(pd.DataFrame()) is False
    assert base.validate_ohlcv_basic(columns.drop(columns=["Volume"])) is False

    monkeypatch.setenv("HEALTH_MIN_ROWS", "2")
    assert base.health_check([1, 2]) is True
    assert base.health_check(object()) is False


def test_base_market_port_process_and_fetch_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    pd = pytest.importorskip("pandas")

    open_calls: list[dt.datetime] = []

    def fake_is_market_open(moment: dt.datetime) -> bool:
        open_calls.append(moment)
        return len(open_calls) == 2

    monkeypatch.setattr(base, "is_market_open", fake_is_market_open)
    start = dt.datetime(2024, 1, 2, 10, 1, tzinfo=dt.UTC)
    end = dt.datetime(2024, 1, 2, 10, 0, tzinfo=dt.UTC)
    assert base.market_open_between(start, end) is True
    assert open_calls[0] == end

    import pandas_market_calendars as mcal

    class _NoMarketOpenCalendar:
        @staticmethod
        def schedule(start_date: dt.date, end_date: dt.date):
            return pd.DataFrame({"not_open": []})

    monkeypatch.setattr(mcal, "get_calendar", lambda name: _NoMarketOpenCalendar())
    next_open = base.next_market_open(dt.datetime(2024, 1, 5, 21, 0, tzinfo=dt.UTC))
    assert next_open.date() == dt.date(2024, 1, 8)
    assert next_open.time() == base.MARKET_OPEN_TIME

    port = base.get_free_port()
    assert isinstance(port, int)
    assert base.get_free_port(start=1, end=0) is None

    monkeypatch.setattr(base.os, "listdir", lambda path: ["abc", "123"] if path == "/proc" else ["4"])
    monkeypatch.setattr(base.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(base.os, "readlink", lambda path: "socket:[777]")
    assert base._pid_from_inode("777") == 123

    monkeypatch.setattr(base, "_pid_from_inode", lambda inode: 456 if inode == "999" else None)
    proc_tcp = (
        "sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
        "0: 0100007F:2329 00000000:0000 0A 0 0 0 0 0 999\n"
    )
    proc_tcp6 = "sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"

    def fake_open(path: str, *args: object, **kwargs: object) -> io.StringIO:
        return io.StringIO(proc_tcp if path == "/proc/net/tcp" else proc_tcp6)

    with monkeypatch.context() as open_patch:
        open_patch.setattr("builtins.open", fake_open)
        assert base.get_pid_on_port(9001) == 456

    monkeypatch.setattr(base, "alpaca_get", lambda *args, **kwargs: {"ap": "12.34"})
    monkeypatch.setattr(base, "get_execution_feed", lambda: "iex")
    assert base.get_current_price("AAPL") == 12.34

    monkeypatch.setattr(base, "alpaca_get", lambda *args, **kwargs: {"ap": 0})
    fetch_mod = sys.modules["ai_trading.data.fetch"]
    monkeypatch.setattr(fetch_mod, "get_daily_df", lambda *args, **kwargs: pd.DataFrame({"Close": [9.87]}))
    assert base.get_current_price("MSFT") == 9.87

    monkeypatch.setattr(base, "alpaca_get", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("down")))
    monkeypatch.setattr(fetch_mod, "get_daily_df", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("empty")))
    assert base.get_current_price("FAIL") == 0.01

    monkeypatch.setattr(base, "health_check", lambda df, resolution=None: bool(df["ok"].iloc[0]))
    monkeypatch.setattr(pd, "read_csv", lambda path: pd.DataFrame({"ok": [path.endswith("GOOD.csv")]}))
    assert base.check_symbol("GOOD", object()) is True
    assert base.check_symbol("BAD", object()) is False
    assert base.pre_trade_health_check(["GOOD", "BAD"], object()) == {"GOOD": True, "BAD": False}

    serializable = base.to_serializable(MappingProxyType({"a": (MappingProxyType({"b": 1}), 2)}))
    assert serializable == {"a": [{"b": 1}, 2]}

    class _Settings:
        use_market_calendar_lib = True

    monkeypatch.setattr(base, "get_settings", lambda: _Settings())
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="pandas_market_calendars"):
        base.enable_market_calendar_lib()
