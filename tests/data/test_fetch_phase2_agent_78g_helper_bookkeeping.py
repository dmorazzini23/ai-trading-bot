from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import fetch


BASE_START = datetime(2026, 4, 24, 14, 30, tzinfo=UTC)
BASE_END = BASE_START + timedelta(minutes=5)


class _AlertManager:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def create_alert(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class _Monitor:
    threshold = 2
    cooldown = 0
    min_recovery_seconds = 0

    def __init__(self) -> None:
        self.fail_counts: dict[str, int] = {}
        self.alert_manager = _AlertManager()
        self.switchovers: list[tuple[str, str]] = []

    def is_disabled(self, provider: str) -> bool:
        return provider == "alpaca_iex"

    def safe_mode_cycle_marker(self) -> tuple[int, str]:
        return (7, "provider_safe_mode")

    def record_switchover(self, from_provider: str, to_provider: str) -> None:
        self.switchovers.append((from_provider, to_provider))


@pytest.fixture(autouse=True)
def _reset_fetch_bookkeeping(monkeypatch: pytest.MonkeyPatch) -> None:
    stores = (
        fetch._ALPACA_FAILURE_EVENTS,
        fetch._ALPACA_CONSECUTIVE_FAILURES,
        fetch._BACKUP_PRIMARY_PROBE_ALERT_AT,
        fetch._BACKUP_PRIMARY_PROBE_AT,
        fetch._BACKUP_PRIMARY_PROBE_LAST_SEEN_AT,
        fetch._BACKUP_SKIP_ACTIVE_SINCE,
        fetch._BACKUP_SKIP_UNTIL,
        fetch._BACKUP_USAGE_LOGGED,
        fetch._FALLBACK_METADATA,
        fetch._FALLBACK_SUPPRESS_UNTIL,
        fetch._FALLBACK_UNTIL,
        fetch._FALLBACK_WINDOWS,
        fetch._FEED_FAILOVER_ATTEMPTS,
        fetch._FEED_SWITCH_CACHE,
        fetch._FEED_SWITCH_HISTORY,
        fetch._GLOBAL_BACKUP_SKIP_UNTIL,
        fetch._OVERRIDE_MAP,
        fetch._YF_WARNING_CACHE,
        fetch._cycle_feed_override,
        fetch._daily_memo,
        fetch._override_set_ts,
    )
    for store in stores:
        store.clear()
    fetch._FEED_SWITCH_LOGGED.clear()
    fetch._SAFE_MODE_LOGGED.clear()
    fetch._SKIPPED_SYMBOLS.clear()
    fetch._SAFE_MODE_CYCLE_STATE.update({"cycle_id": None, "reason": None, "version": 0})
    fetch._set_fetch_state({})
    monkeypatch.setattr(fetch, "_detect_pytest_env", lambda: True)
    monkeypatch.setattr(fetch, "is_safe_mode_active", lambda: False)
    yield
    fetch._set_fetch_state({})


def _frame(
    close: float | None = 101.0,
    *,
    timestamp: datetime = datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [timestamp],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [close],
            "volume": [1000],
        },
    )


def test_env_bootstrap_and_incomplete_row_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "AI_TRADING_BOOTSTRAP_PRIMARY_ONLY": "false",
        "DATA_PROVIDER": " alpaca ",
        "SOME_INT": "bad",
        "SOME_FLOAT": "",
    }
    monkeypatch.setattr(fetch, "get_env", lambda key, default=None, **_kwargs: values.get(key, default))
    monkeypatch.setattr(fetch, "_drop_last_bar_enabled", lambda: True)

    assert fetch._should_bootstrap_primary_first() is False
    assert fetch._configured_primary_provider() == "alpaca"
    assert fetch._env_int("SOME_INT", 4) == 4
    assert fetch._env_float("SOME_FLOAT", 2.5) == 2.5

    frame = pd.DataFrame(
        {
            "timestamp": [BASE_START, BASE_START + timedelta(minutes=1), BASE_START + timedelta(minutes=2)],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, None, None],
            "volume": [1000, 1100, None],
        },
    )

    result = fetch._apply_incomplete_row_policy(frame, "AAPL", "1Min")

    assert list(result["timestamp"]) == [BASE_START]
    assert list(result["close"]) == [100.5]
    assert fetch._apply_incomplete_row_policy(["not", "a", "frame"], None, None) == ["not", "a", "frame"]


def test_safe_empty_wrappers_backup_get_bars_and_daily_fetch_backup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(fetch, "_empty_should_emit", lambda *_args: (_ for _ in ()).throw(ValueError("boom")))
    monkeypatch.setattr(fetch, "_empty_record", lambda *_args: None)
    monkeypatch.setattr(fetch, "_empty_classify", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(fetch, "_backup_get_bars", lambda *_args: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(fetch, "fetch_yf_batched", lambda *_args, **_kwargs: {"AAPL": _frame(), "MSFT": pd.DataFrame()})
    monkeypatch.setattr(fetch, "_data_fallback_allowed", lambda: False)
    monkeypatch.setattr(fetch, "_current_settings", lambda: SimpleNamespace(backup_data_provider="yahoo"))
    monkeypatch.setattr(
        fetch,
        "activate_data_kill_switch",
        lambda reason, **kwargs: calls.append((reason, kwargs)),
    )
    monkeypatch.setattr(fetch, "log_data_quality_event", lambda *_args, **_kwargs: pytest.fail("blocked fallback should not log usage"))

    key = ("alpaca", "AAPL", "1Min", "empty", "2026-04-24")
    assert fetch._safe_empty_should_emit(key, BASE_START) is False
    assert fetch._safe_empty_record(key, BASE_START) == 0
    assert fetch._safe_empty_classify(symbol="AAPL") == logging.INFO
    assert fetch._safe_backup_get_bars("AAPL", BASE_START, BASE_END, "1m").empty

    assert fetch.fetch_daily_backup(["AAPL", "MSFT"], start=BASE_START, end=BASE_END) == {}
    assert calls == [("backup_provider_blocked", {"provider": "yahoo", "metadata": {"symbols": ["AAPL"]}})]


def test_daily_fetch_backup_logs_when_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(fetch, "fetch_yf_batched", lambda *_args, **_kwargs: {"SPY": _frame()})
    monkeypatch.setattr(fetch, "_data_fallback_allowed", lambda: True)
    monkeypatch.setattr(fetch, "_current_settings", lambda: SimpleNamespace(backup_data_provider="finnhub"))
    monkeypatch.setattr(fetch, "activate_data_kill_switch", lambda *_args, **_kwargs: pytest.fail("allowed fallback should not trip kill switch"))
    monkeypatch.setattr(fetch, "log_data_quality_event", lambda event, **kwargs: events.append({"event": event, **kwargs}))

    result = fetch.fetch_daily_backup(["SPY"], period="5d")

    assert list(result) == ["SPY"]
    assert list(result["SPY"].columns[:6]) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert events[0]["event"] == "backup_provider_used"
    assert events[0]["provider"] == "finnhub"
    assert events[0]["context"] == {"symbols": ["SPY"]}


def test_concurrency_and_daily_memo_generator_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "_time_now", lambda default=0.0: 100.0 if default is not None else 100.0)

    async def good(value: int) -> int:
        return value

    async def bad() -> int:
        raise ValueError("failed")

    results, succeeded, failed = asyncio.run(
        fetch.run_with_concurrency(2, [good(1), bad(), good(3)]),
    )
    assert results[0] == 1
    assert isinstance(results[1], ValueError)
    assert results[2] == 3
    assert (succeeded, failed) == (2, 1)

    def generator_factory():
        yield "first"
        yield "second"

    assert fetch.daily_fetch_memo(("AAPL", "1Day"), generator_factory) == "first"
    assert fetch.daily_fetch_memo(("AAPL", "1Day"), lambda: pytest.fail("memo should be fresh")) == "first"

    def stopped_factory():
        if False:
            yield "never"
        return

    assert fetch.daily_fetch_memo(("MSFT", "1Day"), stopped_factory) is None


def test_global_backup_skip_probe_and_alert_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monitor = _Monitor()
    metrics: list[tuple[str, float, dict[str, str] | None]] = []
    env = {
        "AI_TRADING_GLOBAL_BACKUP_SKIP_ENABLED": "1",
        "AI_TRADING_GLOBAL_BACKUP_SKIP_SECONDS": "30",
        "BACKUP_PRIMARY_PROBE_SECONDS": "0",
        "AI_TRADING_BACKUP_PROBE_GUARD_ENABLED": True,
        "AI_TRADING_BACKUP_PROBE_MISSING_SECONDS": 1,
        "AI_TRADING_BACKUP_PROBE_ALERT_COOLDOWN_SECONDS": 60,
    }
    monkeypatch.setattr(fetch, "get_env", lambda key, default=None, **_kwargs: env.get(key, default))
    monkeypatch.setattr(fetch, "provider_monitor", monitor)
    monkeypatch.setattr(fetch, "_incr", lambda name, value=1.0, tags=None: metrics.append((name, value, tags)))

    fetch._set_global_backup_skip("1Min")
    assert fetch._get_global_backup_skip_until("1Min") is not None
    assert fetch._backup_primary_probe_due("AAPL", "1Min") is False

    fetch._BACKUP_SKIP_ACTIVE_SINCE["1Min"] = datetime.now(tz=UTC) - timedelta(seconds=5)
    fetch._BACKUP_PRIMARY_PROBE_LAST_SEEN_AT["1Min"] = datetime.now(tz=UTC) - timedelta(seconds=5)
    caplog.set_level(logging.ERROR)
    fetch._note_backup_skip_activity("1Min", symbol="AAPL", global_skip=True)

    assert any(record.message == "ALERT_PRIMARY_RECOVERY_PROBE_MISSING" for record in caplog.records)
    assert monitor.alert_manager.calls
    assert metrics == [("data.fetch.primary_probe_missing", 1.0, {"provider": "alpaca", "timeframe": "1Min"})]

    fetch._record_primary_probe_seen("1Min", symbol="AAPL", route="primary")
    assert "1Min" in fetch._BACKUP_PRIMARY_PROBE_LAST_SEEN_AT
    fetch._clear_global_backup_skip("1Min")
    assert fetch._get_global_backup_skip_until("1Min") is None
    assert "1Min" not in fetch._BACKUP_SKIP_ACTIVE_SINCE


def test_backup_skip_expiry_cleanup_and_primary_probe_due(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"BACKUP_PRIMARY_PROBE_SECONDS": "1", "AI_TRADING_GLOBAL_BACKUP_SKIP_ENABLED": "0"}
    monkeypatch.setattr(fetch, "get_env", lambda key, default=None, **_kwargs: env.get(key, default))
    past = datetime.now(tz=UTC) - timedelta(seconds=1)
    future = datetime.now(tz=UTC) + timedelta(seconds=30)
    fetch._BACKUP_SKIP_UNTIL[("AAPL", "1Min")] = past
    fetch._BACKUP_SKIP_UNTIL[("MSFT", "1Min")] = future.replace(tzinfo=None)

    assert fetch._timeframe_has_active_backup_skip("1Min") is True
    assert ("AAPL", "1Min") not in fetch._BACKUP_SKIP_UNTIL

    assert fetch._backup_primary_probe_due("AAPL", "1Min") is False
    fetch._BACKUP_PRIMARY_PROBE_AT[("AAPL", "1Min")] = datetime.now(tz=UTC) - timedelta(seconds=1)
    assert fetch._backup_primary_probe_due("AAPL", "1Min") is True

    fetch._clear_backup_skip("MSFT", "1Min")
    assert ("MSFT", "1Min") not in fetch._BACKUP_SKIP_UNTIL
    assert ("MSFT", "1Min") not in fetch._SKIPPED_SYMBOLS


def test_safe_mode_cycle_and_preferred_feed_bookkeeping(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = _Monitor()
    monkeypatch.setattr(fetch, "provider_monitor", monitor)
    monkeypatch.setattr(fetch, "_get_cycle_id", lambda: "cycle-78g")
    monkeypatch.setattr(fetch, "safe_mode_reason", lambda: "manual_safe_mode")

    fetch.notify_primary_provider_safe_mode(version=None)
    assert fetch._cycle_safe_mode_active("cycle-78g") == (True, "manual_safe_mode")
    assert fetch.is_primary_provider_enabled() is False
    assert fetch.is_primary_provider_enabled() is False

    fetch.clear_safe_mode_cycle()
    assert fetch._cycle_safe_mode_active("cycle-78g") == (False, None)

    monkeypatch.setattr(fetch, "alpaca_feed_failover", lambda: ())
    monkeypatch.setattr(fetch, "provider_priority", lambda: ("alpaca_sip", "alpaca_iex", "bad-feed", "yahoo"))
    monkeypatch.setattr(fetch, "_is_sip_unauthorized", lambda: True)
    monkeypatch.setattr(fetch, "_sip_allowed", lambda: True)
    assert fetch._iter_preferred_feeds("AAPL", "1Min", "iex") == ("yahoo",)
    assert fetch._FEED_FAILOVER_ATTEMPTS[("AAPL", "1Min")] == {"sip", "yahoo"}


def test_prepare_sip_fallback_records_metrics_and_optional_safe_mode_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pushed: list[tuple[str, dict[str, Any]]] = []
    metrics: list[tuple[str, float, dict[str, str]]] = []
    unauthorized: list[dict[str, Any]] = []
    monkeypatch.setattr(fetch, "_intraday_feed_prefers_sip", lambda: True)
    monkeypatch.setattr(fetch, "_sip_allowed", lambda: True)
    monkeypatch.setattr(fetch, "_is_sip_unauthorized", lambda: False)
    monkeypatch.setattr(fetch, "_sip_explicitly_disabled", lambda: False)
    monkeypatch.setattr(fetch, "_now_ts", lambda: 100.0)
    monkeypatch.setattr(fetch, "_get_cycle_id", lambda: "cycle-78g")
    monkeypatch.setattr(fetch, "_incr", lambda name, value=1.0, tags=None: metrics.append((name, value, tags or {})))
    monkeypatch.setattr(fetch, "inc_provider_fallback", lambda *_args: None)
    monkeypatch.setattr(fetch, "record_unauthorized_sip_event", lambda payload: unauthorized.append(payload))

    payload = fetch._prepare_sip_fallback(
        "AAPL",
        "1Min",
        "iex",
        occurrences=3,
        correlation_id="cid-1",
        push_to_caplog=lambda message, **kwargs: pushed.append((message, kwargs)),
        tags_factory=lambda: {"symbol": "AAPL", "feed": "sip"},
    )

    assert payload["occurrences"] == 3
    assert pushed[0][0] == "ALPACA_IEX_FALLBACK_SIP"
    assert metrics == [("data.fetch.feed_switch", 1.0, {"symbol": "AAPL", "feed": "sip"})]
    assert unauthorized and unauthorized[0]["correlation_id"] == "cid-1"
    assert fetch._get_cached_or_primary("AAPL", "iex") == "sip"


def test_metadata_and_timestamp_helpers_preserve_attrs_and_payload_keys() -> None:
    raw = pd.DataFrame(
        {
            "Timestamp": [BASE_START],
            "Open": [100.0],
            "High": [102.0],
            "Low": [99.0],
            "Close": [101.0],
            "Volume": [1000],
        },
    )
    raw.attrs["request_id"] = "req-1"

    annotated = fetch._annotate_df_source(raw, provider="yahoo", feed="1m")
    assert annotated is raw
    assert raw.attrs["data_provider"] == "yahoo"
    assert raw.attrs["fallback_feed"] == "1m"

    normalized = fetch._normalize_with_attrs(raw)
    assert normalized.attrs["request_id"] == "req-1"
    assert normalized.attrs["data_provider"] == "yahoo"

    indexed = normalized.drop(columns=["timestamp"])
    restored = fetch._restore_timestamp_column(indexed)
    assert "timestamp" in restored.columns

    fetch._attach_payload_metadata(
        restored,
        payload={"bars": [{"t": "2026-04-24T14:30:00Z", "c": 101}, {"ignored": True}]},
        provider="alpaca",
        feed="iex",
        timeframe="1Min",
        symbol="AAPL",
    )

    assert restored.attrs["raw_payload_keys"] == ("c", "ignored", "t")
    assert restored.attrs["raw_payload_symbol"] == "AAPL"
    assert getattr(restored, "_raw_payload_symbol") == "AAPL"
    assert fetch._extract_payload_keys({"bars": {"AAPL": {"c": 101}, "MSFT": {"o": 100}}}) == ("c", "o")


def test_exception_and_reason_helpers_cover_fallback_categories() -> None:
    missing = fetch.MissingOHLCVColumnsError(
        "missing",
        metadata={
            "raw_payload_columns": ["close", "volume"],
            "raw_payload_keys": ("bars", "message"),
            "raw_payload_feed": "iex",
            "raw_payload_timeframe": "1Min",
            "raw_payload_provider": "alpaca",
            "raw_payload_symbol": "AAPL",
        },
    )
    assert missing.raw_payload_columns == ("close", "volume")
    assert missing.raw_payload_keys == ("bars", "message")
    assert missing.raw_payload_provider == "alpaca"

    reason, details = fetch._classify_fallback_reason(
        "quote timestamp missing during fallback_ttl primary_probe",
        {"provider": "alpaca_iex", "retry_after": "2.5", "gap_ratio": "0"},
    )
    assert reason == "quote_timestamp_missing"
    assert details["retry_after"] == 2.5
    assert details["gap_ratio_pct"] == 0.0

    assert fetch._coerce_reason_text(None) is None
    assert fetch._coerce_reason_text("  rate_limited  ") == "rate_limited"
    assert fetch._build_backup_usage_extra("yahoo", "spy", "1Min", None, {})["reason"] == "upstream_unavailable"
