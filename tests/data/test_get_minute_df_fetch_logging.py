import logging
from datetime import UTC, datetime, timedelta
import types

import pytest

import ai_trading.data.fetch as data_fetcher
from ai_trading.telemetry import runtime_state

pd = pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def _restore_globals(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_SIP_UNAVAILABLE_LOGGED", set(), raising=False)
    monkeypatch.setattr(data_fetcher, "_FALLBACK_WINDOWS", set(), raising=False)
    monkeypatch.setattr(data_fetcher, "_FALLBACK_METADATA", {}, raising=False)
    monkeypatch.setattr(data_fetcher, "_FALLBACK_UNTIL", {}, raising=False)
    monkeypatch.setattr(data_fetcher, "_SIP_DISALLOWED_WARNED", False, raising=False)


@pytest.fixture(autouse=True)
def _reset_runtime_state():
    runtime_state.update_data_provider_state(primary="alpaca", active="alpaca", backup=None, using_backup=False, reason=None, cooldown_sec=0.0)


def _dt_range():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=5)
    return start, end


class _DummyProviderMonitor:
    def record_switchover(self, *args, **kwargs):
        return None

    def record_failure(self, *args, **kwargs):
        return None

    def disable(self, *args, **kwargs):
        return None

    def is_disabled(self, *args, **kwargs):
        return False

    def register_disable_callback(self, *args, **kwargs):
        return None

    def update_data_health(self, *args, **kwargs):
        return None


def test_sip_unauthorized_branch_annotates_backup(monkeypatch, caplog):
    start, end = _dt_range()
    timestamps = pd.date_range(start=start, periods=5, freq="1min", tz=UTC)
    fallback_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0] * 5,
            "high": [1.5] * 5,
            "low": [0.5] * 5,
            "close": [1.2] * 5,
            "volume": [100] * 5,
        }
    )

    def _fake_yahoo(*args, **kwargs):
        return fallback_df.copy()

    monkeypatch.setattr(data_fetcher, "provider_monitor", _DummyProviderMonitor())
    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", _fake_yahoo)
    monkeypatch.setattr(data_fetcher, "DATA_HTTP_FALLBACK_INTRADAY", True, raising=False)
    monkeypatch.setattr(data_fetcher, "DATA_HTTP_FALLBACK_OUT_OF_SESSION", True, raising=False)
    monkeypatch.setattr(data_fetcher, "_SIP_UNAUTHORIZED", True, raising=False)
    monkeypatch.setattr(data_fetcher, "_ALLOW_SIP", True, raising=False)
    monkeypatch.setattr(data_fetcher, "_sip_configured", lambda: True)
    monkeypatch.setattr(data_fetcher, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(
        data_fetcher,
        "get_settings",
        lambda: types.SimpleNamespace(backup_data_provider="yahoo"),
    )
    monkeypatch.setattr(data_fetcher, "_has_alpaca_keys", lambda: True, raising=False)
    monkeypatch.setenv("DATA_FEED_INTRADAY", "sip")
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    with caplog.at_level(logging.INFO):
        result = data_fetcher.get_minute_df("AAPL", start, end, feed="sip")

    assert isinstance(result, pd.DataFrame)
    assert result.attrs.get("data_provider") == "yahoo"
    assert result.attrs.get("data_feed") == "yahoo"
    assert any(
        record.message.startswith("USING_BACKUP_PROVIDER")
        and getattr(record, "provider", None) == "yahoo"
        and "provider=yahoo" in record.message
        for record in caplog.records
    )
    assert any(record.message == "UNAUTHORIZED_SIP" for record in caplog.records)


def test_backup_usage_marks_primary_unhealthy(monkeypatch):
    start, end = _dt_range()
    timestamps = pd.date_range(start=start, periods=5, freq="1min", tz=UTC)
    fallback_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0] * 5,
            "high": [1.5] * 5,
            "low": [0.5] * 5,
            "close": [1.2] * 5,
            "volume": [100] * 5,
        }
    )

    class _Monitor(_DummyProviderMonitor):
        def __init__(self):
            self.health_updates = []

        def update_data_health(self, primary, backup, *, healthy, reason, severity=None, **_kwargs):
            self.health_updates.append((primary, backup, healthy, reason, severity))
            return backup if not healthy else primary

    monitor = _Monitor()

    monkeypatch.setattr(data_fetcher, "provider_monitor", monitor)
    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", lambda *_a, **_k: fallback_df.copy())
    monkeypatch.setattr(data_fetcher, "DATA_HTTP_FALLBACK_INTRADAY", True, raising=False)
    monkeypatch.setattr(data_fetcher, "DATA_HTTP_FALLBACK_OUT_OF_SESSION", True, raising=False)
    monkeypatch.setattr(data_fetcher, "_SIP_UNAUTHORIZED", True, raising=False)
    monkeypatch.setattr(data_fetcher, "_ALLOW_SIP", True, raising=False)
    monkeypatch.setattr(data_fetcher, "_sip_configured", lambda: True)
    monkeypatch.setattr(data_fetcher, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(
        data_fetcher,
        "get_settings",
        lambda: types.SimpleNamespace(backup_data_provider="yahoo"),
    )
    monkeypatch.setattr(data_fetcher, "_has_alpaca_keys", lambda: True, raising=False)
    monkeypatch.setenv("DATA_FEED_INTRADAY", "sip")
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "1")
    monkeypatch.setenv("ALPACA_HAS_SIP", "1")
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    result = data_fetcher.get_minute_df("AAPL", start, end, feed="sip")

    assert isinstance(result, pd.DataFrame)
    assert monitor.health_updates
    _, _, healthy, reason, severity = monitor.health_updates[-1]
    assert healthy is False
    assert severity == "hard_fail"
    assert str(reason).strip() != ""


def test_env_without_sip_does_not_schedule_sip(monkeypatch):
    monkeypatch.setenv("ALPACA_ALLOW_SIP", "0")
    monkeypatch.setenv("ALPACA_HAS_SIP", "0")
    monkeypatch.setattr(data_fetcher, "_ALLOW_SIP", None, raising=False)
    monkeypatch.setattr(data_fetcher, "_cycle_feed_override", {}, raising=False)
    monkeypatch.setattr(data_fetcher, "_override_set_ts", {}, raising=False)
    monkeypatch.setattr(data_fetcher, "_CYCLE_FALLBACK_FEED", {}, raising=False)
    monkeypatch.setattr(data_fetcher, "_OVERRIDE_MAP", {}, raising=False)
    monkeypatch.setattr(data_fetcher, "_FEED_SWITCH_CACHE", {}, raising=False)

    assert data_fetcher._sip_allowed() is False
    assert "sip" not in data_fetcher._ordered_fallbacks("iex")

    data_fetcher._record_override("AAPL", "sip", "1Min")
    assert "AAPL" not in data_fetcher._cycle_feed_override

    cycle_id = data_fetcher._get_cycle_id()
    data_fetcher._CYCLE_FALLBACK_FEED[(cycle_id, "AAPL", "1Min")] = "sip"
    assert data_fetcher._fallback_cache_for_cycle(cycle_id, "AAPL", "1Min") is None

    monkeypatch.setattr(data_fetcher, "_time_now", lambda default=0.0: 200.0, raising=False)
    data_fetcher._cycle_feed_override["AAPL"] = "sip"
    data_fetcher._override_set_ts["AAPL"] = 100.0
    assert data_fetcher._get_cached_or_primary("AAPL", "iex") == "iex"


def test_clear_minute_fallback_state_clears_all_metadata(monkeypatch):
    start, end = _dt_range()
    key = data_fetcher._fallback_key("AAPL", "1Min", start, end)
    tf_key = ("AAPL", "1Min")
    data_fetcher._FALLBACK_WINDOWS.add(key)
    data_fetcher._FALLBACK_METADATA[key] = {"fallback_provider": "yahoo"}
    data_fetcher._FALLBACK_UNTIL[tf_key] = 123456

    calls: list[tuple[str, str, bool, str, str | None]] = []

    class _Monitor:
        def update_data_health(self, primary, backup, *, healthy, reason, severity=None):
            calls.append((primary, backup, healthy, reason, severity))
            return primary

    monkeypatch.setattr(data_fetcher, "provider_monitor", _Monitor())

    cleared = data_fetcher._clear_minute_fallback_state(
        "AAPL",
        "1Min",
        start,
        end,
        primary_label="alpaca_iex",
        backup_label="yahoo",
    )

    assert cleared is True
    assert key not in data_fetcher._FALLBACK_WINDOWS
    assert key not in data_fetcher._FALLBACK_METADATA
    assert tf_key not in data_fetcher._FALLBACK_UNTIL
    assert calls == [("alpaca_iex", "yahoo", True, "primary_recovered", "good")]


def test_fallback_logging_suppressed_within_window(monkeypatch, caplog):
    start, end = _dt_range()
    timestamps = pd.date_range(start=start, periods=3, freq="1min", tz=UTC)
    fallback_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0, 1.1, 1.2],
            "high": [1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [100, 110, 120],
        }
    )

    class _Monitor(_DummyProviderMonitor):
        decision_window_seconds = 300

    monkeypatch.setattr(data_fetcher, "provider_monitor", _Monitor())
    monkeypatch.setattr(data_fetcher, "_FALLBACK_WINDOWS", set(), raising=False)
    monkeypatch.setattr(data_fetcher, "_FALLBACK_METADATA", {}, raising=False)
    monkeypatch.setattr(data_fetcher, "_FALLBACK_UNTIL", {}, raising=False)
    monkeypatch.setattr(data_fetcher, "_FALLBACK_SUPPRESS_UNTIL", {}, raising=False)
    monkeypatch.setattr(data_fetcher, "DATA_HTTP_FALLBACK_INTRADAY", True, raising=False)
    monkeypatch.setattr(data_fetcher, "DATA_HTTP_FALLBACK_OUT_OF_SESSION", True, raising=False)

    with caplog.at_level(logging.INFO):
        data_fetcher._mark_fallback(
            "AAPL",
            "1Min",
            start,
            end,
            from_provider="alpaca_iex",
            fallback_df=fallback_df,
            resolved_provider="yahoo",
        )
        data_fetcher._mark_fallback(
            "AAPL",
            "1Min",
            start + timedelta(minutes=1),
            end + timedelta(minutes=1),
            from_provider="alpaca_iex",
            fallback_df=fallback_df,
            resolved_provider="yahoo",
        )

    using_backup_logs = [
        record for record in caplog.records if record.message.startswith("USING_BACKUP_PROVIDER")
    ]
    assert len(using_backup_logs) == 1
    assert "provider=yahoo" in using_backup_logs[0].message
    assert "timeframe=1Min" in using_backup_logs[0].message
    assert "reason=" in using_backup_logs[0].message
    provider_state = runtime_state.observe_data_provider_state()
    assert provider_state["using_backup"] is True
    assert provider_state["active"] == "yahoo"


def test_no_session_intraday_skips_backup_when_toggle_disabled(monkeypatch, caplog):
    start, end = _dt_range()
    session_stub = types.SimpleNamespace(get=lambda *args, **kwargs: None)

    monkeypatch.setattr(data_fetcher, "_HTTP_SESSION", session_stub, raising=False)
    monkeypatch.setattr(data_fetcher, "_window_has_trading_session", lambda *a, **k: False)
    monkeypatch.setattr(data_fetcher, "DATA_HTTP_FALLBACK_INTRADAY", False, raising=False)
    monkeypatch.setattr(data_fetcher, "DATA_HTTP_FALLBACK_OUT_OF_SESSION", False, raising=False)
    monkeypatch.setattr(data_fetcher, "_ENABLE_HTTP_FALLBACK", True, raising=False)
    monkeypatch.setattr(data_fetcher, "_FALLBACK_WINDOWS", set(), raising=False)
    monkeypatch.setattr(data_fetcher, "_FALLBACK_METADATA", {}, raising=False)
    monkeypatch.setattr(data_fetcher, "_has_alpaca_keys", lambda: True, raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")

    with caplog.at_level(logging.INFO):
        result = data_fetcher.get_minute_df("AAPL", start, end, feed="iex")

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert set(result.columns) >= {"timestamp", "open", "high", "low", "close", "volume"}
    assert not any("USING_BACKUP_PROVIDER" in record.message for record in caplog.records)
    assert "DATA_WINDOW_NO_SESSION" in caplog.text


def test_no_session_intraday_fallback_reenabled_with_toggle(monkeypatch, caplog):
    start, end = _dt_range()
    timestamps = pd.date_range(start=start, periods=3, freq="1min", tz=UTC)
    yahoo_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.05, 10.15, 10.25],
            "volume": [1000, 1100, 1200],
        }
    )

    session_stub = types.SimpleNamespace(get=lambda *args, **kwargs: types.SimpleNamespace(status_code=200, json=lambda: {}))

    monkeypatch.setattr(data_fetcher, "_HTTP_SESSION", session_stub, raising=False)
    monkeypatch.setattr(data_fetcher, "_session_get", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(data_fetcher, "_window_has_trading_session", lambda *a, **k: False)
    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", lambda *a, **k: yahoo_df.copy(), raising=False)
    monkeypatch.setattr(data_fetcher, "DATA_HTTP_FALLBACK_INTRADAY", True, raising=False)
    monkeypatch.setattr(data_fetcher, "DATA_HTTP_FALLBACK_OUT_OF_SESSION", True, raising=False)
    monkeypatch.setattr(data_fetcher, "_ENABLE_HTTP_FALLBACK", True, raising=False)
    monkeypatch.setattr(data_fetcher, "provider_monitor", _DummyProviderMonitor())
    monkeypatch.setattr(data_fetcher, "_FALLBACK_WINDOWS", set(), raising=False)
    monkeypatch.setattr(data_fetcher, "_FALLBACK_METADATA", {}, raising=False)
    monkeypatch.setattr(data_fetcher, "_has_alpaca_keys", lambda: True, raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")

    with caplog.at_level(logging.INFO):
        result = data_fetcher.get_minute_df("AAPL", start, end, feed="iex")

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert result.attrs.get("data_provider") == "yahoo"
    backup_logs = [record.message for record in caplog.records if record.message.startswith("USING_BACKUP_PROVIDER")]
    assert backup_logs, "expected backup provider log message"
    assert any("reason=" in message for message in backup_logs)
