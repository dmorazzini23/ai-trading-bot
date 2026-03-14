from types import SimpleNamespace
from datetime import datetime, UTC, timedelta

import pytest

import ai_trading.data.fetch as data_fetcher
from tests.helpers.dummy_http import DummyResp

pd = pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def _force_window(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_window_has_trading_session", lambda *a, **k: True)


@pytest.fixture(autouse=True)
def _reset_backup_state():
    data_fetcher._FALLBACK_WINDOWS.clear()
    data_fetcher._FALLBACK_UNTIL.clear()
    data_fetcher._BACKUP_SKIP_UNTIL.clear()
    data_fetcher._GLOBAL_BACKUP_SKIP_UNTIL.clear()
    data_fetcher._BACKUP_PRIMARY_PROBE_AT.clear()
    data_fetcher._SKIPPED_SYMBOLS.clear()
    data_fetcher._BACKUP_SKIP_ACTIVE_SINCE.clear()
    data_fetcher._BACKUP_PRIMARY_PROBE_LAST_SEEN_AT.clear()
    data_fetcher._BACKUP_PRIMARY_PROBE_ALERT_AT.clear()
    data_fetcher._alpaca_disabled_until = None
    yield
    data_fetcher._FALLBACK_WINDOWS.clear()
    data_fetcher._FALLBACK_UNTIL.clear()
    data_fetcher._BACKUP_SKIP_UNTIL.clear()
    data_fetcher._GLOBAL_BACKUP_SKIP_UNTIL.clear()
    data_fetcher._BACKUP_PRIMARY_PROBE_AT.clear()
    data_fetcher._SKIPPED_SYMBOLS.clear()
    data_fetcher._BACKUP_SKIP_ACTIVE_SINCE.clear()
    data_fetcher._BACKUP_PRIMARY_PROBE_LAST_SEEN_AT.clear()
    data_fetcher._BACKUP_PRIMARY_PROBE_ALERT_AT.clear()
    data_fetcher._alpaca_disabled_until = None


def test_alpaca_skipped_after_yahoo_fallback(monkeypatch):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)

    monkeypatch.setenv("ENABLE_HTTP_FALLBACK", "1")
    monkeypatch.setattr(data_fetcher, "_has_alpaca_keys", lambda: True)

    calls = {"alpaca": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["alpaca"] += 1
        return DummyResp({"bars": []})

    monkeypatch.setattr(data_fetcher, "requests", SimpleNamespace(get=fake_get))
    monkeypatch.setattr(data_fetcher._HTTP_SESSION, "get", fake_get, raising=False)

    yahoo_calls = {"n": 0}
    df_fallback = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(start)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1],
        }
    )

    def fake_yahoo(symbol, s, e, interval):  # noqa: ARG001 - test stub
        yahoo_calls["n"] += 1
        return df_fallback

    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", fake_yahoo)
    monkeypatch.setattr(data_fetcher, "_backup_get_bars", fake_yahoo)

    out1 = data_fetcher.get_minute_df("AAPL", start, end)
    assert not out1.empty
    assert yahoo_calls["n"] == 1
    tf_key = ("AAPL", "1Min")
    skip_until = data_fetcher._BACKUP_SKIP_UNTIL.get(tf_key)
    assert isinstance(skip_until, datetime)
    remaining = skip_until - datetime.now(UTC)
    assert remaining >= timedelta(minutes=9, seconds=50)
    assert tf_key in data_fetcher._SKIPPED_SYMBOLS
    first_calls = calls["alpaca"]

    out2 = data_fetcher.get_minute_df("AAPL", start, end)
    assert not out2.empty
    assert yahoo_calls["n"] == 2
    assert calls["alpaca"] == first_calls

    out3 = data_fetcher.get_minute_df("AAPL", start, end)
    assert not out3.empty
    assert yahoo_calls["n"] == 3
    assert calls["alpaca"] == first_calls
    refreshed_until = data_fetcher._BACKUP_SKIP_UNTIL.get(tf_key)
    assert isinstance(refreshed_until, datetime)
    assert refreshed_until > datetime.now(UTC)
    assert tf_key in data_fetcher._SKIPPED_SYMBOLS


def test_backup_skip_window_rechecks_primary_on_probe_due(monkeypatch):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    tf_key = ("AAPL", "1Min")

    monkeypatch.setenv("ENABLE_HTTP_FALLBACK", "1")
    monkeypatch.setenv("BACKUP_PRIMARY_PROBE_SECONDS", "60")
    monkeypatch.setenv("TESTING", "0")
    monkeypatch.setattr(data_fetcher, "_has_alpaca_keys", lambda: True)
    monkeypatch.setattr(data_fetcher, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(data_fetcher.provider_monitor, "active_provider", lambda primary, backup: primary)
    data_fetcher.provider_monitor.disabled_until.clear()
    monkeypatch.setattr(data_fetcher, "_alpaca_disabled_until", None, raising=False)

    calls = {"alpaca": 0, "yahoo": 0}
    bypass_flags: list[bool] = []
    df_primary = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(start)],
            "open": [2.0],
            "high": [2.0],
            "low": [2.0],
            "close": [2.0],
            "volume": [2],
        }
    )

    def fake_fetch_bars(*_args, **_kwargs):
        calls["alpaca"] += 1
        bypass_flags.append(bool(_kwargs.get("bypass_backup_skip")))
        return df_primary.copy()

    def fake_yahoo(*_args, **_kwargs):
        calls["yahoo"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(data_fetcher, "_fetch_bars", fake_fetch_bars)
    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", fake_yahoo)
    monkeypatch.setattr(data_fetcher, "_backup_get_bars", fake_yahoo)

    data_fetcher._set_backup_skip("AAPL", "1Min")
    data_fetcher._BACKUP_PRIMARY_PROBE_AT[tf_key] = datetime.now(UTC) - timedelta(seconds=1)

    out = data_fetcher.get_minute_df("AAPL", start, end)
    assert not out.empty
    assert calls["alpaca"] >= 1
    assert calls["yahoo"] == 0
    assert any(bypass_flags)


def test_backup_probe_due_forces_primary_when_monitor_prefers_backup(monkeypatch):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    tf_key = ("AAPL", "1Min")

    monkeypatch.setenv("ENABLE_HTTP_FALLBACK", "1")
    monkeypatch.setenv("BACKUP_PRIMARY_PROBE_SECONDS", "60")
    monkeypatch.setenv("TESTING", "0")
    monkeypatch.setattr(data_fetcher, "_has_alpaca_keys", lambda: True)
    monkeypatch.setattr(data_fetcher, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(data_fetcher.provider_monitor, "active_provider", lambda primary, backup: backup)
    data_fetcher.provider_monitor.disabled_until.clear()
    monkeypatch.setattr(data_fetcher, "_alpaca_disabled_until", None, raising=False)

    calls = {"alpaca": 0, "yahoo": 0}
    bypass_flags: list[bool] = []
    df_primary = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(start)],
            "open": [2.0],
            "high": [2.0],
            "low": [2.0],
            "close": [2.0],
            "volume": [2],
        }
    )

    def fake_fetch_bars(*_args, **_kwargs):
        calls["alpaca"] += 1
        bypass_flags.append(bool(_kwargs.get("bypass_backup_skip")))
        return df_primary.copy()

    def fake_yahoo(*_args, **_kwargs):
        calls["yahoo"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(data_fetcher, "_fetch_bars", fake_fetch_bars)
    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", fake_yahoo)
    monkeypatch.setattr(data_fetcher, "_backup_get_bars", fake_yahoo)

    data_fetcher._set_backup_skip("AAPL", "1Min")
    data_fetcher._BACKUP_PRIMARY_PROBE_AT[tf_key] = datetime.now(UTC) - timedelta(seconds=1)

    out = data_fetcher.get_minute_df("AAPL", start, end)
    assert not out.empty
    assert calls["alpaca"] >= 1
    assert calls["yahoo"] == 0
    assert any(bypass_flags)


def test_backup_probe_due_forces_primary_when_provider_disabled(monkeypatch):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    tf_key = ("AAPL", "1Min")

    monkeypatch.setenv("ENABLE_HTTP_FALLBACK", "1")
    monkeypatch.setenv("BACKUP_PRIMARY_PROBE_SECONDS", "60")
    monkeypatch.setenv("AI_TRADING_BACKUP_PROBE_ALLOW_WHEN_DISABLED", "1")
    monkeypatch.setenv("TESTING", "0")
    monkeypatch.setattr(data_fetcher, "_has_alpaca_keys", lambda: True)
    monkeypatch.setattr(data_fetcher, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(data_fetcher.provider_monitor, "active_provider", lambda primary, backup: backup)
    data_fetcher.provider_monitor.disabled_until.clear()
    data_fetcher.provider_monitor.disabled_until["alpaca"] = datetime.now(UTC) + timedelta(minutes=5)
    monkeypatch.setattr(data_fetcher, "_alpaca_disabled_until", None, raising=False)

    calls = {"alpaca": 0, "yahoo": 0}
    bypass_flags: list[bool] = []
    df_primary = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(start)],
            "open": [2.0],
            "high": [2.0],
            "low": [2.0],
            "close": [2.0],
            "volume": [2],
        }
    )

    def fake_fetch_bars(*_args, **_kwargs):
        calls["alpaca"] += 1
        bypass_flags.append(bool(_kwargs.get("bypass_backup_skip")))
        return df_primary.copy()

    def fake_yahoo(*_args, **_kwargs):
        calls["yahoo"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(data_fetcher, "_fetch_bars", fake_fetch_bars)
    monkeypatch.setattr(data_fetcher, "_yahoo_get_bars", fake_yahoo)
    monkeypatch.setattr(data_fetcher, "_backup_get_bars", fake_yahoo)

    data_fetcher._set_backup_skip("AAPL", "1Min")
    data_fetcher._BACKUP_PRIMARY_PROBE_AT[tf_key] = datetime.now(UTC) - timedelta(seconds=1)

    out = data_fetcher.get_minute_df("AAPL", start, end)
    assert not out.empty
    assert calls["alpaca"] >= 1
    assert calls["yahoo"] == 0
    assert any(bypass_flags)


def test_backup_skip_sets_global_minute_cooldown(monkeypatch):
    monkeypatch.setenv("AI_TRADING_GLOBAL_BACKUP_SKIP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_GLOBAL_BACKUP_SKIP_SECONDS", "120")
    data_fetcher._GLOBAL_BACKUP_SKIP_UNTIL.clear()

    data_fetcher._set_backup_skip("AAPL", "1Min")
    global_until = data_fetcher._get_global_backup_skip_until("1Min")

    assert isinstance(global_until, datetime)
    assert global_until > datetime.now(UTC) + timedelta(seconds=100)


def test_backup_skip_does_not_set_global_non_minute(monkeypatch):
    monkeypatch.setenv("AI_TRADING_GLOBAL_BACKUP_SKIP_ENABLED", "1")
    data_fetcher._GLOBAL_BACKUP_SKIP_UNTIL.clear()

    data_fetcher._set_backup_skip("AAPL", "1Day")

    assert data_fetcher._get_global_backup_skip_until("1Day") is None


def test_backup_skip_active_window_keeps_probe_schedule(monkeypatch):
    monkeypatch.setenv("BACKUP_PRIMARY_PROBE_SECONDS", "60")
    monkeypatch.setenv("AI_TRADING_GLOBAL_BACKUP_SKIP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_GLOBAL_BACKUP_SKIP_SECONDS", "120")
    data_fetcher._BACKUP_SKIP_UNTIL.clear()
    data_fetcher._BACKUP_PRIMARY_PROBE_AT.clear()
    data_fetcher._GLOBAL_BACKUP_SKIP_UNTIL.clear()

    key = ("AAPL", "1Min")
    data_fetcher._set_backup_skip(*key)
    first_until = data_fetcher._BACKUP_SKIP_UNTIL.get(key)
    first_probe_due = data_fetcher._BACKUP_PRIMARY_PROBE_AT.get(key)
    first_global_until = data_fetcher._get_global_backup_skip_until("1Min")

    assert isinstance(first_until, datetime)
    assert isinstance(first_probe_due, datetime)
    assert isinstance(first_global_until, datetime)

    data_fetcher._set_backup_skip(*key)
    second_until = data_fetcher._BACKUP_SKIP_UNTIL.get(key)
    second_probe_due = data_fetcher._BACKUP_PRIMARY_PROBE_AT.get(key)
    second_global_until = data_fetcher._get_global_backup_skip_until("1Min")

    # Repeated backup confirmations should not push recovery probes farther out.
    assert second_until == first_until
    assert second_probe_due == first_probe_due
    assert second_global_until == first_global_until


def test_backup_skip_new_symbol_does_not_extend_active_global_window(monkeypatch):
    monkeypatch.setenv("AI_TRADING_GLOBAL_BACKUP_SKIP_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_GLOBAL_BACKUP_SKIP_SECONDS", "120")
    data_fetcher._BACKUP_SKIP_UNTIL.clear()
    data_fetcher._BACKUP_PRIMARY_PROBE_AT.clear()
    data_fetcher._GLOBAL_BACKUP_SKIP_UNTIL.clear()

    data_fetcher._set_backup_skip("AAPL", "1Min")
    first_global_until = data_fetcher._get_global_backup_skip_until("1Min")
    assert isinstance(first_global_until, datetime)

    data_fetcher._set_backup_skip("MSFT", "1Min")
    second_global_until = data_fetcher._get_global_backup_skip_until("1Min")
    assert isinstance(second_global_until, datetime)

    assert second_global_until == first_global_until


def test_backup_probe_watchdog_alerts_when_probe_missing(monkeypatch, caplog):
    monkeypatch.setenv("AI_TRADING_BACKUP_PROBE_GUARD_ENABLED", "1")
    monkeypatch.setenv("AI_TRADING_BACKUP_PROBE_MISSING_SECONDS", "60")
    monkeypatch.setenv("AI_TRADING_BACKUP_PROBE_ALERT_COOLDOWN_SECONDS", "300")

    data_fetcher._BACKUP_SKIP_ACTIVE_SINCE.clear()
    data_fetcher._BACKUP_PRIMARY_PROBE_LAST_SEEN_AT.clear()
    data_fetcher._BACKUP_PRIMARY_PROBE_ALERT_AT.clear()

    now_dt = datetime.now(UTC)
    data_fetcher._BACKUP_SKIP_ACTIVE_SINCE["1Min"] = now_dt - timedelta(seconds=180)
    data_fetcher._BACKUP_PRIMARY_PROBE_LAST_SEEN_AT["1Min"] = now_dt - timedelta(seconds=180)

    metrics_events: list[tuple[str, float, dict[str, str] | None]] = []
    alerts: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        data_fetcher,
        "_incr",
        lambda name, value=1.0, tags=None: metrics_events.append((name, value, tags)),
    )
    monkeypatch.setattr(
        data_fetcher.provider_monitor.alert_manager,
        "create_alert",
        lambda *args, **kwargs: alerts.append((args, kwargs)),
    )

    caplog.set_level("ERROR")
    data_fetcher._note_backup_skip_activity(
        "1Min",
        symbol="AAPL",
        skip_until=now_dt + timedelta(minutes=5),
        global_skip=True,
    )

    assert any(record.message == "ALERT_PRIMARY_RECOVERY_PROBE_MISSING" for record in caplog.records)
    assert alerts
    assert any(name == "data.fetch.primary_probe_missing" for name, _value, _tags in metrics_events)

    caplog.clear()
    data_fetcher._note_backup_skip_activity(
        "1Min",
        symbol="AAPL",
        skip_until=now_dt + timedelta(minutes=5),
        global_skip=True,
    )
    assert not any(record.message == "ALERT_PRIMARY_RECOVERY_PROBE_MISSING" for record in caplog.records)
