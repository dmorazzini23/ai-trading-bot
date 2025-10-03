import logging
from datetime import UTC, datetime, timedelta
import types

import pytest

import ai_trading.data.fetch as data_fetcher

pd = pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def _restore_globals(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_SIP_UNAVAILABLE_LOGGED", set(), raising=False)
    monkeypatch.setattr(data_fetcher, "_FALLBACK_WINDOWS", set(), raising=False)
    monkeypatch.setattr(data_fetcher, "_FALLBACK_METADATA", {}, raising=False)
    monkeypatch.setattr(data_fetcher, "_FALLBACK_UNTIL", {}, raising=False)
    monkeypatch.setattr(data_fetcher, "_SIP_DISALLOWED_WARNED", False, raising=False)


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
    monkeypatch.setattr(data_fetcher, "_SIP_UNAUTHORIZED", True, raising=False)
    monkeypatch.setattr(data_fetcher, "_ALLOW_SIP", True, raising=False)
    monkeypatch.setattr(data_fetcher, "_sip_configured", lambda: True)
    monkeypatch.setattr(data_fetcher, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(
        data_fetcher,
        "get_settings",
        lambda: types.SimpleNamespace(backup_data_provider="yahoo"),
    )

    with caplog.at_level(logging.INFO):
        result = data_fetcher.get_minute_df("AAPL", start, end, feed="sip")

    assert isinstance(result, pd.DataFrame)
    assert result.attrs.get("data_provider") == "yahoo"
    assert result.attrs.get("data_feed") == "yahoo"
    assert any(
        record.message == "USING_BACKUP_PROVIDER" and getattr(record, "provider", None) == "yahoo"
        for record in caplog.records
    )
    assert any(record.message == "UNAUTHORIZED_SIP" for record in caplog.records)


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
