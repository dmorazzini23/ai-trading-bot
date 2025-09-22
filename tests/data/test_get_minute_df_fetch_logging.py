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
