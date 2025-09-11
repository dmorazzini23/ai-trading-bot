import logging
from datetime import UTC, datetime, timedelta

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.data import fetch
from ai_trading.data.provider_monitor import provider_monitor


def _dt_range():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return start, end


def _setup_common(monkeypatch):
    """Prepare environment so fetch.get_minute_df uses stubs and fallback."""
    monkeypatch.setattr(fetch, "_ensure_pandas", lambda: pd)
    monkeypatch.setattr(fetch, "pd", pd)
    monkeypatch.setattr(
        fetch,
        "_backup_get_bars",
        lambda *a, **k: pd.DataFrame(
            {
                "t": [datetime(2024, 1, 1, tzinfo=UTC)],
                "o": [1],
                "h": [1],
                "l": [1],
                "c": [1],
                "v": [1],
            }
        ),
    )
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)
    provider_monitor.threshold = 1
    provider_monitor.cooldown = 0
    provider_monitor.fail_counts.clear()
    provider_monitor.disabled_until.clear()
    provider_monitor.disable_counts.clear()
    provider_monitor.outage_start.clear()
    monkeypatch.setattr(fetch, "_alpaca_disabled_until", None, raising=False)


def test_connection_error_fallback_logs(monkeypatch, caplog):
    start, end = _dt_range()
    _setup_common(monkeypatch)

    def _fail(*a, **k):
        provider_monitor.record_failure("alpaca", "connection_error", "boom")
        raise RuntimeError("boom")

    monkeypatch.setattr(fetch, "_fetch_bars", _fail)

    with caplog.at_level(logging.INFO):
        df = fetch.get_minute_df("AAPL", start, end)

    assert isinstance(df, pd.DataFrame)
    assert any(r.message == "DATA_PROVIDER_FAILURE" and getattr(r, "error", "") == "boom" for r in caplog.records)
    assert any(r.message == "BACKUP_PROVIDER_USED" for r in caplog.records)


def test_timeout_fallback_logs(monkeypatch, caplog):
    start, end = _dt_range()
    _setup_common(monkeypatch)

    def _fail(*a, **k):
        provider_monitor.record_failure("alpaca", "timeout", "slow")
        raise RuntimeError("slow")

    monkeypatch.setattr(fetch, "_fetch_bars", _fail)

    with caplog.at_level(logging.INFO):
        df = fetch.get_minute_df("AAPL", start, end)

    assert isinstance(df, pd.DataFrame)
    assert any(r.message == "DATA_PROVIDER_FAILURE" and getattr(r, "error", "") == "slow" for r in caplog.records)
    assert any(r.message == "BACKUP_PROVIDER_USED" for r in caplog.records)
