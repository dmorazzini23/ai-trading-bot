import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

import ai_trading.data.fetch as fetch
from ai_trading.monitoring.alerts import AlertSeverity, AlertType


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self.headers = {"Content-Type": "application/json", "x-request-id": "id"}
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = 0
        self.get = self._get

    def _get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        return _Resp(self._payloads.pop(0))


class DummyAlerts:
    def __init__(self):
        self.calls = []

    def create_alert(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _dt_range():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return start, end


def test_primary_provider_success(monkeypatch):
    start, end = _dt_range()
    payload = {"bars": [{"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]}
    sess = _Session([payload])
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_has_alpaca_keys", lambda: True)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)
    alerts = DummyAlerts()
    monkeypatch.setattr(fetch.provider_monitor, "alert_manager", alerts)

    df = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert not df.empty
    assert alerts.calls == []


def test_primary_provider_missing_keys_alert(monkeypatch):
    start, end = _dt_range()
    monkeypatch.setattr(fetch, "_has_alpaca_keys", lambda: False)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)
    monkeypatch.setattr(fetch, "_backup_get_bars", lambda *a, **k: pd.DataFrame())
    alerts = DummyAlerts()
    monkeypatch.setattr(fetch.provider_monitor, "alert_manager", alerts)
    monkeypatch.setattr(fetch, "_ALPACA_KEYS_MISSING_LOGGED", False, raising=False)

    fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert alerts.calls
    args, _ = alerts.calls[0]
    assert args[0] is AlertType.SYSTEM
    assert args[1] is AlertSeverity.CRITICAL
    assert "credentials" in args[2].lower()


def test_primary_provider_disabled_no_alert(monkeypatch):
    start, end = _dt_range()
    monkeypatch.setattr(fetch, "_has_alpaca_keys", lambda: True)
    monkeypatch.setattr(fetch, "_alpaca_disabled_until", datetime.now(UTC) + timedelta(minutes=1), raising=False)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)

    def fake_backup(*_a, **_k):
        return pd.DataFrame([
            {"timestamp": start, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}
        ])

    monkeypatch.setattr(fetch, "_backup_get_bars", fake_backup)
    alerts = DummyAlerts()
    monkeypatch.setattr(fetch.provider_monitor, "alert_manager", alerts)

    df = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert not df.empty
    assert alerts.calls == []
