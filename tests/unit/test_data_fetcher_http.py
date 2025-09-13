from __future__ import annotations


import json
from datetime import UTC, datetime, timedelta

import pytest

pd = pytest.importorskip("pandas")

import ai_trading.data.fetch as df


class _Resp:
    def __init__(
        self,
        status_code: int,
        payload: dict | None = None,
        content_type: str = "application/json",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = {"Content-Type": content_type}
        if headers:
            self.headers.update(headers)
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


def _bars_payload(ts_iso: str) -> dict:
    return {
        "bars": [
            {
                "t": ts_iso,
                "o": 10.0,
                "h": 11.0,
                "l": 9.5,
                "c": 10.5,
                "v": 1000,
            }
        ]
    }


def _dt_range(minutes: int = 5):
    end = datetime.now(UTC).replace(microsecond=0)
    start = end - timedelta(minutes=minutes)
    return start, end


@pytest.fixture(autouse=True)
def _force_window(monkeypatch):
    monkeypatch.setattr(df, "_window_has_trading_session", lambda *a, **k: True)


@pytest.mark.parametrize("status_first", [401, 403])
def test_sip_unauthorized_returns_empty(monkeypatch: pytest.MonkeyPatch, status_first: int):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    monkeypatch.setattr(df, "_ALLOW_SIP", True, raising=False)
    monkeypatch.setattr(df, "_FALLBACK_WINDOWS", set(), raising=False)
    monkeypatch.setattr(df, "_FALLBACK_UNTIL", {}, raising=False)
    monkeypatch.setattr(df, "_alpaca_disabled_until", None, raising=False)
    calls = {"count": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1
        return _Resp(status_first, payload={"message": "auth required"})

    monkeypatch.setattr(df._HTTP_SESSION, "get", fake_get)
    monkeypatch.setattr(df.requests, "get", fake_get, raising=False)
    monkeypatch.setattr(df, "_backup_get_bars", lambda *a, **k: pd.DataFrame())

    start, end = _dt_range(2)
    out = df._fetch_bars("TEST", start, end, "1Min", feed="sip")
    assert isinstance(out, pd.DataFrame) and out.empty
    assert calls["count"] == 1
    assert df._SIP_UNAUTHORIZED is True


def test_sip_fallback_skipped_when_marked_unauthorized(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", True, raising=False)
    calls = {"count": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1
        return _Resp(429, payload={"message": "rate limit"})

    monkeypatch.setattr(df._HTTP_SESSION, "get", fake_get)
    monkeypatch.setattr(df.requests, "get", fake_get, raising=False)

    start, end = _dt_range(2)
    with pytest.raises(ValueError, match="rate_limited"):
        df._fetch_bars("TEST", start, end, "1Min", feed="iex")
    assert calls["count"] == 1


def test_no_additional_sip_requests_after_unauthorized(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    monkeypatch.setattr(df, "_ALLOW_SIP", True, raising=False)
    monkeypatch.setattr(df, "_FALLBACK_WINDOWS", set(), raising=False)
    monkeypatch.setattr(df, "_FALLBACK_UNTIL", {}, raising=False)
    monkeypatch.setattr(df, "_alpaca_disabled_until", None, raising=False)
    feeds: list[str | None] = []

    def fake_get(url, params=None, headers=None, timeout=None):
        feeds.append((params or {}).get("feed"))
        return _Resp(401, payload={"message": "auth"})

    monkeypatch.setattr(df._HTTP_SESSION, "get", fake_get)

    backup_calls = {"count": 0}

    def fake_backup(symbol, start, end, interval):
        backup_calls["count"] += 1
        ts = datetime.now(UTC)
        return pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.5,
                    "close": 10.5,
                    "volume": 1000,
                }
            ]
        ).set_index("timestamp")

    monkeypatch.setattr(df, "_backup_get_bars", fake_backup)

    start, end = _dt_range(2)
    out = df._fetch_bars("TEST", start, end, "1Min", feed="sip")

    assert feeds == ["sip"]
    assert backup_calls["count"] == 1
    assert isinstance(out, pd.DataFrame) and not out.empty
    assert df._SIP_UNAUTHORIZED is True


def test_timeout_triggers_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    calls = {"count": 0}

    class _Timeout(df.Timeout):
        pass

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1
        feed = (params or {}).get("feed")
        if calls["count"] == 1 and feed == "sip":
            raise _Timeout("timeout")
        ts_iso = datetime.now(UTC).isoformat()
        return _Resp(200, payload=_bars_payload(ts_iso))

    monkeypatch.setattr(df._HTTP_SESSION, "get", fake_get)

    start, end = _dt_range(2)
    out = df.get_bars("TEST", timeframe="1Min", start=start, end=end, feed="sip", adjustment="raw")
    assert isinstance(out, pd.DataFrame) and not out.empty
    assert calls["count"] >= 2


def test_429_rate_limit_triggers_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    calls = {"count": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1
        feed = (params or {}).get("feed")
        ts_iso = datetime.now(UTC).isoformat()
        if calls["count"] == 1 and feed == "sip":
            return _Resp(429, payload={"message": "rate limit"})
        return _Resp(200, payload=_bars_payload(ts_iso))

    monkeypatch.setattr(df._HTTP_SESSION, "get", fake_get)

    start, end = _dt_range(2)
    out = df.get_bars("TEST", timeframe="1Min", start=start, end=end, feed="sip", adjustment="raw")
    assert isinstance(out, pd.DataFrame) and not out.empty
    assert calls["count"] >= 2


def test_empty_bars_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    calls = {"count": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1
        ts_iso = datetime.now(UTC).isoformat()
        if calls["count"] == 1:
            return _Resp(200, payload={"bars": []})
        return _Resp(200, payload=_bars_payload(ts_iso))

    monkeypatch.setattr(df._HTTP_SESSION, "get", fake_get)

    start, end = _dt_range(2)
    out = df.get_bars("TEST", timeframe="1Min", start=start, end=end, feed="iex", adjustment="raw")
    assert isinstance(out, pd.DataFrame) and not out.empty
    assert calls["count"] >= 2


def test_sip_fallback_precheck_skips_request(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    monkeypatch.setattr(df, "_SIP_PRECHECK_DONE", False, raising=False)
    feeds: list[str | None] = []

    def fake_get(url, params=None, headers=None, timeout=None):  # noqa: ARG001
        feed = (params or {}).get("feed")
        feeds.append(feed)
        if feed == "iex":
            return _Resp(429, payload={"message": "rate limit"})
        return _Resp(403, payload={"message": "auth"})

    monkeypatch.setattr(df._HTTP_SESSION, "get", fake_get)
    start, end = _dt_range(2)
    with caplog.at_level("WARNING"):
        with pytest.raises(ValueError, match="rate_limited"):
            df._fetch_bars("TEST", start, end, "1Min", feed="iex")
    assert feeds == ["iex", "sip"]
    assert df._SIP_UNAUTHORIZED is True
    assert "UNAUTHORIZED_SIP" in caplog.text


def test_rate_limit_backoff(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(df, "_SIP_UNAUTHORIZED", False, raising=False)
    monkeypatch.setattr(df, "_FALLBACK_WINDOWS", set(), raising=False)
    monkeypatch.setattr(df, "_FALLBACK_UNTIL", {}, raising=False)
    monkeypatch.setattr(df, "max_data_fallbacks", lambda: 0)
    monkeypatch.setattr(df, "provider_priority", lambda: ["alpaca_iex"])
    sleep_calls: list[float] = []
    monkeypatch.setattr(df.time, "sleep", lambda s: sleep_calls.append(s))
    resp_iter = iter(
        [
            _Resp(429, payload={"message": "rate limit"}),
            _Resp(200, payload=_bars_payload(datetime.now(UTC).isoformat())),
        ]
    )
    monkeypatch.setattr(df._HTTP_SESSION, "get", lambda *a, **k: next(resp_iter))
    start, end = _dt_range(2)
    out = df._fetch_bars("TEST", start, end, "1Min", feed="iex")
    assert isinstance(out, pd.DataFrame) and not out.empty
    assert sleep_calls and sleep_calls[0] >= 1.0


def test_rate_limit_disable_and_recover(monkeypatch: pytest.MonkeyPatch):
    from ai_trading.data.fetch.metrics import (
        inc_provider_disable_total,
        provider_disabled,
    )
    from ai_trading.config import settings as config_settings

    monkeypatch.setenv("BACKUP_DATA_PROVIDER", "yahoo")
    config_settings.get_settings.cache_clear()
    monkeypatch.setattr(df, "_FALLBACK_WINDOWS", set(), raising=False)
    monkeypatch.setattr(df, "_FALLBACK_UNTIL", {}, raising=False)
    monkeypatch.setattr(df, "_alpaca_disabled_until", None, raising=False)
    monkeypatch.setattr(df.provider_monitor, "threshold", 1, raising=False)
    monkeypatch.setattr(df.provider_monitor, "cooldown", 1, raising=False)
    df.provider_monitor.fail_counts.clear()
    df.provider_monitor.disabled_until.clear()
    calls = {"alpaca": 0, "backup": 0}

    def rate_limit_resp(url, params=None, headers=None, timeout=None):
        calls["alpaca"] += 1
        return _Resp(429, payload={"message": "rate limit"}, headers={"Retry-After": "1"})

    def backup_resp(symbol, start, end, interval):
        calls["backup"] += 1
        ts = datetime.now(UTC)
        return pd.DataFrame(
            {
                "timestamp": [ts],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [0],
            }
        ).set_index("timestamp")

    monkeypatch.setattr(df._HTTP_SESSION, "get", rate_limit_resp)
    monkeypatch.setattr(df, "_backup_get_bars", backup_resp)
    start, end = _dt_range(2)
    before = inc_provider_disable_total("alpaca")
    out = df._fetch_bars("TEST", start, end, "1Min", feed="iex")
    assert calls["alpaca"] == 1
    assert calls["backup"] == 1
    assert provider_disabled("alpaca") == 1
    assert inc_provider_disable_total("alpaca") == before + 1
    assert isinstance(out, pd.DataFrame) and not out.empty
    assert df._alpaca_disabled_until is not None

    # Subsequent call while disabled uses backup without hitting alpaca
    calls["backup"] = 0
    out2 = df._fetch_bars("TEST", start, end, "1Min", feed="iex")
    assert calls["alpaca"] == 1
    assert calls["backup"] == 1
    assert isinstance(out2, pd.DataFrame)

    # Simulate cooldown expiry and ensure provider re-enables
    df._FALLBACK_WINDOWS.clear()
    df._FALLBACK_UNTIL.clear()
    df._alpaca_disabled_until = datetime.now(UTC) - timedelta(seconds=1)

    def ok_resp(url, params=None, headers=None, timeout=None):
        calls["alpaca"] += 1
        ts_iso = datetime.now(UTC).isoformat()
        return _Resp(200, payload=_bars_payload(ts_iso))

    monkeypatch.setattr(df._HTTP_SESSION, "get", ok_resp)
    out3 = df._fetch_bars("TEST", start, end, "1Min", feed="iex")
    assert calls["alpaca"] == 2
    assert provider_disabled("alpaca") == 0
    assert df._alpaca_disabled_until is None
    assert isinstance(out3, pd.DataFrame) and not out3.empty


def test_disable_exponential_backoff(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(df, "_alpaca_disabled_until", None, raising=False)
    monkeypatch.setattr(df, "_alpaca_disable_count", 0, raising=False)
    base = timedelta(seconds=1)
    df._disable_alpaca(base)
    first_until = df._alpaca_disabled_until
    df._disable_alpaca(base)
    second_until = df._alpaca_disabled_until
    assert second_until - first_until >= base


def test_success_resets_alert_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(df, "_ALPACA_DISABLED_ALERTED", True, raising=False)
    monkeypatch.setattr(df, "_alpaca_disabled_until", None, raising=False)
    monkeypatch.setattr(df, "_FALLBACK_WINDOWS", set(), raising=False)
    monkeypatch.setattr(df, "_FALLBACK_UNTIL", {}, raising=False)

    def ok_resp(url, params=None, headers=None, timeout=None):
        ts_iso = datetime.now(UTC).isoformat()
        return _Resp(200, payload=_bars_payload(ts_iso))

    monkeypatch.setattr(df._HTTP_SESSION, "get", ok_resp)

    start, end = _dt_range(2)
    out = df._fetch_bars("TEST", start, end, "1Min", feed="iex")
    assert isinstance(out, pd.DataFrame) and not out.empty
    assert df._ALPACA_DISABLED_ALERTED is False
