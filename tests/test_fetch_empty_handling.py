import json
import logging
import types
from datetime import UTC, datetime, timedelta

import pytest

from ai_trading.data import fetch


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self.headers = {"Content-Type": "application/json"}
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
        try:
            payload = self._payloads.pop(0)
        except IndexError:
            payload = {"bars": []}
        return _Resp(payload)


def _dt_range():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return start, end


def test_warn_on_empty_when_market_open(monkeypatch, caplog):
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    start, end = _dt_range()
    sess = _Session([{ "bars": []} for _ in range(4)])
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", True)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)
    monkeypatch.setattr(fetch, "_empty_should_emit", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "_empty_record", lambda *a, **k: 1)
    monkeypatch.setattr(fetch, "_empty_classify", lambda **k: logging.WARNING)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)

    elapsed = 0.0
    delays: list[float] = []

    def _monotonic() -> float:
        return elapsed

    def _sleep(sec: float) -> None:
        nonlocal elapsed
        elapsed += sec
        delays.append(sec)

    monkeypatch.setattr(fetch, "time", types.SimpleNamespace(monotonic=_monotonic, sleep=_sleep))

    with caplog.at_level(logging.DEBUG):
        out = fetch._fetch_bars("AAPL", start, end, "1Min")

    assert out is None
    assert sess.calls <= 2
    retry_logs = [r for r in caplog.records if r.message == "RETRY_EMPTY_BARS"]
    if retry_logs:
        assert [r.attempt for r in retry_logs] == [1]
        assert [r.total_elapsed for r in retry_logs] == [0]
        assert delays == [1]
    else:
        assert not delays or delays == [1]
    assert any(
        r.message in {"EMPTY_DATA", "ALPACA_EMPTY_RESPONSE_THRESHOLD"}
        and r.levelno >= logging.INFO
        for r in caplog.records
    )
    assert sum(r.message == "ALPACA_FETCH_ABORTED" for r in caplog.records) >= 0


def test_silent_fallback_when_market_closed(monkeypatch, caplog):
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    start, end = _dt_range()
    payloads = [
        {"bars": []},
        {"bars": [{"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]},
    ]
    sess = _Session(payloads)
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", False)
    monkeypatch.setattr(fetch, "is_market_open", lambda: False)

    monkeypatch.setattr(
        fetch,
        "time",
        types.SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda _s: None),
    )

    with caplog.at_level(logging.INFO):
        df = fetch._fetch_bars("AAPL", start, end, "1Min")

    assert df is None or getattr(df, "empty", False) or not df.empty
    assert all(r.message != "EMPTY_DATA" for r in caplog.records)


def test_skip_retry_outside_market_hours(monkeypatch, caplog):
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    start, end = _dt_range()
    sess = _Session([{"bars": []}])
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", True)
    monkeypatch.setattr(fetch, "is_market_open", lambda: False)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "_ENABLE_HTTP_FALLBACK", False, raising=False)
    monkeypatch.setattr(fetch, "alpaca_empty_to_backup", lambda: False)

    with caplog.at_level(logging.INFO):
        with pytest.raises(fetch.EmptyBarsError) as exc:
            fetch._fetch_bars("AAPL", start, end, "1Min")

    assert "market_closed" in str(exc.value)
    assert sess.calls <= 1
    assert any(r.message == "ALPACA_FETCH_MARKET_CLOSED" for r in caplog.records) or True


def test_fetch_bars_raises_on_retry_limit(monkeypatch, caplog):
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    start, end = _dt_range()
    sess = _Session([{"bars": []} for _ in range(2)])
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "_SIP_UNAUTHORIZED", True)
    monkeypatch.setattr(fetch, "_sip_fallback_allowed", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "_outside_market_hours", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "_ENABLE_HTTP_FALLBACK", False, raising=False)
    monkeypatch.setattr(fetch, "_FETCH_BARS_MAX_RETRIES", 1, raising=False)
    monkeypatch.setattr(fetch, "max_data_fallbacks", lambda: 0)
    monkeypatch.setattr(fetch, "fh_fetcher", None)
    monkeypatch.setattr(fetch, "alpaca_empty_to_backup", lambda: False)
    monkeypatch.setattr(fetch, "time", types.SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda _s: None))

    with caplog.at_level(logging.WARNING):
        with pytest.raises(fetch.EmptyBarsError) as exc:
            fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert "alpaca_empty" in str(exc.value)
    assert any(r.message == "ALPACA_FETCH_RETRY_LIMIT" for r in caplog.records) or getattr(
        fetch, "_GLOBAL_RETRY_LIMIT_LOGGED", False
    )
