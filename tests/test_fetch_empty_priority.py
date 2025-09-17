import json
import types
from datetime import UTC, datetime, timedelta

import pandas as pd

from ai_trading.data import fetch


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


def _dt_range():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    return start, end


def test_fetch_bars_handles_empty_priority(monkeypatch):
    start, end = _dt_range()
    payload = {"bars": [{"t": "2024-01-01T00:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]}
    sess = _Session([payload])
    monkeypatch.setattr(fetch, "_HTTP_SESSION", sess)
    monkeypatch.setattr(fetch, "provider_priority", lambda: [])
    monkeypatch.setattr(fetch, "max_data_fallbacks", lambda: 1)
    monkeypatch.setattr(fetch, "_window_has_trading_session", lambda *a, **k: True)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)

    df = fetch._fetch_bars("AAPL", start, end, "1Min", feed="iex")

    assert not df.empty
    assert sess.calls == 1


def test_get_minute_df_handles_empty_priority(monkeypatch):
    start, end = _dt_range()
    monkeypatch.setattr(fetch, "_has_alpaca_keys", lambda: True)
    monkeypatch.setattr(fetch, "fh_fetcher", None)
    monkeypatch.setenv("ENABLE_FINNHUB", "0")
    monkeypatch.setattr(fetch, "provider_priority", lambda: [])
    monkeypatch.setattr(fetch, "max_data_fallbacks", lambda: 1)
    monkeypatch.setattr(fetch, "is_market_open", lambda: True)
    monkeypatch.setattr(fetch, "time", types.SimpleNamespace(sleep=lambda _s: None))

    tf_key = ("AAPL", "1Min")
    fetch._EMPTY_BAR_COUNTS[tf_key] = fetch._EMPTY_BAR_THRESHOLD - 1

    def _raise(*_a, **_k):
        raise fetch.EmptyBarsError("no bars")

    monkeypatch.setattr(fetch, "_fetch_bars", _raise)

    df = fetch.get_minute_df("AAPL", start, end)

    assert df is None or (isinstance(df, pd.DataFrame) and df.empty)
