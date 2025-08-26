from datetime import datetime, UTC

import pandas as pd

import ai_trading.data_fetcher as data_fetcher
from ai_trading.core import bot_engine


class _RespForbidden:
    status_code = 403
    text = ""
    headers = {}

    def json(self):
        return {}


def test_get_bars_unauthorized_sip_returns_empty(monkeypatch):
    """Unauthorized SIP access returns an empty DataFrame."""

    def fake_get(url, params=None, headers=None, timeout=None):  # noqa: ARG001
        return _RespForbidden()

    monkeypatch.setattr(data_fetcher.requests, "get", fake_get)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    df = data_fetcher.get_bars("AAPL", "1Min", start, end, feed="sip")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_data_check_skips_unauthorized_symbols(monkeypatch):
    """Symbols returning empty data are skipped during data_check."""

    class _RespOK:
        status_code = 200
        headers = {"Content-Type": "application/json"}
        text = "{\"bars\":[{\"t\":\"2024-01-01T00:00:00Z\",\"o\":1,\"h\":1,\"l\":1,\"c\":1,\"v\":1}]}"

        def json(self):
            import json

            return json.loads(self.text)

    def fake_get(url, params=None, headers=None, timeout=None):  # noqa: ARG001
        if params and params.get("symbols") == "MSFT":
            return _RespForbidden()
        return _RespOK()

    monkeypatch.setattr(data_fetcher.requests, "get", fake_get)
    symbols = ["AAPL", "MSFT"]
    result = bot_engine.data_check(symbols, feed="sip")
    assert "AAPL" in result
    assert "MSFT" not in result
