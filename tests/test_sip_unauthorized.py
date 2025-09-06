from datetime import datetime, UTC
import os
from typing import Any

import ai_trading.data.fetch as data_fetcher
from ai_trading.core import bot_engine
from ai_trading.utils.lazy_imports import load_pandas


pd = load_pandas()

class _RespForbidden:
    status_code = 403
    text = ""
    headers = {}

    def json(self):
        return {}


def test_get_bars_unauthorized_sip_returns_empty(monkeypatch):
    """Unauthorized SIP access returns an empty DataFrame."""
    monkeypatch.setattr(data_fetcher, "_SIP_UNAUTHORIZED", False, raising=False)

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
    monkeypatch.setattr(data_fetcher, "_SIP_UNAUTHORIZED", False, raising=False)

    if os.getenv("ALLOW_EXTERNAL_NETWORK", "0") != "1":
        def fake_get_bars_df(symbol: str, timeframe: str | Any, *a, **k):  # noqa: ANN401
            if symbol == "MSFT":
                raise ValueError("unauthorized")
            return pd.DataFrame({"v": [1]})

        monkeypatch.setattr(bot_engine, "get_bars_df", fake_get_bars_df)

    symbols = ["AAPL", "MSFT"]
    result = bot_engine.data_check(symbols, feed="sip")
    assert "AAPL" in result
    assert "MSFT" not in result

