import types

import pandas as pd

import ai_trading.data.bars as data_bars
from ai_trading.data.bars import safe_get_stock_bars


def test_minute_fallback_none_safe(monkeypatch):
    class Client:
        def get_stock_bars(self, request):
            class Resp:
                df = pd.DataFrame()
            return Resp()

    monkeypatch.setattr(data_bars, "get_minute_df", lambda *a, **k: None)

    req = types.SimpleNamespace(
        symbol_or_symbols=["SPY"],
        timeframe="1Min",
        start=None,
        end=None,
        feed="iex",
    )
    df = safe_get_stock_bars(Client(), req, "SPY", "TEST")
    assert df.empty
