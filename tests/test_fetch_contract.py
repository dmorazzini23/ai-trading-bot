from tests.optdeps import require
require("pandas")
import os
import sys
import types

import pandas as pd

os.environ.setdefault("ALPACA_API_KEY", "dummy")
os.environ.setdefault("ALPACA_SECRET_KEY", "dummy")
dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *a, **k: None
dotenv_stub.dotenv_values = lambda *a, **k: {}
dotenv_stub.find_dotenv = lambda *a, **k: ""
sys.modules["dotenv"] = dotenv_stub

from ai_trading import data_fetcher


class DummyClient:
    pass


def test_get_bars_never_none(monkeypatch):
    now = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": [now],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [100],
        }
    )
    monkeypatch.setattr(
        data_fetcher,
        "_alpaca_get_bars",
        lambda client, symbol, start, end, timeframe="1Day": df,
    )
    result = data_fetcher.get_bars(
        "AAPL", "1Day", now - pd.Timedelta(days=1), now, feed=DummyClient()
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert str(result["timestamp"].dt.tz) == "UTC"
