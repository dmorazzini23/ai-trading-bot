import os
import importlib
import pandas as pd

def test_get_minute_df_uses_finnhub_when_key(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "testkey")
    monkeypatch.delenv("ENABLE_FINNHUB", raising=False)
    import ai_trading.logging as logging_mod
    importlib.reload(logging_mod)
    assert os.getenv("ENABLE_FINNHUB") == "1"
    from ai_trading.data import fetch

    called = {}
    class DummyFetcher:
        is_stub = False
        def fetch(self, symbol, start, end, resolution="1"):
            called["called"] = True
            return pd.DataFrame({"t": [1, 2], "c": [1.0, 2.0]})
    monkeypatch.setattr(fetch, "fh_fetcher", DummyFetcher())
    df = fetch.get_minute_df("AAPL", "2024-01-01", "2024-01-02")
    assert called.get("called") is True
    assert list(df["c"]) == [1.0, 2.0]
