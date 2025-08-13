from __future__ import annotations
import types
import pandas as pd
from datetime import datetime

__version__ = "stub-0.0"

# AI-AGENT-REF: offline yfinance stub

def _make_df():
    # tiny deterministic OHLCV frame
    idx = pd.date_range(datetime(2024,1,2,9,30), periods=5, freq="1min")
    return pd.DataFrame({
        "Open":[100,101,102,103,104],
        "High":[101,102,103,104,105],
        "Low":[ 99,100,101,102,103],
        "Close":[100,101,102,103,104],
        "Volume":[1000,1001,1002,1003,1004],
    }, index=idx)

def download(tickers, period=None, interval=None, **kwargs):
    # mimic yfinance.download signature, return a simple DataFrame
    df = _make_df()
    if isinstance(tickers, (list, tuple, set)) or (isinstance(tickers, str) and "," in tickers):
        return {t.strip(): df.copy() for t in (tickers if not isinstance(tickers, str) else tickers.split(","))}
    return df

class _FastInfo(dict):
    def __getattr__(self, k):
        return self.get(k)

class Ticker:
    def __init__(self, symbol: str):
        self.ticker = symbol
        self.info = {"symbol": symbol, "sector": "Technology", "currency": "USD"}
        self.fast_info = _FastInfo(last_price=104.0, day_high=105.0, day_low=99.0)

    def history(self, period="1d", interval="1m", **kwargs):
        return _make_df()

__all__ = ["download", "Ticker", "__version__"]
