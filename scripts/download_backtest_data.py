"""Download historical data for backtesting.

This script fetches daily OHLCV bars for a predefined list of tickers and
stores them as CSV files under ``data/``. Existing files are left untouched.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from ai_trading.env import ensure_dotenv_loaded
from ai_trading.config.management import get_env


def main() -> None:
    """Fetch bars for each symbol and save to ``data`` directory."""
    ensure_dotenv_loaded()
    api_key = get_env("ALPACA_API_KEY")
    secret_key = get_env("ALPACA_SECRET_KEY")
    base_url = get_env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    client = StockHistoricalDataClient(
        api_key=api_key, secret_key=secret_key, base_url=base_url
    )

    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "META"]
    start = datetime(2023, 1, 1, tzinfo=ZoneInfo("UTC"))
    end = datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC"))
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    for symbol in symbols:
        out_file = data_dir / f"{symbol}.csv"
        if out_file.exists():
            continue
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment="raw",
        )
        try:
            bars = client.get_stock_bars(req).df
        except (pd.errors.EmptyDataError, KeyError, ValueError, TypeError):
            continue
        if bars is None or bars.empty:
            continue
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level=0)
        df = bars.reset_index()
        rename_map = {
            "timestamp": "timestamp",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns=rename_map)
        expected_cols = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in expected_cols if c in df.columns]]
        if df.empty or "Close" not in df.columns:
            continue
        try:
            df.to_csv(out_file, index=False)
        except OSError:
            pass


if __name__ == "__main__":
    main()
