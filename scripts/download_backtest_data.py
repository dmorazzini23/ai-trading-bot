#!/usr/bin/env python3
"""Download historical data for backtesting.

This script fetches daily OHLCV bars for a predefined list of tickers and
stores them as CSV files under ``data/``. Existing files are left untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from ai_trading.utils.base import _get_alpaca_rest
from alpaca_trade_api import TimeFrame
from dotenv import load_dotenv


def main() -> None:
    """Fetch bars for each symbol and save to ``data`` directory."""
    load_dotenv(dotenv_path=".env", override=True)

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if api_key:
        os.environ.setdefault("APCA_API_KEY_ID", api_key)
    if secret_key:
        os.environ.setdefault("APCA_API_SECRET_KEY", secret_key)

    api = _get_alpaca_rest()(api_key, secret_key, base_url)

    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "META"]
    start = "2023-01-01"
    end = "2024-01-01"

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    for symbol in symbols:
        out_file = data_dir / f"{symbol}.csv"
        if out_file.exists():
            continue

        try:
            bars = api.get_bars(
                symbol,
                TimeFrame.Day,  # TimeFrame.Minute can be used for intraday data
                start=start,
                end=end,
                adjustment="raw",
            ).df
        except Exception:  # pragma: no cover - network call
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
        except OSError:  # pragma: no cover - disk error
            pass


if __name__ == "__main__":  # pragma: no cover - manual utility
    main()
