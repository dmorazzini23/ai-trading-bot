"""Download historical data for backtesting.

This script fetches daily OHLCV bars for a predefined list of tickers and
stores them as CSV files under ``data/``. Existing files are left untouched.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from ai_trading.utils.base import _get_alpaca_rest
from alpaca_trade_api import TimeFrame
from ai_trading.env import ensure_dotenv_loaded
from ai_trading.config.management import get_env

def main() -> None:
    """Fetch bars for each symbol and save to ``data`` directory."""
    ensure_dotenv_loaded()
    api_key = get_env('ALPACA_API_KEY')
    secret_key = get_env('ALPACA_SECRET_KEY')
    base_url = get_env('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    api = _get_alpaca_rest()(api_key, secret_key, base_url)
    symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META']
    start = '2023-01-01'
    end = '2024-01-01'
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    for symbol in symbols:
        out_file = data_dir / f'{symbol}.csv'
        if out_file.exists():
            continue
        try:
            bars = api.get_bars(symbol, TimeFrame.Day, start=start, end=end, adjustment='raw').df
        except (pd.errors.EmptyDataError, KeyError, ValueError, TypeError):
            continue
        if bars is None or bars.empty:
            continue
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level=0)
        df = bars.reset_index()
        rename_map = {'timestamp': 'timestamp', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        df = df.rename(columns=rename_map)
        expected_cols = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[c for c in expected_cols if c in df.columns]]
        if df.empty or 'Close' not in df.columns:
            continue
        try:
            df.to_csv(out_file, index=False)
        except OSError:
            pass
if __name__ == '__main__':
    main()