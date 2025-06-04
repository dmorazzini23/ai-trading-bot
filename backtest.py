"""Simple backtesting and hyperparameter optimization.

Usage::

    python backtest.py --symbols SPY,AAPL --start 2023-01-01 --end 2023-06-30 --mode grid

This will search over a default hyperparameter grid and write the best set to
``best_hyperparams.json``.
"""

import argparse
import json
import os
import time
from itertools import product
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf


def load_price_data(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """
    Download historical daily data for all `symbols` in one yfinance call (to minimize rate-limit errors),
    cache each symbol to CSV, and return a dict mapping each symbol -> its DataFrame.

    - If a cached file "cache_{symbol}_{start}_{end}.csv" exists, read from that instead of downloading.
    - Otherwise, download all missing symbols in a single yf.download([...]) call,
      split the combined DataFrame, save each to its cache CSV, and return them.
    """
    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    # 1) Check for existing cache files
    for s in symbols:
        cache_fname = f"cache_{s}_{start}_{end}.csv"
        if os.path.exists(cache_fname):
            try:
                df_cached = pd.read_csv(cache_fname, index_col=0, parse_dates=True)
                out[s] = df_cached
            except Exception:
                # If reading fails, treat as missing
                missing.append(s)
        else:
            missing.append(s)

    # 2) Download any missing symbols in one batch
    if missing:
        try:
            combined = yf.download(
                missing,
                start=start,
                end=end,
                progress=False
            )
        except Exception:
            combined = pd.DataFrame()

        # If combined is empty or missing columns, we’ll assign empty DataFrames below
        for s in missing:
            cache_fname = f"cache_{s}_{start}_{end}.csv"
            df_s: pd.DataFrame

            try:
                # yfinance returns a MultiIndex if you passed a list of tickers
                # Level 0 = OHLCV, Level 1 = symbol.
                df_symbol = combined.xs(s, axis=1, level=1).copy()
                df_symbol.index = pd.to_datetime(df_symbol.index)
                # Keep only the standard columns if they exist
                df_s = df_symbol.rename(
                    columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"}
                )[[ "Open", "High", "Low", "Close", "Volume" ]]
            except Exception:
                # If download failed or symbol not present, return an empty DataFrame
                df_s = pd.DataFrame()

            out[s] = df_s

            # Write to cache for next time
            try:
                df_s.to_csv(cache_fname)
            except Exception:
                pass

        # Small pause to be polite, in case we re-run in a loop
        time.sleep(1)

    return out


def run_backtest(symbols: list[str], start: str, end: str, params: dict) -> dict:
    """
    Run a very simple backtest over business days between `start` and `end`.
    - symbols: list of tickers to backtest (e.g. ["SPY","AAPL"])
    - params: dict with keys:
        "BUY_THRESHOLD", "SCALING_FACTOR", "LIMIT_ORDER_SLIPPAGE",
        "TAKE_PROFIT_FACTOR", "TRAILING_FACTOR"
    Returns: {"net_pnl": <final equity - initial equity>, "sharpe": <annualized>}.
    """
    # 1) Load (or download+cache) all price data
    data = load_price_data(symbols, start, end)

    cash = 100_000.0
    positions = {s: 0 for s in symbols}
    entry_price = {s: 0.0 for s in symbols}
    peak_price = {s: 0.0 for s in symbols}
    portfolio = []

    dates = pd.date_range(start, end, freq="B")
    for d in dates:
        # 2) For each symbol, decide whether to enter/exit
        for sym in symbols:
            df = data.get(sym, pd.DataFrame())
            if df is None or df.empty or d not in df.index:
                continue

            price_open = df.loc[d, "Open"]
            price_close = df.loc[d, "Close"]
            ret = price_close / price_open - 1

            # If not currently holding: consider entering long if return > BUY_THRESHOLD
            if positions[sym] == 0:
                if ret > params["BUY_THRESHOLD"] and cash > 0:
                    qty = int((cash * params["SCALING_FACTOR"]) / price_open)
                    if qty > 0:
                        cost = qty * price_open * (1 + params["LIMIT_ORDER_SLIPPAGE"])
                        cash -= cost
                        positions[sym] += qty
                        entry_price[sym] = price_open
                        peak_price[sym] = price_open
            else:
                # If already holding, update peak price and check for TP or trailing stop
                peak_price[sym] = max(peak_price[sym], price_close)
                drawdown = (price_close - peak_price[sym]) / peak_price[sym]
                gain = (price_close - entry_price[sym]) / entry_price[sym]
                if (gain >= params["TAKE_PROFIT_FACTOR"]) or (abs(drawdown) >= params["TRAILING_FACTOR"]):
                    proceeds = positions[sym] * price_close * (1 - params["LIMIT_ORDER_SLIPPAGE"])
                    cash += proceeds
                    positions[sym] = 0
                    entry_price[sym] = 0
                    peak_price[sym] = 0

        # 3) Compute total portfolio value at close of this date
        total_value = cash
        for sym in symbols:
            df = data.get(sym, pd.DataFrame())
            if df is None or df.empty or d not in df.index:
                continue
            total_value += positions[sym] * df.loc[d, "Close"]

        portfolio.append(total_value)

    # 4) Compute sharpe
    series = pd.Series(portfolio)
    pct = series.pct_change().dropna()
    if not pct.empty and pct.std() != 0:
        sharpe = (pct.mean() / pct.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    net_pnl = portfolio[-1] - portfolio[0] if portfolio else 0.0
    return {"net_pnl": net_pnl, "sharpe": sharpe}


def optimize_hyperparams(
    ctx,
    symbols: list[str],
    backtest_data: dict,
    param_grid: dict,
    metric: str = "sharpe"
) -> dict:
    """
    Grid search over hyperparameters in `param_grid`.
    - symbols: list of tickers (e.g. ["SPY","AAPL"])
    - backtest_data: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    - param_grid: e.g.
        {
          "BUY_THRESHOLD": [0.15, 0.2, 0.25],
          "TRAILING_FACTOR": [1.0, 1.2, 1.5],
          "TAKE_PROFIT_FACTOR": [1.5, 1.8, 2.0],
          "SCALING_FACTOR": [0.2, 0.3],
          "LIMIT_ORDER_SLIPPAGE": [0.001, 0.005],
        }
    Returns the best‐scoring param set by the given metric (default="sharpe").
    """
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    best_params = None
    best_score = -float("inf")

    for combo in combos:
        params = dict(zip(keys, combo))
        result = run_backtest(symbols, backtest_data["start"], backtest_data["end"], params)
        score = result.get(metric, 0)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params or {}


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimizer")
    parser.add_argument("--symbols", required=True, help="Comma separated symbols")
    parser.add_argument("--start", required=True, help="Backtest start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--mode", choices=["grid"], default="grid")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    param_grid = {
        "BUY_THRESHOLD": [0.15, 0.2, 0.25],
        "TRAILING_FACTOR": [1.0, 1.2, 1.5],
        "TAKE_PROFIT_FACTOR": [1.5, 1.8, 2.0],
        "SCALING_FACTOR": [0.2, 0.3],
        "LIMIT_ORDER_SLIPPAGE": [0.001, 0.005],
    }

    data_cfg = {"start": args.start, "end": args.end}
    best = optimize_hyperparams(None, symbols, data_cfg, param_grid, metric="sharpe")

    with open("best_hyperparams.json", "w") as f:
        json.dump(best, f, indent=2)
    print("Best hyperparameters:", best)


if __name__ == "__main__":
    main()
