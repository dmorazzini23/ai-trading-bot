"""Simple backtesting and hyperparameter optimization.

Usage::

    python3 backtest.py --symbols SPY,AAPL --start 2023-01-01 --end 2023-06-30 --mode grid

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


def load_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Load (or re‐use cached) historical daily data for `symbol` using yfinance.
    Caches to "cache_{symbol}_{start}_{end}.csv" on disk, so future calls are instant.
    Retries up to 3 times if any exception occurs during download.
    """
    cache_fname = f"cache_{symbol}_{start}_{end}.csv"

    # 1) If cached file exists, load it and return
    if os.path.exists(cache_fname):
        try:
            df_cached = pd.read_csv(cache_fname, index_col=0, parse_dates=True)
            return df_cached
        except Exception:
            # If cache is corrupted, remove it and re‐download
            try:
                os.remove(cache_fname)
            except Exception:
                pass

    # 2) Otherwise, attempt to download with up to 3 retries
    df_final = pd.DataFrame()
    for attempt in range(1, 4):
        try:
            raw = yf.download(symbol, start=start, end=end, progress=False)
            raw.index = pd.to_datetime(raw.index)
            if not raw.empty:
                # Keep only OHLCV columns
                df_final = raw.rename(columns={
                    "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"
                })[["Open", "High", "Low", "Close", "Volume"]]
            break
        except Exception as e:
            if attempt < 3:
                print(f"  ▶ Failed to download {symbol} (attempt {attempt}/3): {e!r}. Sleeping 2s…")
                time.sleep(2)
            else:
                print(f"  ▶ Final attempt failed for {symbol}; proceeding with empty DataFrame.")
    # 3) Save to cache (even if empty)
    try:
        df_final.to_csv(cache_fname)
    except Exception:
        pass

    # 4) Polite 1s pause before next symbol
    time.sleep(1)
    return df_final


def run_backtest(symbols, start, end, params) -> dict:
    """
    Run a very small simulated backtest using cached yfinance data.
    Returns a dict with "net_pnl" and "sharpe".
    """
    # 1) Load (or download+cache) each symbol’s DataFrame
    data: dict[str, pd.DataFrame] = {}
    for s in symbols:
        data[s] = load_price_data(s, start, end)

    cash = 100000.0
    positions = {s: 0 for s in symbols}
    entry_price = {s: 0.0 for s in symbols}
    peak_price = {s: 0.0 for s in symbols}
    portfolio = []

    dates = pd.date_range(start, end, freq="B")
    for d in dates:
        for sym, df in data.items():
            if d not in df.index:
                continue
            price = df.loc[d, "Open"]
            ret = df.loc[d, "Close"] / df.loc[d, "Open"] - 1

            if positions[sym] == 0:
                # Entry logic
                if (not np.isnan(ret)) and ret > params["BUY_THRESHOLD"] and cash > 0:
                    qty = int((cash * params["SCALING_FACTOR"]) / price)
                    if qty > 0:
                        cost = qty * price * (1 + params["LIMIT_ORDER_SLIPPAGE"])
                        cash -= cost
                        positions[sym] += qty
                        entry_price[sym] = price
                        peak_price[sym] = price
            else:
                # Update peak, possibly exit
                peak_price[sym] = max(peak_price[sym], price)
                drawdown = (price - peak_price[sym]) / peak_price[sym]
                gain = (price - entry_price[sym]) / entry_price[sym]
                if gain >= params["TAKE_PROFIT_FACTOR"] or abs(drawdown) >= params["TRAILING_FACTOR"]:
                    cash += positions[sym] * price * (1 - params["LIMIT_ORDER_SLIPPAGE"])
                    positions[sym] = 0
                    entry_price[sym] = 0
                    peak_price[sym] = 0

        # Compute portfolio value at close
        total_value = cash
        for sym, df in data.items():
            if d in df.index:
                total_value += positions[sym] * df.loc[d, "Close"]
        portfolio.append(total_value)

    if not portfolio:
        return {"net_pnl": 0.0, "sharpe": float("nan")}

    series = pd.Series(portfolio, index=dates)
    pct = series.pct_change().dropna()

    if pct.empty or pct.std() == 0:
        sharpe = float("nan")
    else:
        sharpe = (pct.mean() / pct.std()) * np.sqrt(252)

    net_pnl = portfolio[-1] - portfolio[0]
    return {"net_pnl": net_pnl, "sharpe": sharpe}


def optimize_hyperparams(ctx, symbols, backtest_data, param_grid: dict, metric: str = "sharpe") -> dict:
    """
    Grid search over hyperparameters.
    If Sharpe is nan, treat it as a very low score (never wins).
    """
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    best_params = None
    best_score = -float("inf")

    for combo in combos:
        params = dict(zip(keys, combo))
        result = run_backtest(symbols, backtest_data["start"], backtest_data["end"], params)
        score = result.get(metric, 0.0)

        # If score is nan, replace with very low number
        if isinstance(score, float) and np.isnan(score):
            score = -float("inf")

        print(f"  ▶ Testing {params}  →  {metric}={score:.6f}")
        if score > best_score:
            best_score = score
            best_params = params.copy()

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
    print(f"▶ Starting grid search over {len(symbols)} symbols from {args.start} to {args.end}...\n")
    best = optimize_hyperparams(None, symbols, data_cfg, param_grid, metric="sharpe")

    # Write results
    with open("best_hyperparams.json", "w") as f:
        json.dump(best, f, indent=2)

    print(f"\n✔ Best hyperparameters: {best}")


if __name__ == "__main__":
    main()
