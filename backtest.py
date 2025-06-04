"""Simple backtesting and hyperparameter optimization.

Usage::

    python backtest.py --symbols SPY,AAPL --start 2023-01-01 --end 2023-06-30 --mode grid

This will search over a default hyperparameter grid and write the best set to
``best_hyperparams.json``.
"""

import argparse
import json
import os
from itertools import product
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf


def load_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load historical daily data for ``symbol`` using yfinance."""
    df = yf.download(symbol, start=start, end=end, progress=False)
    df.index = pd.to_datetime(df.index)
    return df


def run_backtest(symbols, start, end, params) -> dict:
    """Run a very small simulated backtest."""
    data = {s: load_price_data(s, start, end) for s in symbols}
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
                if ret > params["BUY_THRESHOLD"] and cash > 0:
                    qty = int((cash * params["SCALING_FACTOR"]) / price)
                    if qty > 0:
                        cost = qty * price * (1 + params["LIMIT_ORDER_SLIPPAGE"])
                        cash -= cost
                        positions[sym] += qty
                        entry_price[sym] = price
                        peak_price[sym] = price
            else:
                peak_price[sym] = max(peak_price[sym], price)
                drawdown = (price - peak_price[sym]) / peak_price[sym]
                gain = (price - entry_price[sym]) / entry_price[sym]
                if gain >= params["TAKE_PROFIT_FACTOR"] or abs(drawdown) >= params["TRAILING_FACTOR"]:
                    cash += positions[sym] * price * (1 - params["LIMIT_ORDER_SLIPPAGE"])
                    positions[sym] = 0
                    entry_price[sym] = 0
                    peak_price[sym] = 0
        value = cash
        for sym, df in data.items():
            if d in df.index:
                value += positions[sym] * df.loc[d, "Close"]
        portfolio.append(value)

    series = pd.Series(portfolio)
    pct = series.pct_change().dropna()
    sharpe = (pct.mean() / pct.std()) * np.sqrt(252) if not pct.empty else 0.0
    return {"net_pnl": portfolio[-1] - portfolio[0], "sharpe": sharpe}


def optimize_hyperparams(ctx, symbols, backtest_data, param_grid: dict, metric: str = "sharpe") -> dict:
    """Grid search over hyperparameters."""
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
