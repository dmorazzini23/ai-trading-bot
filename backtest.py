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
import warnings
from itertools import product
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

BACKTEST_WINDOW_DAYS = 365
import pandas as pd
import numpy as np
from data_fetcher import get_historical_data, DataFetchError
from alpaca_trade_api.rest import APIError
from tenacity import RetryError


def load_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load (or fetch and cache) historical daily data via Alpaca."""
    cache_fname = f"cache_{symbol}_{start}_{end}.csv"

    # 1) If cached file exists, load it and return
    if os.path.exists(cache_fname):
        try:
            df_cached = pd.read_csv(cache_fname, index_col=0, parse_dates=True)
            return df_cached
        except (OSError, pd.errors.ParserError, ValueError):
            # If cache is corrupted, remove it and re‐download
            try:
                os.remove(cache_fname)
            except OSError:
                pass

    # 2) Otherwise, attempt to fetch from Alpaca with retries
    df_final = pd.DataFrame()
    for attempt in range(1, 4):
        try:
            df_final = get_historical_data(
                symbol,
                datetime.fromisoformat(start).date(),
                datetime.fromisoformat(end).date(),
                "1Day",
            )
            break
        except (APIError, DataFetchError, RetryError) as e:
            if attempt < 3:
                print(
                    f"  ▶ Failed to fetch {symbol} (attempt {attempt}/3): {e!r}. Sleeping 2s…"
                )
                time.sleep(2)
            else:
                print(
                    f"  ▶ Final attempt failed for {symbol}; proceeding with empty DataFrame."
                )
    # 3) Save to cache (even if empty)
    try:
        df_final.to_csv(cache_fname)
    except OSError:
        pass

    # 4) Polite 1s pause before next symbol
    time.sleep(1)
    return df_final


def run_backtest(symbols, start, end, params) -> dict:
    """
    Run a very small simulated backtest using cached Alpaca data.
    Returns a dict with "net_pnl" and "sharpe".
    """
    # 1) Load (or download+cache) each symbol’s DataFrame
    data: dict[str, pd.DataFrame] = {}
    for s in symbols:
        df_sym = load_price_data(s, start, end)
        if not df_sym.empty:
            df_sym["ret"] = df_sym["Close"].pct_change().fillna(0)
        data[s] = df_sym

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
            ret = df.loc[d, "ret"]

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
                if (
                    gain >= params["TAKE_PROFIT_FACTOR"]
                    or abs(drawdown) >= params["TRAILING_FACTOR"]
                ):
                    cash += (
                        positions[sym] * price * (1 - params["LIMIT_ORDER_SLIPPAGE"])
                    )
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


def optimize_hyperparams(
    ctx, symbols, backtest_data, param_grid: dict, metric: str = "sharpe"
) -> dict:
    """
    Grid search over hyperparameters.

    1) Compute Sharpe for each combination.
    2) If at least one combination yields a finite Sharpe, pick the highest‐Sharpe combo.
    3) Otherwise, fall back to whichever combo gave the highest net_pnl.
    """
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    best_params_sharpe: dict = {}
    best_score_sharpe = -float("inf")
    best_params_pnl: dict = {}
    best_score_pnl = -float("inf")

    for combo in combos:
        params = dict(zip(keys, combo))
        result = run_backtest(
            symbols, backtest_data["start"], backtest_data["end"], params
        )
        score_sh = result.get(metric, 0.0)
        netp = result.get("net_pnl", 0.0)

        # If Sharpe is nan, treat it as extremely low
        if isinstance(score_sh, float) and np.isnan(score_sh):
            score_sh = -float("inf")

        print(f"  ▶ Testing {params}  →  {metric}={score_sh:.6f},  net_pnl={netp:.2f}")

        # Track best by Sharpe
        if score_sh > best_score_sharpe:
            best_score_sharpe = score_sh
            best_params_sharpe = params.copy()
        # Track best by net_pnl (in case we need fallback)
        if netp > best_score_pnl:
            best_score_pnl = netp
            best_params_pnl = params.copy()

    if best_score_sharpe > -float("inf"):
        # At least one combo had a finite Sharpe
        print(f"\n✔ Selected by Sharpe (best Sharpe={best_score_sharpe:.6f})")
        return best_params_sharpe
    else:
        # All Sharpe‐ratios were NaN → fallback to net_pnl
        print(
            f"\n⚠ All Sharpe‐ratios = NaN → falling back to highest net_pnl ({best_score_pnl:.2f})"
        )
        return best_params_pnl or {}


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
    print(
        f"▶ Starting grid search over {len(symbols)} symbols from {args.start} to {args.end}...\n"
    )
    best = optimize_hyperparams(None, symbols, data_cfg, param_grid, metric="sharpe")

    # Write results
    with open("best_hyperparams.json", "w") as f:
        json.dump(best, f, indent=2)

    print(f"\n✔ Best hyperparameters: {best}")


if __name__ == "__main__":
    main()
