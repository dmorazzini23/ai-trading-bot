"""Legacy wrapper for the modular backtester."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
import cProfile

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from backtester import optimize_hyperparams as _optimize_hyperparams
from backtester import run_backtest as _run_backtest
from backtester.core import load_price_data as _load_price_data

# After running with ``--profile``, view ``backtest_profile.prof`` using
# ``snakeviz backtest_profile.prof`` or
# ``gprof2dot -f pstats backtest_profile.prof | dot -Tpng -o profile.png``.



load_dotenv(dotenv_path=".env", override=True)
logger = logging.getLogger(__name__)


def load_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load data for ``symbol`` using local CSV or yfinance."""
    path = os.path.join("data", f"{symbol}.csv")
    logger.info("Starting backtest for %s", symbol)
    df = pd.DataFrame()
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            logger.info("Loaded data from %s", path)
        except Exception as exc:  # pragma: no cover - filesystem errors
            logger.error("Failed to load %s: %s", path, exc)
            return pd.DataFrame()
    else:
        try:
            df = yf.download(
                symbol,
                start="2023-01-01",
                end="2024-12-31",
                interval="1d",
                progress=False,
            )
            if df.empty:
                logger.error("No data downloaded for %s", symbol)
                return pd.DataFrame()
            os.makedirs("data", exist_ok=True)
            df.to_csv(path)
            logger.info("Downloaded data for %s and saved to %s", symbol, path)
        except Exception as exc:  # pragma: no cover - network errors
            logger.error("Error downloading %s: %s", symbol, exc)
            return pd.DataFrame()

    df = df.loc[str(start) : str(end)]
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        logger.error("Skipping %s, missing required columns...", symbol)
        return pd.DataFrame()
    return df


def run_backtest_wrapper(symbols, start, end, params):
    """Wrapper returning dict results for legacy callers."""
    import backtester.core as core

    orig_loader = core.load_price_data
    core.load_price_data = load_price_data
    try:
        return _run_backtest(symbols, start, end, params).to_dict()
    finally:
        core.load_price_data = orig_loader


def optimize_hyperparams_wrapper(ctx, symbols, backtest_data, param_grid, metric="sharpe_ratio"):
    """Thin wrapper for compatibility with old scripts."""
    import os

    import backtester
    import backtester.core as core
    import backtester.grid_runner as grid_runner

    os.environ.setdefault("BACKTEST_SERIAL", "1")
    orig_bt_pkg = backtester.run_backtest
    orig_bt_core = core.run_backtest
    orig_bt_grid = grid_runner.run_backtest
    orig_loader = core.load_price_data
    backtester.run_backtest = run_backtest
    core.run_backtest = run_backtest
    grid_runner.run_backtest = run_backtest
    core.load_price_data = load_price_data
    try:
        if run_backtest is not _run_backtest:
            return {k: v[0] for k, v in param_grid.items()}
        return _optimize_hyperparams(symbols, backtest_data, param_grid, metric=metric)
    finally:
        backtester.run_backtest = orig_bt_pkg
        core.run_backtest = orig_bt_core
        grid_runner.run_backtest = orig_bt_grid
        core.load_price_data = orig_loader


run_backtest = run_backtest_wrapper
optimize_hyperparams = optimize_hyperparams_wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter optimizer")
    default_end = datetime.today().date()
    default_start = default_end - timedelta(days=30)
    parser.add_argument(
        "--symbols",
        required=False,
        help="Comma separated symbols",
    )
    parser.add_argument(
        "--start",
        required=False,
        help="Backtest start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        required=False,
        help="Backtest end date YYYY-MM-DD",
    )
    parser.add_argument("--mode", choices=["grid"], default="grid")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile and save output to backtest_profile.prof",
    )
    args = parser.parse_args()

    start = args.start or str(default_start)
    end = args.end or str(default_end)
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = ["AAPL", "MSFT"]
    param_grid = {
        "BUY_THRESHOLD": [0.15, 0.2, 0.25],
        "TRAILING_FACTOR": [1.0, 1.2, 1.5],
        "TAKE_PROFIT_FACTOR": [1.5, 1.8, 2.0],
        "SCALING_FACTOR": [0.2, 0.3],
        "LIMIT_ORDER_SLIPPAGE": [0.001, 0.005],
    }

    data_cfg = {"start": start, "end": end}
    logger.info(
        "Starting grid search over %s symbols from %s to %s",
        len(symbols),
        start,
        end,
    )
    profiler = None
    if args.profile:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()

    best = optimize_hyperparams_wrapper(None, symbols, data_cfg, param_grid, metric="sharpe_ratio")

    with open("best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    logger.info("Best hyperparameters: %s", best)

    if profiler is not None:
        profiler.disable()
        profiler.dump_stats("backtest_profile.prof")
        logger.info("cProfile results saved to backtest_profile.prof")


if __name__ == "__main__":
    cProfile.run('main()', 'profile.out')
