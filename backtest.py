"""Legacy wrapper for the modular backtester."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv

from backtester import optimize_hyperparams as _optimize_hyperparams
from backtester import run_backtest as _run_backtest
from backtester.core import load_price_data as _load_price_data

load_dotenv(dotenv_path=".env", override=True)
logger = logging.getLogger(__name__)


def run_backtest_wrapper(symbols, start, end, params):
    """Wrapper returning dict results for legacy callers."""
    return _run_backtest(symbols, start, end, params).to_dict()


def optimize_hyperparams_wrapper(ctx, symbols, backtest_data, param_grid, metric="sharpe_ratio"):
    """Thin wrapper for compatibility with old scripts."""
    return _optimize_hyperparams(symbols, backtest_data, param_grid, metric=metric)


run_backtest = run_backtest_wrapper
optimize_hyperparams = optimize_hyperparams_wrapper
load_price_data = _load_price_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter optimizer")
    default_end = datetime.today().date()
    default_start = default_end - timedelta(days=30)
    parser.add_argument(
        "--symbols",
        required=False,
        default="AAPL,MSFT,SPY",
        help="Comma separated symbols (default: AAPL,MSFT,SPY)",
    )
    parser.add_argument(
        "--start",
        required=False,
        default=str(default_start),
        help=f"Backtest start date YYYY-MM-DD (default: {default_start})",
    )
    parser.add_argument(
        "--end",
        required=False,
        default=str(default_end),
        help=f"Backtest end date YYYY-MM-DD (default: {default_end})",
    )
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
    logger.info(
        "Starting grid search over %s symbols from %s to %s",
        len(symbols),
        args.start,
        args.end,
    )
    best = optimize_hyperparams_wrapper(None, symbols, data_cfg, param_grid, metric="sharpe_ratio")

    with open("best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    logger.info("Best hyperparameters: %s", best)


if __name__ == "__main__":
    main()
