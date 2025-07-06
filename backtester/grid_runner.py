"""Utilities for running parameter grid searches in parallel.

Example:
    >>> param_list = [{"BUY_THRESHOLD": 0.2}, {"BUY_THRESHOLD": 0.3}]
    >>> run_grid_search(param_list, ["AAPL"], "2024-01-01", "2024-02-01")
"""

from __future__ import annotations

import logging
import os
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import Any, Iterable

from .core import BacktestResult, run_backtest

logger = logging.getLogger(__name__)


def _run(params_symbols_start_end: tuple[dict[str, float], list[str], str, str]):
    params, symbols, start, end = params_symbols_start_end
    res = run_backtest(symbols, start, end, params)
    return params, res


def run_grid_search(
    param_list: Iterable[dict[str, float]],
    symbols: list[str],
    start: str,
    end: str,
    metric: str = "sharpe_ratio",
    top_n: int = 3,
) -> list[tuple[dict[str, float], BacktestResult]]:
    """Run a grid search over ``param_list`` in parallel."""
    tasks = [(p, symbols, start, end) for p in param_list]
    workers = min(len(tasks), cpu_count() or 1)
    if os.getenv("BACKTEST_SERIAL") == "1" or workers == 1:
        results = [_run(t) for t in tasks]
    else:
        with Pool(processes=1) as pool:  # AI-AGENT-REF: single worker for consistency
            results = pool.map(_run, tasks)

    sort_key = {
        "sharpe_ratio": lambda pr: pr[1].sharpe_ratio,
        "cumulative_return": lambda pr: pr[1].cumulative_return,
        "net_pnl": lambda pr: pr[1].net_pnl,
    }.get(metric, lambda pr: pr[1].sharpe_ratio)
    ranked = sorted(results, key=sort_key, reverse=True)
    return ranked[:top_n]


def optimize_hyperparams(
    symbols: list[str],
    backtest_data: dict[str, Any],
    param_grid: dict[str, Iterable[float]],
    metric: str = "sharpe_ratio",
) -> dict[str, float]:
    """Grid search returning the best parameter set by ``metric``."""
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*param_grid.values())]
    ranked = run_grid_search(combos, symbols, backtest_data["start"], backtest_data["end"], metric=metric, top_n=1)
    if ranked:
        best_params, best_res = ranked[0]
        logger.info(
            "Selected config with %s=%.4f", metric, getattr(best_res, metric, float("nan"))
        )
        return best_params
    logger.warning("No results returned from grid search")
    return {}

# AI-AGENT-REF: additional grid tests for indicator triggers and scaling


__all__ = ["run_grid_search", "optimize_hyperparams"]

