"""Backtester package exposing main interfaces."""

from .core import BacktestResult, load_price_data, run_backtest
from .grid_runner import optimize_hyperparams, run_grid_search
from .logger import MetricsLogger
from .plot import plot_drawdown, plot_equity_curve

__all__ = [
    "BacktestResult",
    "run_backtest",
    "load_price_data",
    "run_grid_search",
    "optimize_hyperparams",
    "MetricsLogger",
    "plot_equity_curve",
    "plot_drawdown",
]
