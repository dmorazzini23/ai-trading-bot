"""Plotting helpers for backtest results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(equity: pd.Series, output_path: str | None = None):
    """Plot the equity curve and optionally save to ``output_path``."""
    fig, ax = plt.subplots()
    equity.plot(ax=ax)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


def plot_drawdown(equity: pd.Series, output_path: str | None = None):
    """Plot the drawdown curve."""
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    fig, ax = plt.subplots()
    drawdown.plot(ax=ax)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig
