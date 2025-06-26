"""Core backtesting logic and utilities."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from alpaca.common.exceptions import APIError
from tenacity import RetryError

from data_fetcher import DataFetchError, get_historical_data

from .config import CACHE_DIR, DEFAULT_SLIPPAGE, TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""

    equity: pd.Series
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float

    @property
    def net_pnl(self) -> float:
        if self.equity.empty:
            return 0.0
        return float(self.equity.iloc[-1] - self.equity.iloc[0])

    def to_dict(self) -> dict:
        return {
            "cumulative_return": self.cumulative_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "net_pnl": self.net_pnl,
        }


def load_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load or fetch daily data for ``symbol`` between ``start`` and ``end``."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_fname = os.path.join(CACHE_DIR, f"cache_{symbol}_{start}_{end}.csv")
    if os.path.exists(cache_fname):
        try:
            return pd.read_csv(cache_fname, index_col=0, parse_dates=True)
        except (OSError, pd.errors.ParserError, ValueError):
            try:
                os.remove(cache_fname)
            except OSError:
                pass

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
        except (APIError, DataFetchError, RetryError) as exc:
            if attempt < 3:
                logger.warning(
                    "Failed to fetch %s (attempt %s/3): %s – sleeping 2s",
                    symbol,
                    attempt,
                    exc,
                )
                time.sleep(2)
            else:
                logger.error("Final attempt failed for %s", symbol)
    try:
        df_final.to_csv(cache_fname)
    except OSError:
        pass
    time.sleep(1)
    return df_final


def run_backtest(
    symbols: list[str],
    start: str,
    end: str,
    params: dict[str, float],
) -> BacktestResult:
    """Execute a simple backtest over daily bars."""
    data: dict[str, pd.DataFrame] = {}
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    for sym in symbols:
        df_sym = load_price_data(sym, start, end)
        missing = required_cols - set(df_sym.columns)
        if missing:
            logger.warning("Missing required columns for %s: %s", sym, missing)
            continue
        if not df_sym.empty:
            df_sym["ret"] = df_sym["Close"].pct_change(fill_method=None).fillna(0)
        data[sym] = df_sym

    cash = 100000.0
    positions = {s: 0 for s in symbols}
    entry_price = {s: 0.0 for s in symbols}
    peak_price = {s: 0.0 for s in symbols}
    portfolio: list[float] = []

    dates = pd.date_range(start, end, freq="B")
    for d in dates:
        for sym, df in data.items():
            if d not in df.index:
                continue
            price = df.loc[d, "Open"]
            ret = df.loc[d, "ret"]
            if positions[sym] == 0:
                if (not np.isnan(ret)) and ret > params["BUY_THRESHOLD"] and cash > 0:
                    qty = int((cash * params["SCALING_FACTOR"]) / price)
                    if qty > 0:
                        cost = qty * price * (1 + params.get("LIMIT_ORDER_SLIPPAGE", DEFAULT_SLIPPAGE))
                        cash -= cost
                        positions[sym] += qty
                        entry_price[sym] = price
                        peak_price[sym] = price
            else:
                peak_price[sym] = max(peak_price[sym], price)
                drawdown = (price - peak_price[sym]) / peak_price[sym]
                gain = (price - entry_price[sym]) / entry_price[sym]
                if gain >= params["TAKE_PROFIT_FACTOR"] or abs(drawdown) >= params["TRAILING_FACTOR"]:
                    cash += positions[sym] * price * (1 - params.get("LIMIT_ORDER_SLIPPAGE", DEFAULT_SLIPPAGE))
                    positions[sym] = 0
                    entry_price[sym] = 0
                    peak_price[sym] = 0
        total_value = cash
        for sym, df in data.items():
            if d in df.index:
                total_value += positions[sym] * df.loc[d, "Close"]
        portfolio.append(total_value)

    equity = pd.Series(portfolio, index=dates)
    if equity.empty:
        return BacktestResult(equity, 0.0, float("nan"), 0.0)

    pct = equity.pct_change(fill_method=None).dropna()
    if pct.empty or pct.std() == 0:
        sharpe = float("nan")
    else:
        sharpe = (pct.mean() / pct.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)

    cumulative_return = equity.iloc[-1] / equity.iloc[0] - 1
    drawdown = ((equity / equity.cummax()) - 1).min()
    return BacktestResult(equity, float(cumulative_return), float(sharpe), abs(float(drawdown)))
