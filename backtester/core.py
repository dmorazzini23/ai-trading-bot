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
from utils import validate_ohlcv

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
            df_cached = pd.read_csv(cache_fname, index_col=0, parse_dates=True)
            if validate_ohlcv(df_cached):
                return df_cached
            logger.error("Cached data for %s missing required columns", symbol)
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
    if not validate_ohlcv(df_final):
        logger.error("Fetched data for %s missing required columns", symbol)
        return pd.DataFrame()
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
        if not validate_ohlcv(df_sym):
            logger.error("Skipping %s due to missing columns", sym)
            continue
        if not df_sym.empty:
            df_sym["ret"] = df_sym["Close"].pct_change(fill_method=None).fillna(0)
            df_sym["sma_short"] = df_sym["Close"].rolling(20).mean()
            df_sym["sma_long"] = df_sym["Close"].rolling(50).mean()
        data[sym] = df_sym

    cash = 100000.0
    positions = {s: 0 for s in symbols}
    entry_price = {s: 0.0 for s in symbols}
    peak_price = {s: 0.0 for s in symbols}
    last_stop = {s: pd.Timestamp.min for s in symbols}
    portfolio: list[float] = []

    dates = pd.date_range(start, end, freq="B")
    for d in dates:
        for sym, df in data.items():
            if d not in df.index:
                continue
            price = df.loc[d, "Open"]
            ret = df.loc[d, "ret"]
            sma_s = df.loc[d, "sma_short"]
            sma_l = df.loc[d, "sma_long"]
            if positions[sym] == 0:
                buy_cond = (
                    (not np.isnan(ret))
                    and ret > params["BUY_THRESHOLD"]
                    and cash > 0
                    and d > last_stop[sym]
                )
                if pd.notna(sma_s) and pd.notna(sma_l):
                    buy_cond = buy_cond and sma_s > sma_l
                if buy_cond:
                    qty = int((cash * params["SCALING_FACTOR"]) / price)
                    if qty > 0:
                        cost = qty * price * (
                            1 + params.get("LIMIT_ORDER_SLIPPAGE", DEFAULT_SLIPPAGE)
                        )
                        cash -= cost
                        positions[sym] += qty
                        entry_price[sym] = price
                        peak_price[sym] = price
                        last_stop[sym] = pd.Timestamp.min
            else:
                peak_price[sym] = max(peak_price[sym], price)
                drawdown = (price - peak_price[sym]) / peak_price[sym]
                gain = (price - entry_price[sym]) / entry_price[sym]
                sell_signal = pd.notna(sma_s) and pd.notna(sma_l) and sma_s < sma_l
                stop_hit = abs(drawdown) >= params["TRAILING_FACTOR"]
                take_profit = gain >= params["TAKE_PROFIT_FACTOR"]
                if sell_signal or take_profit or stop_hit:
                    cash += positions[sym] * price * (
                        1 - params.get("LIMIT_ORDER_SLIPPAGE", DEFAULT_SLIPPAGE)
                    )
                    positions[sym] = 0
                    entry_price[sym] = 0
                    peak_price[sym] = 0
                    if stop_hit:
                        last_stop[sym] = d
                    else:
                        last_stop[sym] = pd.Timestamp.min
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
