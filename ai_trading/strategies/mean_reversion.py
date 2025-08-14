"""Minimal mean reversion strategy used in tests."""

import logging
from typing import Any

import pandas as pd

from .base import StrategySignal
from ai_trading.core.enums import OrderSide


class MeanReversionStrategy:
    """Simple mean reversion strategy for regression tests."""

    def __init__(self, lookback: int = 20, z: float = 1.0, **_: Any):
        self.lookback = lookback
        self.z = z
        self.logger = logging.getLogger(__name__)

    def generate(self, ctx) -> list[StrategySignal]:
        df = ctx.data_fetcher.get_daily_df(ctx, ctx.tickers[0])
        if len(df) < self.lookback:
            self.logger.warning("insufficient history")
            return []
        roll = df["close"].rolling(self.lookback)
        mean = roll.mean().iloc[-1]
        std = roll.std().iloc[-1]
        if pd.isna(mean) or pd.isna(std) or std == 0:
            self.logger.warning("invalid rolling stats")
            return []
        price = df["close"].iloc[-1]
        sym = ctx.tickers[0]
        if price < mean - self.z * std:
            return [StrategySignal(sym, OrderSide.BUY, 1.0)]
        if price > mean + self.z * std:
            return [StrategySignal(sym, OrderSide.SELL, 1.0)]
        return []


__all__ = ["MeanReversionStrategy"]

