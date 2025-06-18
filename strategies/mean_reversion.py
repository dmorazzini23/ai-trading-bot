import logging
from typing import List

import pandas as pd

from .base import Strategy, TradeSignal, asset_class_for
from logger import logger



class MeanReversionStrategy(Strategy):
    """Simple mean reversion strategy using z-score."""

    name = "mean_reversion"

    def __init__(self, lookback: int = 20, z: float = 2.0) -> None:
        self.lookback = lookback
        self.z = z

    def generate(self, ctx) -> List[TradeSignal]:
        signals: List[TradeSignal] = []
        tickers = getattr(ctx, "tickers", [])
        for sym in tickers:
            df = ctx.data_fetcher.get_daily_df(ctx, sym)
            # Ensure we have data and enough rows before doing rolling calculations
            if df is None or df.empty or len(df) < self.lookback:
                logger.warning(f"{sym}: insufficient data for lookback {self.lookback}")
                # Skip signal generation when we don't have enough history
                continue

            ma = df["close"].rolling(self.lookback).mean().iloc[-1]
            sd = df["close"].rolling(self.lookback).std().iloc[-1]
            # Validate rolling mean/std results before computing the z-score
            if pd.isna(ma) or pd.isna(sd) or sd == 0:
                logger.warning(f"{sym}: invalid rolling statistics")
                # Avoid division by zero or propagating NaNs
                continue

            last_close = df["close"].iloc[-1]
            # Guard against NaN closing prices before computing z-score
            if pd.isna(last_close):
                logger.warning(f"{sym}: last close is NaN")
                continue

            # Calculate the z-score of the latest close price
            z = (last_close - ma) / sd
            if z > self.z:
                signals.append(
                    TradeSignal(
                        symbol=sym,
                        side="sell",
                        confidence=abs(z),
                        strategy=self.name,
                        asset_class=asset_class_for(sym),
                    )
                )
            elif z < -self.z:
                signals.append(
                    TradeSignal(
                        symbol=sym,
                        side="buy",
                        confidence=abs(z),
                        strategy=self.name,
                        asset_class=asset_class_for(sym),
                    )
                )
        return signals
