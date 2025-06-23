"""Moving average crossover trading strategy implementation."""

import logging
from typing import List

import pandas as pd

from strategies.base import Strategy, TradeSignal, asset_class_for

logger = logging.getLogger(__name__)


class MovingAverageCrossoverStrategy(Strategy):
    """Generate buy or sell signals based on SMA crossovers."""

    name = "moving_average_crossover"

    def __init__(self, short: int = 20, long: int = 50) -> None:
        self.short = short
        self.long = long

    def generate(self, ctx) -> List[TradeSignal]:
        signals: List[TradeSignal] = []
        tickers = getattr(ctx, "tickers", [])
        for sym in tickers:
            df = ctx.data_fetcher.get_daily_df(ctx, sym)
            if df is None or df.empty or len(df) < self.long:
                logger.warning(
                    "Insufficient data for %s; need >=%d rows",
                    sym,
                    self.long,
                )
                continue
            short_ma = df["close"].rolling(self.short).mean().iloc[-1]
            long_ma = df["close"].rolling(self.long).mean().iloc[-1]
            if pd.isna(short_ma) or pd.isna(long_ma):
                continue
            if short_ma > long_ma:
                side = "buy"
            elif short_ma < long_ma:
                side = "sell"
            else:
                continue
            confidence = abs(short_ma - long_ma) / long_ma
            signals.append(
                TradeSignal(
                    symbol=sym,
                    side=side,
                    confidence=float(confidence),
                    strategy=self.name,
                    asset_class=asset_class_for(sym),
                )
            )
        return signals
