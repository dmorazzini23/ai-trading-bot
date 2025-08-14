import logging
import pandas as pd


class MeanReversionStrategy:
    """Simple mean reversion strategy for tests."""
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.logger = logging.getLogger(__name__)

    def generate(self, ctx):
        df = ctx.data_fetcher.get_daily_df(ctx, ctx.tickers[0])
        if len(df) < self.lookback:
            self.logger.warning("insufficient history")
            return []
        roll = df['close'].rolling(self.lookback)
        mean = roll.mean().iloc[-1]
        std = roll.std().iloc[-1]
        if pd.isna(mean) or pd.isna(std) or std == 0:
            self.logger.warning("invalid rolling stats")
            return []
        price = df['close'].iloc[-1]
        if price < mean - std:
            return ['buy']
        if price > mean + std:
            return ['sell']
        return []

__all__ = ['MeanReversionStrategy']
