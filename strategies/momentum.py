import logging
from typing import List

import pandas as pd

from .base import Strategy, TradeSignal, asset_class_for
from logger import logger

# Logger added to allow warning messages when data is insufficient


class MomentumStrategy(Strategy):
    """Simple momentum strategy using recent returns."""

    name = "momentum"

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback

    def generate(self, ctx) -> List[TradeSignal]:
        signals: List[TradeSignal] = []
        tickers = getattr(ctx, "tickers", [])
        for sym in tickers:
            df = ctx.data_fetcher.get_daily_df(ctx, sym)
            # Skip ticker when no data or not enough history to compute momentum
            if df is None or df.empty or len(df) <= self.lookback:
                logger.warning("Insufficient data for %s; expected >%d rows", sym, self.lookback)
                continue
            # Data is safe to use; compute lookback return
            ret = df["close"].pct_change(self.lookback).iloc[-1]
            if pd.isna(ret):
                continue
            side = "buy" if ret > 0 else "sell"
            signals.append(
                TradeSignal(
                    symbol=sym,
                    side=side,
                    confidence=abs(ret),
                    strategy=self.name,
                    asset_class=asset_class_for(sym),
                )
            )
        return signals
