from typing import List
import pandas as pd
from .base import Strategy, TradeSignal, asset_class_for

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
            if df is None or len(df) <= self.lookback:
                continue
            ma = df["close"].rolling(self.lookback).mean().iloc[-1]
            sd = df["close"].rolling(self.lookback).std().iloc[-1]
            if pd.isna(ma) or pd.isna(sd) or sd == 0:
                continue
            z = (df["close"].iloc[-1] - ma) / sd
            if z > self.z:
                signals.append(TradeSignal(symbol=sym, side="sell", confidence=abs(z), strategy=self.name, asset_class=asset_class_for(sym)))
            elif z < -self.z:
                signals.append(TradeSignal(symbol=sym, side="buy", confidence=abs(z), strategy=self.name, asset_class=asset_class_for(sym)))
        return signals
