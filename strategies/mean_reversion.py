"""Mean reversion trading strategy implementation."""

import logging
import os
from typing import List

import pandas as pd

from strategies.base import Strategy, TradeSignal, asset_class_for
from utils import get_phase_logger
from indicators import mean_reversion_zscore

logger = get_phase_logger(__name__, "STRATEGY")


class MeanReversionStrategy(Strategy):
    """Simple mean reversion strategy using z-score."""

    name = "mean_reversion"

    def __init__(self, lookback: int = 20, z: float | None = None) -> None:
        self.lookback = lookback
        thresh = float(os.getenv("MEAN_REVERT_THRESHOLD", 0.75))
        self.z = thresh if z is None else z

    def generate(self, ctx) -> List[TradeSignal]:
        signals: List[TradeSignal] = []
        scores = {}
        tickers = getattr(ctx, "tickers", [])
        for sym in tickers:
            df = ctx.data_fetcher.get_daily_df(ctx, sym)
            # Ensure we have data and enough rows before doing rolling calculations
            if df is None or df.empty or len(df) < self.lookback:
                logger.warning(
                    "%s: insufficient data for lookback %s",
                    sym,
                    self.lookback,
                )
                # Skip signal generation when we don't have enough history
                continue

            close_series = df["close"].dropna()
            if len(close_series) < self.lookback:
                logger.warning("%s: no valid close prices", sym)
                continue
            zscores = mean_reversion_zscore(close_series, self.lookback)
            if zscores.empty:
                continue
            z = zscores.iloc[-1]
            if pd.isna(z):
                logger.warning("%s: invalid rolling statistics", sym)  # AI-AGENT-REF: clarify log message
                continue
            scores[sym] = float(z)
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
        if not signals:
            logger.info("No signals generated for mean_reversion. Scores: %s", scores)
        if not scores:
            logger.warning("mean_reversion received no candidate data")
            return []
        return signals
