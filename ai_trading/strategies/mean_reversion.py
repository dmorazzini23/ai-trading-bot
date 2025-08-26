from __future__ import annotations
from typing import TYPE_CHECKING
from ai_trading.logging import logger as log
from .base import StrategySignal

if TYPE_CHECKING:  # pragma: no cover - heavy import only for typing
    import pandas as pd

class MeanReversionStrategy:

    def __init__(self, lookback: int=5, z_entry: float=1.0, *, z: float | None=None, **_: dict) -> None:
        if z is not None:
            z_entry = z
        self.lookback = lookback
        self.z_entry = z_entry

    def _latest_stats(self, series: 'pd.Series', window: int):
        import pandas as pd  # heavy import; keep local
        if len(series) < max(3, window):
            log.warning('mean_reversion: insufficient data')
            return (None, None)
        roll = series.rolling(window=window, min_periods=window)
        mean = roll.mean().iloc[-1]
        std = roll.std(ddof=0).iloc[-1]
        if pd.isna(mean) or pd.isna(std) or std == 0:
            log.warning('mean_reversion: invalid stats')
            return (None, None)
        return (mean, std)

    def generate(self, ctx) -> list[StrategySignal]:
        if not getattr(ctx, 'tickers', None):
            return []
        sym = ctx.tickers[0]
        df = ctx.data_fetcher.get_daily_df(ctx, sym)
        if 'close' not in df.columns:
            return []
        mean, std = self._latest_stats(df['close'], self.lookback)
        if mean is None:
            return []
        px = float(df['close'].iloc[-1])
        z = (px - mean) / std
        if z <= -self.z_entry:
            return [StrategySignal(symbol=sym, side='buy', strength=abs(z))]
        if z >= self.z_entry:
            return [StrategySignal(symbol=sym, side='sell', strength=abs(z))]
        return []