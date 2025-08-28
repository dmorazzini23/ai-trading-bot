from __future__ import annotations
from typing import TYPE_CHECKING
from ai_trading.logging import get_logger
from dataclasses import dataclass
from .base import StrategySignal

if TYPE_CHECKING:  # pragma: no cover - heavy import for typing only
    import pandas as pd
logger = get_logger(__name__)

@dataclass
class MovingAverageCrossoverStrategy:
    """Simple SMA crossover strategy.

    Emits a single signal on the most recent bar:
      - BUY when short SMA crosses above long SMA
      - SELL when short SMA crosses below long SMA
    """
    short_window: int = 20
    long_window: int = 50
    min_history: int = 55

    def _latest_cross(self, short: 'pd.Series', long: 'pd.Series') -> str | None:
        import pandas as pd  # heavy import; keep local
        if len(short) < 2 or len(long) < 2:
            return None
        s_prev, s_now = (short.iloc[-2], short.iloc[-1])
        l_prev, l_now = (long.iloc[-2], long.iloc[-1])
        if pd.isna(s_prev) or pd.isna(s_now) or pd.isna(l_prev) or pd.isna(l_now):
            return None
        if s_prev <= l_prev and s_now > l_now:
            return 'buy'
        if s_prev >= l_prev and s_now < l_now:
            return 'sell'
        return None

    def generate(self, ctx) -> list[StrategySignal]:
        if not getattr(ctx, 'tickers', None):
            return []
        sym = ctx.tickers[0]
        df = ctx.data_fetcher.get_daily_df(ctx, sym)
        if 'close' not in df.columns or len(df) < max(self.min_history, self.long_window + 2):
            return []
        short = df['close'].rolling(window=self.short_window, min_periods=self.short_window).mean()
        long = df['close'].rolling(window=self.long_window, min_periods=self.long_window).mean()
        action = self._latest_cross(short, long)
        if not action:
            return []
        return [StrategySignal(symbol=sym, side=action)]
