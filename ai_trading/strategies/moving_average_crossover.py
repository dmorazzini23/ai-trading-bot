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
    _last_ts: 'pd.Timestamp | None' = None  # type: ignore[name-defined]
    _guard_skips: int = 0
    _guard_attempts: int = 0
    _guard_last_summary: float = 0.0

    def __init__(
        self,
        short_window: int | None = None,
        long_window: int | None = None,
        min_history: int | None = None,
        *,
        short: int | None = None,
        long: int | None = None,
    ) -> None:
        """Allow `short`/`long` kwargs mapping to window lengths.

        Parameters
        ----------
        short_window, long_window, min_history:
            Standard dataclass fields with defaults.
        short, long:
            Optional aliases for ``short_window`` and ``long_window`` respectively.
        """
        if short is not None and short_window is not None:
            raise TypeError("Specify only one of 'short' or 'short_window'")
        if long is not None and long_window is not None:
            raise TypeError("Specify only one of 'long' or 'long_window'")

        if short is not None:
            short_window = short
        if long is not None:
            long_window = long

        if short_window is None:
            short_window = type(self).short_window
        if long_window is None:
            long_window = type(self).long_window
        if min_history is None:
            min_history = type(self).min_history

        self.short_window = short_window
        self.long_window = long_window
        self.min_history = min_history
        self._last_ts = None
        self._guard_skips = 0
        self._guard_attempts = 0
        self._guard_last_summary = 0.0

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
        try:
            if getattr(df, 'index', None) is not None and len(df.index) > 0:
                last_ts = df.index[-1]
                if self._last_ts == last_ts:
                    logger.debug('SMA_GUARD_SKIP', extra={'symbol': sym, 'ts': str(last_ts)})
                    self._guard_skips += 1
                    self._guard_attempts += 1
                    return []
                self._last_ts = last_ts
                self._guard_attempts += 1
        except (AttributeError, IndexError, TypeError):
            pass
        if 'close' not in df.columns or len(df) < max(self.min_history, self.long_window + 2):
            return []
        short = df['close'].rolling(window=self.short_window, min_periods=self.short_window).mean()
        long = df['close'].rolling(window=self.long_window, min_periods=self.long_window).mean()
        action = self._latest_cross(short, long)
        if not action:
            return []
        try:
            import time as _t
            now = _t.monotonic()
            if self._guard_last_summary == 0.0:
                self._guard_last_summary = now
            elif now - self._guard_last_summary >= 60.0:
                logger.info('STRATEGY_GUARD_SUMMARY', extra={'strategy': 'sma_crossover', 'skips': self._guard_skips, 'attempts': self._guard_attempts})
                self._guard_skips = 0
                self._guard_attempts = 0
                self._guard_last_summary = now
        except (ValueError, TypeError):
            pass
        return [StrategySignal(symbol=sym, side=action)]
