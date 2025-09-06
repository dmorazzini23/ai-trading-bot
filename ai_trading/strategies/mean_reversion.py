from __future__ import annotations
from typing import TYPE_CHECKING
from ai_trading.logging import logger as log
from .base import StrategySignal
from ai_trading.config.profiles import load_strategy_profile, lookup_overrides

if TYPE_CHECKING:  # pragma: no cover - heavy import only for typing
    import pandas as pd

class MeanReversionStrategy:

    def __init__(self, lookback: int=5, z_entry: float=1.0, *, z: float | None=None, **_: dict) -> None:
        if z is not None:
            z_entry = z
        self.lookback = lookback
        self.z_entry = z_entry
        self._last_ts = None
        self._guard_skips = 0
        self._guard_attempts = 0
        self._guard_last_summary = 0.0

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
        try:
            if getattr(df, 'index', None) is not None and len(df.index) > 0:
                last_ts = df.index[-1]
                if self._last_ts == last_ts:
                    log.debug('MEAN_REVERSION_GUARD_SKIP', extra={'symbol': sym, 'ts': str(last_ts)})
                    self._guard_skips += 1
                    self._guard_attempts += 1
                    return []
                self._last_ts = last_ts
                self._guard_attempts += 1
        except (AttributeError, IndexError, TypeError):
            pass
        if 'close' not in df.columns:
            return []
        # Optional profile overrides
        prof = load_strategy_profile()
        lookback = int(getattr(self, 'lookback', 5))
        z_entry = float(getattr(self, 'z_entry', self.z_entry)) if hasattr(self, 'z_entry') else float(getattr(self, 'z_entry', 1.0))
        if prof:
            ov = lookup_overrides(prof, sym, 'mean_reversion')
            try:
                lookback = int(ov.get('lookback', lookback))
                z_entry = float(ov.get('z_entry', z_entry))
            except (ValueError, TypeError, KeyError):
                pass
        mean, std = self._latest_stats(df['close'], lookback)
        if mean is None:
            log.warning('mean_reversion: invalid rolling')
            return []
        px = float(df['close'].iloc[-1])
        z = (px - mean) / std
        if z <= -z_entry:
            return [StrategySignal(symbol=sym, side='buy', strength=abs(z))]
        if z >= z_entry:
            return [StrategySignal(symbol=sym, side='sell', strength=abs(z))]
        try:
            import time as _t
            now = _t.monotonic()
            if self._guard_last_summary == 0.0:
                self._guard_last_summary = now
            elif now - self._guard_last_summary >= 60.0:
                log.info('STRATEGY_GUARD_SUMMARY', extra={'strategy': 'mean_reversion', 'skips': self._guard_skips, 'attempts': self._guard_attempts})
                self._guard_skips = 0
                self._guard_attempts = 0
                self._guard_last_summary = now
        except (ValueError, TypeError):
            pass
        return []
