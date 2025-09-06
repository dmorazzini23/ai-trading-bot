"""
Momentum strategy leveraging price return over a lookback window.

This module provides a simple long-only momentum strategy used in tests and
example workflows.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from ai_trading.logging import logger
from ..core.enums import RiskLevel
from ..core.interfaces import OrderSide
from ai_trading.config.profiles import load_strategy_profile, lookup_overrides
from .base import BaseStrategy, StrategySignal

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

pd: Any | None = None

class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy."""

    def __init__(self, strategy_id: str='momentum', name: str='Simple Momentum Strategy', risk_level: RiskLevel=RiskLevel.MODERATE, *, lookback: int=50, threshold: float=0.0):
        super().__init__(strategy_id, name, risk_level)
        self.lookback = lookback
        self.threshold = threshold
        # Lightweight guard: skip recomputation when input series lengths unchanged
        self._last_len_by_symbol: dict[str, int] = {}
        self._guard_skips = 0
        self._guard_attempts = 0
        self._guard_last_summary = 0.0

    def generate_signals(self, market_data: dict) -> list[StrategySignal]:
        global pd
        if pd is None:  # heavy import; keep global for monkeypatching
            import pandas as _pd
            pd = _pd
        """
        Generate momentum-based long-only signals using price changes over a
        lookback period. Accepts market_data with keys:
          - 'symbols': list[str] (optional)
          - 'prices': dict[str, pd.Series]
        If 'symbols' is missing, derive from the 'prices' keys.
        """
        lookback = int(getattr(self, 'lookback', 50))
        threshold = float(getattr(self, 'threshold', 0.0))
        signals: list[StrategySignal] = []
        symbols = market_data.get('symbols') or list(market_data.get('prices', {}).keys())
        if not symbols:
            return signals
        prof = load_strategy_profile()
        for symbol in symbols:
            if prof:
                ov = lookup_overrides(prof, symbol, 'momentum')
                try:
                    if 'lookback' in ov:
                        lookback = int(ov['lookback'])
                    if 'threshold' in ov:
                        threshold = float(ov['threshold'])
                except (ValueError, TypeError, KeyError):
                    pass
            prices = market_data.get('prices', {}).get(symbol)
            if prices is None:
                continue
            # Skip work if no new bars
            try:
                n = len(prices)
                if self._last_len_by_symbol.get(symbol) == n:
                    logger.debug("MOMENTUM_GUARD_SKIP", extra={"symbol": symbol, "n": n})
                    self._guard_skips += 1
                    self._guard_attempts += 1
                    continue
                self._last_len_by_symbol[symbol] = n
                self._guard_attempts += 1
            except (TypeError, AttributeError):
                pass
            if len(prices) <= lookback + 1:
                logger.warning('Insufficient data for %s', symbol)
                continue
            p_now = prices.iloc[-1]
            p_then = prices.iloc[-1 - lookback]
            if pd.isna(p_now) or pd.isna(p_then):
                continue
            mom = p_now / p_then - 1.0
            if not np.isfinite(mom):
                continue
            if mom > threshold:
                strength = float(min(max(mom, 0.0), 1.0))
                signals.append(StrategySignal(symbol=symbol, side=OrderSide.BUY, strength=strength, confidence=float(min(1.0, 0.5 + strength)), metadata={'lookback': lookback, 'momentum': float(mom)}))
        # Periodic summary (once per ~60s)
        try:
            import time as _t
            now = _t.monotonic()
            if self._guard_last_summary == 0.0:
                self._guard_last_summary = now
            elif now - self._guard_last_summary >= 60.0:
                logger.info(
                    "STRATEGY_GUARD_SUMMARY",
                    extra={"strategy": "momentum", "skips": self._guard_skips, "attempts": self._guard_attempts},
                )
                self._guard_skips = 0
                self._guard_attempts = 0
                self._guard_last_summary = now
        except (ValueError, TypeError):
            pass
        return signals

    def generate(self, ctx) -> list[StrategySignal]:
        """Fetch close prices and generate signals.

        When the provided context lacks sufficient price history for the
        configured ``lookback`` window, a warning is logged and an empty list
        is returned.  This keeps the method lightweight for tests and example
        workflows that supply a simple ``ctx`` with ``tickers`` and a
        ``data_fetcher``.
        """
        fetcher = getattr(ctx, "data_fetcher", None)
        symbols = list(getattr(ctx, "tickers", []) or [])
        if not fetcher or not symbols:
            logger.warning("Insufficient data")
            return []

        prices: dict[str, Any] = {}
        for sym in symbols:
            df = fetcher.get_daily_df(ctx, sym)
            try:
                series = df["close"]
            except Exception:  # pragma: no cover - defensive
                logger.warning("Insufficient data")
                return []
            if df is None or len(series) <= self.lookback:
                logger.warning("Insufficient data")
                return []
            prices[sym] = series

        market_data = {"symbols": symbols, "prices": prices}
        return self.generate_signals(market_data)

    def calculate_position_size(self, signal: StrategySignal, portfolio_value: float, current_position: float=0) -> int:
        """Simple position sizing based on signal strength."""
        max_dollar_amount = portfolio_value * 0.05
        assumed_price = 100.0
        max_shares = int(max_dollar_amount / assumed_price)
        position_size = int(max_shares * signal.strength)
        return max(1, position_size)
