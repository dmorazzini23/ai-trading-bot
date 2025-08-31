"""
Momentum strategy leveraging price return over a lookback window.

This module provides a simple long-only momentum strategy used in tests and
example workflows.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from ai_trading.logging import logger
from ..core.enums import RiskLevel
from ..core.interfaces import OrderSide
from ai_trading.config.profiles import load_strategy_profile, lookup_overrides
from .base import BaseStrategy, StrategySignal

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy."""

    def __init__(self, strategy_id: str='momentum', name: str='Simple Momentum Strategy', risk_level: RiskLevel=RiskLevel.MODERATE, *, lookback: int=50, threshold: float=0.0):
        super().__init__(strategy_id, name, risk_level)
        self.lookback = lookback
        self.threshold = threshold
        # Lightweight guard: skip recomputation when input series lengths unchanged
        self._last_len_by_symbol: dict[str, int] = {}

    def generate_signals(self, market_data: dict) -> list[StrategySignal]:
        import pandas as pd  # heavy import; keep local
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
                except Exception:
                    pass
            prices = market_data.get('prices', {}).get(symbol)
            if prices is None:
                continue
            # Skip work if no new bars
            try:
                n = len(prices)
                if self._last_len_by_symbol.get(symbol) == n:
                    continue
                self._last_len_by_symbol[symbol] = n
            except Exception:
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
        return signals

    def generate(self, ctx) -> list[StrategySignal]:
        return super().generate(ctx)

    def calculate_position_size(self, signal: StrategySignal, portfolio_value: float, current_position: float=0) -> int:
        """Simple position sizing based on signal strength."""
        max_dollar_amount = portfolio_value * 0.05
        assumed_price = 100.0
        max_shares = int(max_dollar_amount / assumed_price)
        position_size = int(max_shares * signal.strength)
        return max(1, position_size)
