"""
Public exports for the `ai_trading.position` package.
The canonical implementation lives in `ai_trading.position.market_regime`.
No fallbacks.
"""
from .market_regime import MarketRegime  # AI-AGENT-REF: expose MarketRegime

__all__ = ["MarketRegime"]

