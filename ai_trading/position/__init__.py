# Public test-facing API surface
from .market_regime import MarketRegime, detect_market_regime
from .api import Allocation, Allocator

__all__ = [
    "MarketRegime",
    "detect_market_regime",
    "Allocation",
    "Allocator",
]

# Note: We purposely export symbols from ai_trading.position so legacy test imports like
# from ai_trading.position import MarketRegime collect cleanly. No fallback shims; just a tiny, stable surface.
