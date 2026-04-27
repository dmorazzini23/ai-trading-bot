from ai_trading.strategies.base import BaseStrategy, StrategySignal

__test__ = False


class TestStrategy(BaseStrategy):
    """Simple strategy used for unit tests."""

    def generate_signals(self, market_data: dict) -> list[StrategySignal]:
        """Return no signals for provided market data."""
        return []
