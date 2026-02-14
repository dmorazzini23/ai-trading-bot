from ai_trading.strategies.base import StrategySignal, BaseStrategy
from ai_trading.core.enums import RiskLevel


class _DummyStrategy(BaseStrategy):
    def __init__(self) -> None:
        super().__init__(strategy_id="dummy", name="Dummy", risk_level=RiskLevel.MODERATE)

    def generate_signals(self, market_data: dict) -> list[StrategySignal]:
        return []


def test_signal_canonical_side_and_score():
    sig = StrategySignal("AAPL", "BUY", strength=0.7, confidence=0.8)
    assert sig.side == "buy"
    assert sig.score == 0.7
    sig2 = StrategySignal("AAPL", "sell", strength=0.4, confidence=0.5)
    assert sig2.score == -0.4
    sig3 = StrategySignal("AAPL", "hold", strength=0.9, confidence=0.9)
    assert sig3.score == 0.0


def test_signal_requires_explicit_strength_confidence():
    strat = _DummyStrategy()
    sig = StrategySignal("AAPL", "buy")
    assert strat.validate_signal(sig) is False
