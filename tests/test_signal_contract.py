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
    sig4 = StrategySignal("AAPL", "sell_short", strength=0.5, confidence=0.7)
    assert sig4.side == "sell_short"
    assert sig4.score == -0.5
    assert sig4.is_sell
    assert sig4.is_sell_short


def test_signal_requires_explicit_strength_confidence():
    strat = _DummyStrategy()
    sig = StrategySignal("AAPL", "buy")
    assert strat.validate_signal(sig) is False


def test_signal_neutralizes_non_finite_strength_confidence():
    sig = StrategySignal("AAPL", "buy", strength=float("nan"), confidence=float("inf"))

    assert sig.strength == 0.0
    assert sig.confidence == 0.0
    assert sig.weighted_strength == 0.0
    assert sig.score == 0.0


def test_validate_signal_rejects_non_finite_values():
    strat = _DummyStrategy()
    sig = StrategySignal("AAPL", "buy", strength=0.7, confidence=0.8)

    sig.strength = float("nan")
    assert strat.validate_signal(sig) is False

    sig.strength = 0.7
    sig.confidence = float("inf")
    assert strat.validate_signal(sig) is False

    sig.confidence = 0.8
    sig.risk_score = float("nan")
    assert strat.validate_signal(sig) is False
