from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ai_trading.core.enums import RiskLevel, TimeFrame
from ai_trading.order.types import OrderSide
from ai_trading.strategies.base import BaseStrategy, StrategyRegistry, StrategySignal
from ai_trading.strategies.multi_timeframe import (
    MultiTimeframeAnalyzer,
    MultiTimeframeSignal,
    SignalDirection,
    SignalStrength,
    TimeframeHierarchy,
)
from ai_trading.strategies.regime_detection import (
    MarketRegime,
    RegimeDetector,
    TrendStrength,
    VolatilityRegime,
)


pd = pytest.importorskip("pandas")


class _ToyStrategy(BaseStrategy):
    def __init__(self) -> None:
        super().__init__("toy", "Toy", RiskLevel.CONSERVATIVE)
        self._signals: list[StrategySignal] = []

    def generate_signals(self, _market_data: dict) -> list[StrategySignal]:
        return list(self._signals)


def test_strategy_signal_validation_registry_and_context_generation() -> None:
    signal = StrategySignal("AAPL", OrderSide.BUY, strength=1.5, confidence=-1.0)
    assert signal.side == "buy"
    assert signal.strength == 1.0
    assert signal.confidence == 0.0
    assert signal.score == 1.0

    strategy = _ToyStrategy()
    assert strategy.validate_signal(StrategySignal("AAPL", "buy", 0.5, 0.7)) is True
    assert strategy.validate_signal(StrategySignal("", "buy", 0.5, 0.7)) is False
    assert strategy.validate_signal(StrategySignal("AAPL", "buy")) is False
    assert strategy.generate(SimpleNamespace(market_data={"x": 1})) == []

    strategy.update_performance(0.10, True)
    strategy.update_performance(-0.05, False)
    assert strategy.get_performance_summary()["win_rate"] == 0.5
    assert strategy.max_drawdown == -0.05

    registry = StrategyRegistry()
    strategy._signals = [
        StrategySignal("AAPL", "buy", 0.5, 0.7),
        StrategySignal("MSFT", "buy", 0.5, 0.1),
    ]
    assert registry.register_strategy(strategy) is True
    assert registry.register_strategy(strategy) is False
    assert registry.activate_strategy("toy") is True
    generated = registry.generate_signals_from_active_strategies({})
    assert [item.symbol for item in generated] == ["AAPL"]
    assert generated[0].strategy_id == "toy"
    assert strategy.signals_generated == 1
    assert registry.deactivate_strategy("toy") is True
    assert registry.unregister_strategy("toy") is True


def test_multi_timeframe_combines_alignment_recommendation_and_history() -> None:
    hierarchy = TimeframeHierarchy()
    day = MultiTimeframeSignal(TimeFrame.DAY_1, SignalDirection.BULLISH, SignalStrength.VERY_STRONG, 0.9, "MA")
    hour = MultiTimeframeSignal(TimeFrame.HOUR_1, SignalDirection.BULLISH, SignalStrength.STRONG, 0.8, "MA")
    bear = MultiTimeframeSignal(TimeFrame.MINUTE_15, SignalDirection.BEARISH, SignalStrength.STRONG, 0.7, "RSI")

    assert hierarchy.is_higher_timeframe(TimeFrame.DAY_1, TimeFrame.HOUR_1) is True
    assert hierarchy.calculate_timeframe_score({TimeFrame.DAY_1: day}) == pytest.approx(day.score)

    analyzer = MultiTimeframeAnalyzer(primary_timeframes=[TimeFrame.DAY_1, TimeFrame.HOUR_1])
    signals = {TimeFrame.DAY_1: [day], TimeFrame.HOUR_1: [hour, bear]}
    combined = analyzer._combine_timeframe_signals("SPY", signals)
    alignment = analyzer._analyze_signal_alignment(signals)

    assert combined["signal_counts"] == {"bullish": 2, "bearish": 1, "neutral": 0, "total": 3}
    assert alignment["alignment_by_indicator"]["MA"] == 1.0
    assert analyzer._calculate_indicator_alignment({TimeFrame.DAY_1: day, TimeFrame.HOUR_1: bear}) == 0.5

    buy = analyzer._generate_trading_recommendation(
        {"overall_score": 2.5, "average_confidence": 0.8, "signal_counts": {"neutral": 0}},
        {"overall_alignment": 0.75},
    )
    weak = analyzer._generate_trading_recommendation(
        {"overall_score": -1.5, "average_confidence": 0.4, "signal_counts": {"neutral": 3, "bullish": 0, "bearish": 1}},
        {"overall_alignment": 0.3},
    )
    assert buy["action"] == "BUY"
    assert weak["action"] == "WEAK_SELL"
    assert weak["risk_level"] == "high"
    assert "Many neutral signals indicate uncertainty" in weak["warnings"]

    analyzer._update_signal_history("SPY", {TimeFrame.DAY_1: [day]})
    analyzer._update_signal_history("SPY", {TimeFrame.DAY_1: [hour]})
    trend = analyzer.get_signal_trend("SPY")
    assert trend["periods_analyzed"] == 2
    assert analyzer.get_current_signals("MISSING") == {}


def test_multi_timeframe_indicator_generation_and_regime_helpers() -> None:
    close = np.linspace(100.0, 130.0, 80)
    frame = pd.DataFrame(
        {
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": [1000.0] * 79 + [3000.0],
        }
    )
    analyzer = MultiTimeframeAnalyzer(primary_timeframes=[TimeFrame.DAY_1])

    result = analyzer.analyze_symbol("SPY", {TimeFrame.DAY_1: frame.copy()})
    assert result["symbol"] == "SPY"
    assert result["signal_count"] == 1
    assert len(result["timeframe_signals"][TimeFrame.DAY_1]) >= 5

    detector = RegimeDetector()
    assert detector.detect_regime(frame.head(10)) == {"error": "Insufficient data"}
    assert detector._calculate_trend_strength(0.06, 0.11, 0.16, 0.06, 0.11) is TrendStrength.VERY_STRONG
    assert detector._calculate_volatility_percentile(0.4, 0.2) == 0.85
    assert detector._calculate_momentum_strength(75.0, 0.06, 0.06) == "very_strong"
    assert detector._determine_primary_regime(
        {"direction": "bearish", "returns_6m": -0.25},
        {"regime": VolatilityRegime.EXTREME_VOL, "current_vol": 0.6},
        {"state": "bearish"},
    ) is MarketRegime.CRISIS
    assert detector.get_regime_summary() == {"error": "No regime data available"}

    analysis = detector.detect_regime(frame.copy(), {"vix": 55, "put_call_ratio": 1.3})
    assert analysis["primary_regime"] in set(MarketRegime)
    assert detector.get_regime_summary()["regime_history_length"] == 1
    assert detector.get_regime_recommendations()["strategy_type"]
